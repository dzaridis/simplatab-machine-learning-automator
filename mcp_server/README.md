# Simplatab MCP server

Every Simplatab automator as [Model Context Protocol](https://modelcontextprotocol.io) tools, so that AI agents
can run validated machine learning experiments on their own: the agent **provides the data and the
configuration**; the server checks the data, runs the same pipelines as the Simplatab web application (K-fold or
hold-out validation, external test, explanations, exported models and validation splits) and returns the results.

| Automator id | Task | Train / Test |
|---|---|---|
| `tabular` | Binary or multiclass classification | `Train.csv`, `Test.csv` |
| `image-classification` | 2D images or 3D studies (one or more series) | zip or folder, one folder per class |
| `object-detection` | Boxes in 2D images or 3D volumes | zip or folder with COCO / YOLO / VOC / CSV / masks |
| `image-segmentation` | 2D or 3D masks, incl. the official nnU-Net v2 | zip or folder: `images/` + `masks/`, or nnU-Net raw |
| `time-series-forecasting` | Forecasting many series with covariates | `Train.csv`, `Test.csv` in long format |

An agent can also let the server **choose the automator from the data** (`inspect_data`, or `automator="auto"`): a CSV
with `ID`, `Time` and `Target` is a forecasting problem, a CSV with a `Target` of classes is tabular, class folders
of images are image classification, `images/` + `masks/` is segmentation, COCO/YOLO/VOC/CSV boxes are detection.

## Run it on your computer (Docker)

The recommended set-up is one container running in the background, which every agent session connects to. Runs
then continue when an agent disconnects, and several agents (or sessions) share the experiments.

1. Set your folders (absolute paths) in `mcp_server/.env`:
   ```bash
   cp mcp_server/.env.example mcp_server/.env     # SIMPLATAB_DATA: your datasets, SIMPLATAB_OUTPUT: experiments and models
   ```
2. Build and start it (from the repository root; the image copies the Simplatab pipelines from `Helpers/`, the
   existing sources are not changed):
   ```bash
   docker compose -f mcp_server/compose.yaml up -d --build                                    # CPU
   docker compose -f mcp_server/compose.yaml -f mcp_server/compose.gpu.yaml up -d --build     # NVIDIA GPU
   ```
   The GPU variant needs the NVIDIA driver and the
   [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
   `docker compose -f mcp_server/compose.yaml logs -f` shows the server, `... down` stops it (the experiments stay in
   `SIMPLATAB_OUTPUT`).
3. Connect the agent, by URL (Streamable HTTP, on this computer only):
   ```bash
   claude mcp add --transport http simplatab http://localhost:8000/mcp
   ```
   or over stdio, for clients that only start commands:
   ```bash
   claude mcp add simplatab -- docker exec -i simplatab-mcp /opt/mcp/bin/python -m simplatab_mcp
   ```
   Claude Desktop, Cursor and other clients (`mcpServers` JSON):
   ```json
   {"mcpServers": {"simplatab": {"url": "http://localhost:8000/mcp"}}}
   {"mcpServers": {"simplatab": {"command": "docker", "args": ["exec", "-i", "simplatab-mcp", "/opt/mcp/bin/python", "-m", "simplatab_mcp"]}}}
   ```

**Paths.** The data folder is mounted on `/data` and the output folder on `/workspace`. Agents can give either the
host paths (`/home/me/simplatab/data/study/Train.csv`, translated by the server) or the container paths
(`/data/study/Train.csv`, or `study/Train.csv`); results also give host paths (`materials_host`, `folder_host`), so
the trained models and figures can be opened directly on the computer. Data outside these folders is refused
(`upload_file` sends it instead).

**Without Compose**, the same container:
```bash
docker build -f mcp_server/Dockerfile -t simplatab-mcp .            # GPU: --build-arg DEVICE=gpu -t simplatab-mcp:gpu
docker run -d --init --name simplatab-mcp -p 127.0.0.1:8000:8000 --shm-size=4g \
    -v /home/me/simplatab/data:/data -v /home/me/simplatab/workspace:/workspace \
    -e SIMPLATAB_HOST_DATA_DIR=/home/me/simplatab/data -e SIMPLATAB_HOST_WORKSPACE_DIR=/home/me/simplatab/workspace \
    simplatab-mcp --transport http --host 0.0.0.0 --port 8000       # GPU: add --gpus all, image simplatab-mcp:gpu
```
A container per session also works (`docker run -i --rm ... simplatab-mcp` as the stdio command), but a run still
going when the session ends is stopped with the container (it is reported as interrupted the next time).

The HTTP transport has no authentication: the port is published on `127.0.0.1` only; do not expose it on a
network without an authenticating proxy.

### Without Docker

The server needs Python ≥ 3.10 with `mcp_server/requirements.txt`; the pipelines run in the Python 3.9 environment of
Simplatab (`requirements.txt` of the repository), given by `SIMPLATAB_WORKER_PYTHON`:
```bash
python3.12 -m venv mcp-venv && mcp-venv/bin/pip install -r mcp_server/requirements.txt
cd mcp_server
SIMPLATAB_WORKER_PYTHON=/path/to/simplatab-venv/bin/python ../mcp-venv/bin/python -m simplatab_mcp            # stdio
SIMPLATAB_WORKER_PYTHON=... ../mcp-venv/bin/python -m simplatab_mcp --transport http --port 8000 --workspace ~/simplatab-runs
```

| Variable | Default | Meaning |
|---|---|---|
| `SIMPLATAB_WORKSPACE` | `/workspace`, else `~/.simplatab-mcp` | experiments, uploads, examples |
| `SIMPLATAB_DATA_DIR` | `/data` | relative data paths are looked up here (then in the uploads and the workspace) |
| `SIMPLATAB_WORKER_PYTHON` | the server's Python | Python of the Simplatab pipelines |
| `SIMPLATAB_MAX_PARALLEL` | `1` | runs executed at the same time (the others wait in a queue) |
| `SIMPLATAB_HOST_DATA_DIR`, `SIMPLATAB_HOST_WORKSPACE_DIR` | (unset) | host folders mounted on the data folder and the workspace: host paths are translated both ways |
| `SIMPLATAB_ALLOW_ANY_PATH` | `0` | `1`: accept data paths outside the data folder, the uploads and the workspace |
| `SIMPLATAB_PRETRAINED` | `1` | `0`: networks without pretrained weights (offline tests) |

## Tools

| Tool | What it does |
|---|---|
| `list_automators` | The automators, their data and models |
| `inspect_data(train)` | **Which automator** fits a dataset (CSV columns, or the folders and files of a zip/folder), with the reason |
| `get_data_contract(automator, dim?)` | **The data contract**: layout, formats, column/folder rules, examples, the configuration fields (type, default, choices, range), the models (keys), metrics and output files |
| `get_server_info` | GPU or CPU, versions, workspace, data folders |
| `get_example_data(automator, variant)` | A ready-made dataset (2d/3d) and a configuration that runs in minutes on a CPU |
| `upload_file(filename, content_base64, append?)` | Sends a CSV or zip to the server (in chunks for large files) when it cannot read your files |
| `create_experiment(automator, train, test, name?)` | Copies/extracts the data and checks it as the web upload does: summary, **errors**, warnings, default configuration (`automator="auto"`: chosen from the data) |
| `start_experiment(experiment_id, config?, dry_run?)` | Validates the configuration (defaults for any field left out) and queues the run; `dry_run` only validates |
| `run_experiment(automator, train, test, config?)` | Both of the above in one call; a configuration error is returned with the experiment id, to fix and call `start_experiment` |
| `get_experiment(experiment_id)` | State, phase, progress, status of every model, last log lines |
| `list_experiments` | Every experiment of the workspace |
| `get_results(experiment_id)` | Test and validation metrics, best model, **validation splits**, trained models, every output file |
| `read_result_file(experiment_id, path)` | A CSV/JSON/log as text, a figure as an image |
| `get_log`, `cancel_experiment`, `delete_experiment` | Run log, stop a run, remove an experiment |

Resources: `simplatab://automators`, `simplatab://contracts/{automator}`, `simplatab://experiments/{experiment_id}`.
Prompt: `run_simplatab_experiment(train, test, goal)` walks an agent through a complete experiment.

### A session

```text
inspect_data("/home/me/simplatab/data/liver/Train.zip") -> {"suggested_automator": "image-segmentation", "candidates": [...]}
get_data_contract("image-segmentation")          -> layout images/ + masks/, config fields, 20 networks
create_experiment("image-segmentation", "/data/liver/Train.zip", "/data/liver/Test.zip")
    -> {"experiment_id": "20261002-...", "state": "ready", "summary": {"dim": 3, "classes": [...]}, "default_config": {...}}
start_experiment(id, {"models": ["nnunet_3d", "swinunetr"], "validation": "kfold", "k_folds": 5, "nnunet_epochs": 250})
get_experiment(id)   -> {"state": "running", "phase_name": "validation", "progress": 40, "models": [...]}
get_results(id)      -> {"best_model": "nnU-Net 3D full resolution", "test_metrics": [{"model": ..., "Dice": 0.91, ...}],
                         "splits": {"kind": "kfold", "folds": [{"fold": 1, "train": 80, "validation": 20}, ...]}, "files": [...]}
read_result_file(id, "Splits/splits.csv")
```

Every configuration error names the field and its valid values or range, and every data error says what to fix, so
an agent can correct and retry. Runs take minutes (tabular, CPU-friendly feature extraction) to hours (fine-tuning,
nnU-Net); they run one at a time and survive the agent's disconnection (poll `get_experiment` later).

## How it works

```
agent ──MCP──> server (Python 3.12: protocol, experiment queue)
                  └─ worker processes (Python 3.9: Simplatab pipelines, GPU) ── <workspace>/experiments/<id>/
                       input/ (the data)  Materials/ (outputs)  status.json  run.log  results.json
```

Several server processes may share one workspace (the HTTP server and `docker exec` sessions of the same
container): a workspace lock makes sure a queued run is started once.

`simplatab_mcp/`: `server.py` (tools), `jobs.py` (experiments, queue, cancellation), `worker.py` (commands run in
the pipelines' environment), `contracts.py` (data contracts and configuration schemas), `configs.py` (configuration
→ pipeline parameters, as the web forms), `datasets.py` (data preparation and checks, as the web uploads),
`tabular.py` (the tabular run of the web application), `detect.py` (automator from the data), `paths.py` (data
folders, host paths), `status.py`, `results.py`, `examples.py`.

## Tests

```bash
cd mcp_server
# worker: contracts, configurations, data checks, automator detection, host paths, a tabular run (Python 3.9 environment)
SIMPLATAB_PRETRAINED=0 /path/to/simplatab-venv/bin/python -m unittest tests.test_worker
# MCP client end to end (Python >= 3.10); SIMPLATAB_MCP_ALL=1 also runs every automator on its example (~15-20 min on a CPU)
SIMPLATAB_PRETRAINED=0 SIMPLATAB_MCP_ALL=1 SIMPLATAB_WORKER_PYTHON=/path/to/simplatab-venv/bin/python \
    /path/to/mcp-venv/bin/python -m unittest tests.test_server
# the Docker image, used over HTTP (host paths) and over stdio (docker exec)
python tests/docker_smoke.py simplatab-mcp
```
The CI runs the three on every pull request and before every release (jobs `mcp-test` and `mcp-docker`).
