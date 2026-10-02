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

## Run it

The server is a Docker image built from the repository root (it copies the Simplatab pipelines from `Helpers/`;
the existing sources are not changed):

```bash
docker build -f mcp_server/Dockerfile -t simplatab-mcp .                          # CPU
docker build -f mcp_server/Dockerfile --build-arg DEVICE=gpu -t simplatab-mcp:gpu .   # NVIDIA GPU (CUDA 12.8)
```

- **stdio** (the agent starts the server): `docker run -i --rm -v /path/to/data:/data -v simplatab-workspace:/workspace simplatab-mcp`
- **Streamable HTTP** (a shared server, `http://<host>:8000/mcp`):
  `docker run -d -p 8000:8000 -v /path/to/data:/data -v simplatab-workspace:/workspace simplatab-mcp --transport http --host 0.0.0.0 --port 8000`
- **GPU**: add `--gpus all --shm-size=4g` to `docker run` with the `simplatab-mcp:gpu` image (NVIDIA driver and
  [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host).

Volumes: `/data` holds your datasets (paths given to the tools are absolute or relative to it); `/workspace` keeps
the experiments (inputs, outputs, trained models) between runs of the container.

Which to choose: with stdio, the container lives as long as the agent's session (`--rm`), so a run still going when
the session ends is stopped (it is reported as interrupted the next time). For long trainings (fine-tuning, nnU-Net),
keep a server running with the HTTP transport: runs continue whether or not an agent is connected, and agents
can come back for the results.

The HTTP transport has no authentication: expose it only on a trusted network (or behind an authenticating proxy).
Data paths must be inside `/data`, the uploads or the workspace, so that agents cannot read other files of the server.

### Client configuration

Claude Code: `claude mcp add simplatab -- docker run -i --rm -v /path/to/data:/data -v simplatab-workspace:/workspace simplatab-mcp`

Claude Desktop, Cursor and other clients (`mcpServers` JSON):
```json
{
  "mcpServers": {
    "simplatab": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-v", "/path/to/data:/data", "-v", "simplatab-workspace:/workspace", "simplatab-mcp"]
    }
  }
}
```
A running HTTP server: `{"mcpServers": {"simplatab": {"url": "http://localhost:8000/mcp"}}}` (or
`claude mcp add --transport http simplatab http://localhost:8000/mcp`).

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
| `SIMPLATAB_ALLOW_ANY_PATH` | `0` | `1`: accept data paths outside the data folder, the uploads and the workspace |
| `SIMPLATAB_PRETRAINED` | `1` | `0`: networks without pretrained weights (offline tests) |

## Tools

| Tool | What it does |
|---|---|
| `list_automators` | The automators, their data and models |
| `get_data_contract(automator, dim?)` | **The data contract**: layout, formats, column/folder rules, examples, the configuration fields (type, default, choices, range), the models (keys), metrics and output files |
| `get_server_info` | GPU or CPU, versions, workspace, data folders |
| `get_example_data(automator, variant)` | A ready-made dataset (2d/3d) and a configuration that runs in minutes on a CPU |
| `upload_file(filename, content_base64, append?)` | Sends a CSV or zip to the server (in chunks for large files) when it cannot read your files |
| `create_experiment(automator, train, test, name?)` | Copies/extracts the data and checks it as the web upload does: summary, **errors**, warnings, default configuration |
| `start_experiment(experiment_id, config?, dry_run?)` | Validates the configuration (defaults for any field left out) and queues the run; `dry_run` only validates |
| `run_experiment(automator, train, test, config?)` | Both of the above in one call |
| `get_experiment(experiment_id)` | State, phase, progress, status of every model, last log lines |
| `list_experiments` | Every experiment of the workspace |
| `get_results(experiment_id)` | Test and validation metrics, best model, **validation splits**, trained models, every output file |
| `read_result_file(experiment_id, path)` | A CSV/JSON/log as text, a figure as an image |
| `get_log`, `cancel_experiment`, `delete_experiment` | Run log, stop a run, remove an experiment |

Resources: `simplatab://automators`, `simplatab://contracts/{automator}`, `simplatab://experiments/{experiment_id}`.

### A session

```text
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

`simplatab_mcp/`: `server.py` (tools), `jobs.py` (experiments, queue, cancellation), `worker.py` (commands run in
the pipelines' environment), `contracts.py` (data contracts and configuration schemas), `configs.py` (configuration
→ pipeline parameters, as the web forms), `datasets.py` (data preparation and checks, as the web uploads),
`tabular.py` (the tabular run of the web application), `status.py`, `results.py`, `examples.py`.

## Tests

```bash
cd mcp_server
SIMPLATAB_PRETRAINED=0 /path/to/simplatab-venv/bin/python -m unittest tests.test_worker          # Python 3.9 environment
SIMPLATAB_PRETRAINED=0 SIMPLATAB_WORKER_PYTHON=/path/to/simplatab-venv/bin/python \
    /path/to/mcp-venv/bin/python -m unittest tests.test_server                                   # MCP client end to end
```
