"""The Simplatab MCP server: tools for AI agents to run Simplatab experiments end to end.

An agent provides the data and the configuration; the server checks the data against the
automator's contract, queues the run, reports its progress and returns the results.
"""
import base64
import json
import os
import re
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

from . import __version__, paths
from .contracts import AUTOMATORS
from .jobs import Experiments, WorkerError

INSTRUCTIONS = """Simplatab runs validated machine learning experiments on research data: tabular classification,
2D/3D image classification, 2D/3D object detection, 2D/3D image segmentation (incl. the official nnU-Net),
time series forecasting, survival analysis (time-to-event with censoring) and clustering of tabular data
(unsupervised, or evaluated against labels). Each experiment trains several models with K-fold (or hold-out / rolling-origin)
validation, evaluates them on an external test set, explains them, and exports the trained models and the
validation splits.

Workflow:
1. Which automator: inspect_data(train) suggests it from the data (or pass automator="auto" to
   create_experiment / run_experiment). get_data_contract(automator) gives the data layout, formats, rules,
   the configuration fields with their defaults, and the models. get_example_data gives a ready-made dataset.
2. Make the data readable by the server: a path inside the server (e.g. /data/... when the Docker image
   mounts a folder on /data; host paths of the mounted folders are translated) or upload_file (base64, in
   chunks for large files).
3. create_experiment(automator, train, test): checks the data like the Simplatab web upload and returns a
   summary, errors, warnings and the default configuration (test is optional for clustering). Fix the data if
   there are errors.
4. start_experiment(experiment_id, config): fields left out keep their defaults; dry_run=true only validates.
   (run_experiment does 3 and 4 in one call.)
5. get_experiment(experiment_id) every minute or so until state is completed/failed/cancelled (runs take
   minutes to hours; they are queued and run one at a time), then get_results and read_result_file.
"""


def _error(e):
    message = str(e)
    if isinstance(e, KeyError) and message.startswith(("'", '"')):
        message = message[1:-1]
    return ToolError(message)


def _check_automator(automator, auto=False):
    if auto and automator == "auto":
        return
    if automator not in AUTOMATORS:
        raise ToolError(f"Unknown automator {automator!r}: one of {', '.join(AUTOMATORS)}.")


def create_server(experiments=None):
    server = MCPServer(name="simplatab", title="Simplatab", version=__version__, instructions=INSTRUCTIONS,
                       website_url="https://github.com/dzaridis/simplatab-machine-learning-automator")
    store = {"experiments": experiments}

    def ex():
        if store["experiments"] is None:
            store["experiments"] = Experiments()
        return store["experiments"]

    # ---- discovery ------------------------------------------------------------------------
    @server.tool()
    def list_automators() -> dict[str, Any]:
        """The Simplatab automators: id, task, the expected training and test data, models, validation and
        explanations. Use the id with get_data_contract and create_experiment."""
        return {"automators": [dict(id=k, **v) for k, v in AUTOMATORS.items()],
                "next_step": "get_data_contract(automator) for the exact data layout and configuration."}

    @server.tool()
    def get_data_contract(automator: str, dim: Optional[int] = None) -> dict[str, Any]:
        """The data contract of an automator: the layout and formats Train and Test must have, the column or
        folder rules, examples, the configuration fields (type, default, choices or range), the models that
        can be selected (by key), the metrics and the output files. dim (2 or 3) restricts image automators
        to 2D or 3D."""
        _check_automator(automator)
        try:
            return ex().cached(("contract", automator, dim), "contract", automator=automator,
                               dim=str(dim) if dim else None)
        except WorkerError as e:
            raise _error(e)

    @server.tool()
    def get_server_info() -> dict[str, Any]:
        """The compute device (GPU or CPU), library versions, the workspace, where data paths are looked up,
        and how many runs execute at the same time."""
        try:
            info = dict(ex().cached(("info",), "info"))
        except WorkerError as e:
            raise _error(e)
        info.update(server_version=__version__, data_roots=[str(p) for p in paths.data_roots()],
                    uploads=str(paths.uploads_dir()), max_parallel_runs=ex().max_parallel)
        return info

    @server.tool()
    def get_example_data(automator: str, variant: str = "2d") -> dict[str, Any]:
        """Writes an example dataset of an automator in the workspace and returns its train and test paths and
        a quick configuration (a few minutes on a CPU; detection and 3D segmentation up to ~15), ready for
        run_experiment. variant: 2d or 3d (image classification, object detection and segmentation); labeled or
        unlabeled (clustering: with or without the Target column)."""
        _check_automator(automator)
        if variant not in ("2d", "3d", "labeled", "unlabeled"):
            raise ToolError("variant is 2d or 3d (image automators), labeled or unlabeled (clustering).")
        try:
            return ex().worker("example", automator=automator, variant=variant)
        except WorkerError as e:
            raise _error(e)

    # ---- data -----------------------------------------------------------------------------
    @server.tool()
    def upload_file(filename: str, content_base64: str, append: bool = False) -> dict[str, Any]:
        """Stores a file (a CSV or a zip of images) sent as base64 in the server's uploads folder and returns its
        path for create_experiment. Large files: send chunks of a few MB with append=true for every chunk after
        the first. Files already on the server (e.g. a mounted /data folder) need no upload."""
        name = os.path.basename(filename or "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,200}", name):
            raise ToolError("filename: letters, digits, '.', '_', '-' and spaces only (no folders).")
        try:
            data = base64.b64decode(content_base64, validate=True)
        except Exception:
            raise ToolError("content_base64 is not valid base64.")
        path = paths.uploads_dir() / name
        with open(path, "ab" if append else "wb") as f:
            f.write(data)
        return {"path": str(path), "size": path.stat().st_size, "appended": append}

    @server.tool()
    def inspect_data(train: str) -> dict[str, Any]:
        """Suggests the automator for a dataset from a quick look at the training data (CSV columns, or the
        folders and files of a zip or folder, e.g. class folders, images/ + masks/, COCO or YOLO annotations):
        suggested_automator, ranked candidates with the reason, and what was seen. Then read the contract of
        the suggested automator, or create the experiment with automator="auto"."""
        try:
            return ex().inspect(train)
        except WorkerError as e:
            raise _error(e)

    # ---- experiments ----------------------------------------------------------------------
    @server.tool()
    def create_experiment(automator: str, train: str, test: Optional[str] = None, name: Optional[str] = None) -> dict[str, Any]:
        """Creates an experiment from training and test data and checks them against the automator's contract.
        train and test: paths readable by the server (absolute, or relative to /data, the uploads or the
        workspace): CSV files (tabular, time-series-forecasting, survival-analysis, clustering), or zip files or folders (image
        automators). test is required, except for clustering (optional: its samples are assigned to the clusters).
        automator: an automator id, or "auto" to choose it from the data (as inspect_data).
        Returns experiment_id, state (ready or invalid), the automator, errors, warnings, the data summary
        (classes, columns, series, counts...), default_config and the experiment folder."""
        _check_automator(automator, auto=True)
        try:
            return ex().create(automator, train, test, name)
        except (WorkerError, OSError) as e:
            raise _error(e)

    @server.tool()
    def start_experiment(experiment_id: str, config: Optional[dict] = None, dry_run: bool = False) -> dict[str, Any]:
        """Validates a configuration (fields of get_data_contract: config; any field left out keeps its default)
        and queues the run. dry_run=true only validates and returns the resolved pipeline parameters. A finished
        experiment can be started again with another configuration (its previous results are replaced)."""
        try:
            return ex().start(experiment_id, config, dry_run)
        except (WorkerError, ValueError, KeyError) as e:
            raise _error(e)

    @server.tool()
    def run_experiment(automator: str, train: str, test: Optional[str] = None, config: Optional[dict] = None,
                       name: Optional[str] = None) -> dict[str, Any]:
        """create_experiment and start_experiment in one call (automator may be "auto"). If the data has errors,
        nothing runs and the errors are returned; if the configuration is invalid, the experiment stays ready
        and the error says which field to fix (then call start_experiment)."""
        created = create_experiment(automator, train, test, name)
        if created["state"] != "ready":
            return created
        try:
            started = start_experiment(created["experiment_id"], config)
        except ToolError as e:  # the data is fine: the agent fixes the configuration and calls start_experiment
            return {**created, "config_error": str(e)}
        return {**started, "automator": created["automator"], "warnings": created["warnings"],
                "summary": created["summary"], "folder_host": created.get("folder_host")}

    @server.tool()
    def get_experiment(experiment_id: str, log_lines: int = 20) -> dict[str, Any]:
        """The state of an experiment (preparing, ready, invalid, queued, running, completed, failed,
        cancelled), its phase and progress (%), the status of every model (validation, test: pending,
        running, done, skipped), the last log lines and whether results are available."""
        try:
            return ex().status(experiment_id, log_lines)
        except KeyError as e:
            raise _error(e)

    @server.tool()
    def list_experiments() -> dict[str, Any]:
        """Every experiment of the workspace with its automator, state and progress."""
        return {"experiments": ex().list()}

    @server.tool()
    def get_results(experiment_id: str) -> dict[str, Any]:
        """The results of a finished experiment: test and validation metrics per model (and whether lower or
        higher is better), the best model, the validation splits (Splits/splits.csv: the samples of every fold),
        the settings of the run, skipped models, trained models and every output file (read them with
        read_result_file)."""
        try:
            return ex().results(experiment_id)
        except (KeyError, ValueError, WorkerError) as e:
            raise _error(e)

    @server.tool()
    def get_log(experiment_id: str, tail: int = 200) -> dict[str, Any]:
        """The last lines of the run log of an experiment."""
        try:
            return ex().log(experiment_id, tail)
        except KeyError as e:
            raise _error(e)

    @server.tool(structured_output=False)
    def read_result_file(experiment_id: str, path: str, max_chars: int = 100000):
        """Reads an output file of an experiment: path relative to its Materials folder (as listed by get_results:
        files), or run.log. Text files (CSV, JSON, logs, Python) are returned as text (up to max_chars), PNG/JPEG
        figures as images, Excel tables as CSV text; other binary files (models) are only described."""
        try:
            target = ex().file_path(experiment_id, path)
        except (KeyError, ValueError, FileNotFoundError) as e:
            raise _error(e)
        suffix = target.suffix.lower()
        size = target.stat().st_size
        if suffix in (".png", ".jpg", ".jpeg"):
            if size > 8 * 1024 * 1024:
                raise ToolError(f"{path} is {size} bytes: too large to return.")
            return Image(path=target)
        if suffix in (".csv", ".json", ".txt", ".log", ".py", ".yaml", ".yml", ".md"):
            with open(target, errors="replace") as f:
                text = f.read(max(1, int(max_chars)) + 1)
            return text if len(text) <= max_chars else text[:max_chars] + f"\n... (truncated, {size} bytes)"
        if suffix == ".xlsx":
            return f"{path}: an Excel table; its content is returned by get_results (test_metrics, validation_metrics)."
        host = paths.to_host(target)
        return (f"{path}: binary file of {size} bytes (a trained model or archive), at {target} on the server"
                + (f" and {host} on the host." if host else ". Mount the workspace on the host to use it."))

    @server.tool()
    def cancel_experiment(experiment_id: str) -> dict[str, Any]:
        """Cancels a queued or running experiment (the run is stopped)."""
        try:
            return ex().cancel(experiment_id)
        except KeyError as e:
            raise _error(e)

    @server.tool()
    def delete_experiment(experiment_id: str) -> dict[str, Any]:
        """Deletes an experiment that is not running, with its data copy and outputs."""
        try:
            return ex().delete(experiment_id)
        except (KeyError, ValueError) as e:
            raise _error(e)

    # ---- prompts --------------------------------------------------------------------------
    @server.prompt()
    def run_simplatab_experiment(train: str, test: str = "", goal: str = "") -> str:
        """Instructions for an agent to run a complete Simplatab experiment on a dataset."""
        return (f"Run a Simplatab experiment on the training data {train}"
                + (f" and the test data {test}." if test else " (no test data: only clustering accepts that).")
                + (f" Goal: {goal}." if goal else "") +
                "\n1. Call inspect_data on the training data and read the data contract of the suggested automator "
                "(get_data_contract). If the data does not follow the contract, explain what to change and stop."
                "\n2. Call get_server_info: on a CPU prefer fast settings (feature extraction, hold-out validation, "
                "light networks, fewer epochs); with a GPU the defaults are fine."
                "\n3. Call create_experiment, fix any error, then start_experiment with a configuration suited to the "
                "goal (dry_run first if unsure)."
                "\n4. Poll get_experiment every minute until it is completed, failed or cancelled."
                "\n5. Call get_results and report the test metrics of every model, the best model, the validation "
                "splits and where the trained models are; read the main figures with read_result_file.")

    # ---- resources ------------------------------------------------------------------------
    @server.resource("simplatab://automators", mime_type="application/json",
                     description="The automators of Simplatab.")
    def automators_resource() -> str:
        return json.dumps(list_automators(), indent=1)

    @server.resource("simplatab://contracts/{automator}", mime_type="application/json",
                     description="The data contract and configuration schema of an automator.")
    def contract_resource(automator: str) -> str:
        return json.dumps(get_data_contract(automator), indent=1)

    @server.resource("simplatab://experiments/{experiment_id}", mime_type="application/json",
                     description="The state and progress of an experiment.")
    def experiment_resource(experiment_id: str) -> str:
        return json.dumps(get_experiment(experiment_id), indent=1)

    server.experiments = ex
    return server
