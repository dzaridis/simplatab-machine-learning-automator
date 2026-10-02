"""The worker: runs one Simplatab task in the Python environment of Simplatab, as a subprocess of
the MCP server. Python 3.9 compatible, never imports the MCP SDK.

    python -m simplatab_mcp.worker <command> --output result.json [--automator A] [--experiment DIR]
                                   [--config JSON] [--variant 2d|3d]

Commands: info, contract, example, prepare (data of an experiment), configure (configuration of an
experiment), run (the experiment; long), results. The result (or {"error": ...}) is written as
JSON to --output; run writes status.json and run.log in the experiment folder as it goes.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from .paths import simplatab_root, workspace

PIPELINES = {
    "image-classification": ("Helpers.image.pipeline", "run_image_pipeline"),
    "image-classification-3d": ("Helpers.image3d.pipeline", "run_image3d_pipeline"),
    "object-detection": ("Helpers.detection.pipeline", "run_detection_pipeline"),
    "image-segmentation": ("Helpers.segmentation.pipeline", "run_segmentation_pipeline"),
    "time-series-forecasting": ("Helpers.forecasting.pipeline", "run_forecasting_pipeline"),
}


def _setup():
    os.environ.setdefault("MPLBACKEND", "Agg")
    root = str(simplatab_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load(experiment):
    with open(Path(experiment) / "experiment.json") as f:
        return json.load(f)


def info(args):
    import platform
    import torch
    gpu = torch.cuda.is_available()
    return {"python": platform.python_version(), "torch": torch.__version__, "gpu": gpu,
            "device": torch.cuda.get_device_name(0) if gpu else f"CPU ({os.cpu_count()} cores)",
            "simplatab_root": str(simplatab_root()), "workspace": str(workspace()),
            "pretrained_weights": os.environ.get("SIMPLATAB_PRETRAINED", "1") != "0"}


def contract(args):
    from . import contracts
    return contracts.contract(args.automator, int(args.dim) if args.dim else None)


def inspect_data(args):
    from .detect import inspect
    return inspect(args.train)


def example(args):
    from .examples import example as make
    return make(args.automator, args.variant, workspace())


def prepare(args):
    """Puts the data of an experiment into its input folder and checks it."""
    from . import configs, datasets
    experiment = Path(args.experiment)
    meta = _load(experiment)
    sources = datasets.prepare_inputs(meta["automator"], meta["train"], meta["test"], experiment / "input")
    summary, errors, warnings = datasets.check(meta["automator"], experiment / "input")
    out = {"sources": sources, "summary": summary, "errors": errors, "warnings": warnings, "default_config": None}
    if not errors:
        try:
            out["default_config"] = configs.defaults(meta["automator"], summary)
        except configs.ConfigError as e:
            out["errors"].append(str(e))
    with open(experiment / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return out


def configure(args):
    """Validates a configuration against the data of an experiment: the pipeline parameters."""
    from . import configs
    experiment = Path(args.experiment)
    meta = _load(experiment)
    with open(experiment / "summary.json") as f:
        summary = json.load(f)
    params, names = configs.build(meta["automator"], json.loads(args.config or "{}"), summary)
    return {"params": params, "model_names": names}


def run(args):
    """Runs the experiment (cwd: the experiment folder, so that the outputs go to its Materials)."""
    import importlib
    from . import results, status
    experiment = Path(args.experiment).resolve()
    meta = _load(experiment)
    automator, params = meta["automator"], meta["params"]
    os.chdir(experiment)
    os.makedirs("Materials", exist_ok=True)
    tracker = status.Status(str(experiment / "status.json"), automator, meta.get("model_names", []))
    original = sys.stdout
    sys.stdout = status.Tee(original, tracker.line)
    try:
        if automator == "tabular":
            from .tabular import run_tabular_pipeline
            message = run_tabular_pipeline("input", params)
        else:
            key = "image-classification-3d" if automator == "image-classification" and params.get("dim") == 3 else automator
            module, function = PIPELINES[key]
            message = getattr(importlib.import_module(module), function)("input", params)
    except Exception as e:  # the pipelines catch their errors; this is a safety net
        traceback.print_exc()
        message = f"Error: {e}"
    finally:
        sys.stdout.flush()
        sys.stdout = original
    failed = isinstance(message, str) and message.startswith("Error")
    try:
        status.write_json(str(experiment / "results.json"), results.collect(str(experiment), automator))
    except Exception as e:
        print(f"The results could not be summarised: {e}")
    tracker.finish("failed" if failed else "completed", message or "Pipeline completed successfully")
    return {"state": tracker.state, "message": tracker.message}


def results_command(args):
    from . import results
    experiment = Path(args.experiment)
    return results.collect(str(experiment), _load(experiment)["automator"])


COMMANDS = {"info": info, "contract": contract, "example": example, "inspect": inspect_data, "prepare": prepare, "configure": configure,
            "run": run, "results": results_command}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="simplatab_mcp.worker")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--automator")
    parser.add_argument("--experiment")
    parser.add_argument("--config")
    parser.add_argument("--variant")
    parser.add_argument("--dim")
    parser.add_argument("--train")
    args = parser.parse_args(argv)
    _setup()
    output = os.path.abspath(args.output)
    try:
        result = COMMANDS[args.command](args)
    except Exception as e:
        known = type(e).__name__ in ("ConfigError", "DataError", "FileNotFoundError", "PermissionError", "KeyError")
        result = {"error": str(e).strip("'\"") if known else f"{type(e).__name__}: {e}"}
        if not known:
            result["traceback"] = traceback.format_exc()[-4000:]
    with open(output + ".tmp", "w") as f:
        json.dump(result, f, default=str)
    os.replace(output + ".tmp", output)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
