"""The results of a finished run, read from its Materials folder into JSON for the agent:
metrics tables, best model, validation splits, trained models and every file. Python 3.9
compatible (pandas reads the Excel tables)."""
import glob
import json
import math
import os

LOWER_IS_BETTER = {"HD95", "ASSD", "MAE", "RMSE", "sMAPE", "MASE", "Davies-Bouldin", "IBS"}


def _value(v):
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 4)
    if hasattr(v, "item"):
        return _value(v.item())
    return v


def _table(path):
    import pandas as pd
    frame = pd.read_excel(path, index_col=0)
    return [dict(model=str(model), **{str(c): _value(row[c]) for c in frame.columns}) for model, row in frame.iterrows()]


def _best(automator, info, test):
    if info.get("best_model"):
        return info["best_model"]
    if not test:
        return None
    metric = "AUC" if "AUC" in test[0] else next((k for k in test[0] if k != "model"), None)
    rows = [r for r in test if isinstance(r.get(metric), (int, float))]
    if not rows:
        return None
    pick = min if metric in LOWER_IS_BETTER else max
    return pick(rows, key=lambda r: r[metric])["model"]


def collect(experiment_dir, automator):
    root = os.path.join(experiment_dir, "Materials")
    out = {"automator": automator, "materials": root, "test_metrics": [], "validation_metrics": [],
           "validation_file": None, "best_model": None, "run_info": {}, "splits": None, "models": [], "files": [],
           "skipped_models": []}
    if not os.path.isdir(root):
        return out
    info_path = os.path.join(root, "run_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            out["run_info"] = json.load(f)
    info = out["run_info"]
    if os.path.exists(os.path.join(root, "test_results.xlsx")):
        out["test_metrics"] = _table(os.path.join(root, "test_results.xlsx"))
    if os.path.exists(os.path.join(root, "train_results.xlsx")):  # clustering: the clusters of Train.csv
        out["train_metrics"] = _table(os.path.join(root, "train_results.xlsx"))
    candidates = ([os.path.join(root, info["validation_file"])] if info.get("validation_file") else []) + \
        sorted(glob.glob(os.path.join(root, "*_fold_results.xlsx"))) + [os.path.join(root, "holdout_results.xlsx")]
    for path in candidates:
        if os.path.isfile(path):
            out["validation_file"] = os.path.basename(path)
            out["validation_metrics"] = _table(path)
            break
    out["best_model"] = _best(automator, info, out["test_metrics"])
    out["metric_direction"] = {k: ("lower" if k in LOWER_IS_BETTER or k.startswith("Brier@") else "higher")
                               for k in (out["test_metrics"] or out.get("train_metrics") or [{}])[0] if k != "model"}
    splits = os.path.join(root, "Splits", "splits.json")
    if os.path.exists(splits):
        with open(splits) as f:
            data = json.load(f)
        out["splits"] = {"kind": data.get("kind"), "description": data.get("description"),
                         "files": ["Splits/splits.csv", "Splits/splits.json"],
                         "folds": [{"fold": fold["fold"], **{k: len(v) for k, v in fold.items() if k != "fold"}}
                                   for fold in data.get("folds", [])]}
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            out["files"].append({"path": rel, "size": os.path.getsize(path)})
            if rel.startswith("Models/"):
                out["models"].append(rel)
    out["files"].sort(key=lambda f: f["path"])
    log = os.path.join(root, "error_log.log")
    if os.path.exists(log):
        with open(log, errors="replace") as f:
            for line in f:
                if "failed and was skipped:" in line:
                    model, reason = line.split(":ERROR:", 1)[-1].split(" failed and was skipped:", 1)
                    out["skipped_models"].append({"model": model.strip(), "reason": reason.strip()})
    out["skipped_models"] += [s for s in info.get("skipped", []) if isinstance(s, dict)]
    return out
