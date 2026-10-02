"""The validation splits of a run, written so that they can be reproduced exactly.

Every automator writes Materials/Splits/:
- splits.csv: one row per sample and fold, with the columns
      fold         1..K (1 for a hold-out split)
      set          "train" (fitted on), "validation" (scored on) and, where networks stop early
                   on a part of the training fold, "early_stopping"
      id           what identifies the sample in your data: the ID column (tabular), the file or
                   case path inside Train.zip (images), the series ID (forecasting)
  and, depending on the automator, patient (the group kept in one fold), class, row (line of
  Train.csv, the first data line being 1) or the time ranges of the forecasting windows;
- splits.json: the same, as {"kind", "folds": [{"fold": 1, "train": [ids], "validation": [ids]}]}.
"""
import json
import os

import pandas as pd

FOLDER = "Splits"


def write_splits(rows, materials="Materials", kind="kfold", description="", columns=None):
    """rows: dicts with at least fold, set and id. kind: "kfold", "holdout" or "rolling_origin"."""
    folder = os.path.join(materials, FOLDER)
    os.makedirs(folder, exist_ok=True)
    frame = pd.DataFrame(rows)
    order = ["fold", "set", "id"] + [c for c in (columns or frame.columns) if c not in ("fold", "set", "id")]
    frame = frame[[c for c in order if c in frame.columns]]
    frame.to_csv(os.path.join(folder, "splits.csv"), index=False)
    folds = []
    for fold, part in frame.groupby("fold", sort=True):
        entry = {"fold": int(fold)}
        for name, members in part.groupby("set", sort=False):
            entry[name] = [_plain(v) for v in members["id"]]
        folds.append(entry)
    with open(os.path.join(folder, "splits.json"), "w") as f:
        json.dump({"kind": kind, "description": description, "folds": folds}, f, indent=1)
    return frame


def index_rows(fold_indices, ids, extra=None, sets=("train", "validation")):
    """Rows of write_splits from [(train indices, validation indices), ...] (or triples with an
    early-stopping part when ``sets`` has three names); ``extra``: {column: values per sample}."""
    rows = []
    for k, parts in enumerate(fold_indices, start=1):
        for name, indices in zip(sets, parts):
            for i in indices:
                row = {"fold": k, "set": name, "id": _plain(ids[i])}
                for column, values in (extra or {}).items():
                    row[column] = _plain(values[i])
                rows.append(row)
    return rows


def _plain(value):
    return value.item() if hasattr(value, "item") else value
