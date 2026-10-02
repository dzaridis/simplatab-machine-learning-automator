"""The data of an experiment: copied, extracted or linked into <experiment>/input, then checked
with the same functions as the uploads of the web application. Python 3.9 compatible."""
import os
import shutil
import zipfile
from pathlib import Path

from .paths import resolve_data_path

CSV_AUTOMATORS = ("tabular", "time-series-forecasting", "clustering", "survival-analysis")
OPTIONAL_TEST = ("clustering",)


class DataError(ValueError):
    """A problem with the data, reported to the agent as is."""


def prepare_inputs(automator, train, test, input_dir):
    """Puts Train and Test into input_dir: Train.csv/Test.csv, or train/ and test/ folders (zips are
    extracted with the checks of the web upload; folders are linked, not copied)."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for split, value in (("train", train), ("test", test)):
        if not value and split == "test" and automator in OPTIONAL_TEST:
            stale = input_dir / "Test.csv"
            if stale.exists():
                stale.unlink()
            continue
        if not value:
            raise DataError(f"The {split} data is missing" + (" (test is optional for clustering only)." if split == "test" else "."))
        path = resolve_data_path(value)
        sources[split] = str(path)
        if automator in CSV_AUTOMATORS:
            if not (path.is_file() and path.suffix.lower() == ".csv"):
                raise DataError(f"{value}: the {split} data of {automator} is a CSV file.")
            shutil.copyfile(path, input_dir / ("Train.csv" if split == "train" else "Test.csv"))
            continue
        target = input_dir / split
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        if path.is_file() and zipfile.is_zipfile(path):
            from Helpers.image import dataset as image_dataset
            if path.stat().st_size > image_dataset.MAX_ZIP_BYTES:
                raise DataError(f"{value} is larger than {image_dataset.MAX_ZIP_BYTES // 1024 ** 3} GB.")
            try:
                image_dataset.extract_zip(str(path), str(target))
            except image_dataset.DatasetError as e:
                raise DataError(f"{value}: {e}")
        elif path.is_dir():
            os.symlink(path, target, target_is_directory=True)
        else:
            raise DataError(f"{value}: the {split} data of {automator} is a zip file or a folder.")
    return sources


# ---------------------------------------------------------------------------------------
# Checks (as the upload of the web application)
# ---------------------------------------------------------------------------------------

def _tabular(input_dir):
    import pandas as pd
    from Helpers.data_checks import DataChecker
    train = pd.read_csv(input_dir / "Train.csv")
    test = pd.read_csv(input_dir / "Test.csv")
    errors, warnings = [], []
    ids = {"ID", "patient_id"}
    summary = {
        "train_rows": len(train), "test_rows": len(test), "columns": list(train.columns),
        "id_column": next((c for c in ("ID", "patient_id") if c in train.columns), None),
        "features": [c for c in train.columns if c not in ids | {"Target"}],
        "categorical_features": [c for c in train.columns if c not in ids | {"Target"} and train[c].dtype == object],
        "train_rows_with_missing": int(train.isna().any(axis=1).sum()),
        "test_rows_with_missing": int(test.isna().any(axis=1).sum()),
        "bias_feature_candidates": [c for c in train.columns if c not in ids | {"Target"}
                                    and (train[c].dtype == object or train[c].nunique() <= 10)],
    }
    if "Target" not in train.columns or "Target" not in test.columns:
        errors.append("Train.csv and Test.csv need a Target column.")
    elif not pd.api.types.is_numeric_dtype(train["Target"]):
        errors.append("The Target column must hold numeric classes 0, 1, ..., K-1.")
    else:
        complete = train.dropna()
        counts = complete["Target"].value_counts().sort_index()
        summary.update(classes=[int(c) for c in counts.index], num_classes=int(len(counts)),
                       class_counts={str(int(c)): int(n) for c, n in counts.items()},
                       min_class_count=int(counts.min()) if len(counts) else 0)
        issue = DataChecker.target_label_issue(train, test)
        if issue:
            errors.append(issue)
        if len(counts) < 2:
            errors.append("Train.csv holds a single class (after removing rows with missing values).")
        elif counts.min() < 2:
            errors.append("Every class needs at least 2 complete rows in Train.csv.")
    missing = sorted(set(train.columns) - set(test.columns) - ids)
    if missing:
        errors.append(f"Test.csv lacks the columns {', '.join(missing)} of Train.csv.")
    extra = sorted(set(test.columns) - set(train.columns) - ids)
    if extra:
        warnings.append(f"Test.csv columns not in Train.csv are ignored: {', '.join(extra)}.")
    if summary["train_rows_with_missing"]:
        warnings.append(f"{summary['train_rows_with_missing']} Train.csv rows with missing values will be removed.")
    for column in summary["categorical_features"]:
        if column in test.columns and set(train[column].dropna()) != set(test[column].dropna()):
            warnings.append(f"Categorical column {column} has different values in Train.csv and Test.csv: it will be dropped.")
    if summary["id_column"] is None:
        warnings.append("No ID (or patient_id) column: the validation splits identify the rows by their line in Train.csv.")
    return summary, errors, warnings


def _image(input_dir):
    import logging
    import traceback
    from Helpers.image import dataset as image_dataset
    from Helpers.image3d import dataset as image3d_dataset
    summary = image_dataset.summarize(str(input_dir / "train"), str(input_dir / "test"))
    volume3d = None
    kinds = summary.get("kinds", {})
    if not kinds.get("raster") and (kinds.get("dicom") or kinds.get("nifti")):
        try:
            candidate = image3d_dataset.summarize(str(input_dir / "train"), str(input_dir / "test"))
            if candidate["volumetric"] and not candidate["errors"]:
                volume3d = candidate
        except Exception:
            logging.error(traceback.format_exc())
    summary["volume3d"] = volume3d
    summary["dim"] = 3 if volume3d else 2
    errors = [] if volume3d else list(summary.get("errors", []))
    warnings = list(summary.get("warnings", [])) + (list(volume3d.get("warnings", [])) if volume3d else [])
    return summary, errors, warnings


def _detection(input_dir):
    from Helpers.detection import dataset as detection_dataset
    summary = detection_dataset.summarize(str(input_dir / "train"), str(input_dir / "test"))
    return summary, list(summary.get("errors", [])), list(summary.get("warnings", []))


def _segmentation(input_dir):
    from Helpers.segmentation import data as segmentation_data
    try:
        summary = segmentation_data.summarize(str(input_dir / "train"), str(input_dir / "test"))
    except segmentation_data.SegmentationDataError as e:
        return {}, [str(e)], []
    return summary, list(summary.get("errors", [])), list(summary.get("warnings", []))


def _forecasting(input_dir):
    from Helpers.forecasting import data as forecast_data
    summary = forecast_data.summarize(str(input_dir / "Train.csv"), str(input_dir / "Test.csv"))
    return summary, list(summary.get("errors", [])), list(summary.get("warnings", []))


def _clustering(input_dir):
    from Helpers.clustering import data as clustering_data
    test = input_dir / "Test.csv"
    summary = clustering_data.summarize(str(input_dir / "Train.csv"), str(test) if test.exists() else None)
    return summary, list(summary.get("errors", [])), list(summary.get("warnings", []))


def _survival(input_dir):
    from Helpers.survival import data as survival_data
    summary = survival_data.summarize(str(input_dir / "Train.csv"), str(input_dir / "Test.csv"))
    return summary, list(summary.get("errors", [])), list(summary.get("warnings", []))


CHECKS = {"tabular": _tabular, "image-classification": _image, "object-detection": _detection,
          "image-segmentation": _segmentation, "time-series-forecasting": _forecasting,
          "clustering": _clustering, "survival-analysis": _survival}


def check(automator, input_dir):
    """(summary, errors, warnings) of the prepared data of an experiment."""
    try:
        return CHECKS[automator](Path(input_dir))
    except Exception as e:  # unreadable files: a data problem for the agent to fix
        return {}, [f"The data could not be read: {e}"], []
