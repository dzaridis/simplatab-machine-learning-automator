"""Data of the clustering automator: Train.csv (and an optional Test.csv) with one row per sample.

- ``ID`` (or ``patient_id``): optional identifier, never a feature (reported in the outputs);
- ``Target``: optional class labels. They are never used to find the clusters, only to evaluate
  them (external metrics: ARI, AMI, NMI, purity...): supervised evaluation when present,
  unsupervised clustering otherwise;
- every other column is a feature: numeric, or categorical (text, booleans; one-hot encoded).

Missing values are imputed (median, or a "missing" category). Columns that cannot help (constant,
mostly missing, identifier-like text) are left out with a warning. Features are scaled and can be
reduced with PCA; the preprocessing is fitted on Train.csv only.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

ID_COLUMNS = ("ID", "patient_id")
LABEL = "Target"
MAX_ROWS = 100_000
MIN_ROWS = 10
MAX_CATEGORIES = 50
MAX_CLASSES = 50
MISSING = "missing"


class DataError(ValueError):
    """A problem with the data, reported to the user as is."""


def read(path):
    try:
        frame = pd.read_csv(path)
    except Exception as e:
        raise DataError(f"{path.rsplit('/', 1)[-1]} could not be read as a CSV file: {e}")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def id_column(frame):
    return next((c for c in ID_COLUMNS if c in frame.columns), None)


def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def label_kind(series):
    """'classes' if the column holds class labels, 'continuous' if it is a measurement (many distinct
    non-integer values: not usable as classes), None if it is empty."""
    values = series.dropna()
    if values.empty:
        return None
    if is_numeric(values) and values.nunique() > MAX_CLASSES and (values % 1 != 0).any():
        return "continuous"
    if values.nunique() > MAX_CLASSES:
        return "continuous"
    return "classes"


def labels_of(series):
    """Class labels as text ('1' rather than '1.0'); missing labels are None."""
    def text(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v)
    return np.array([text(v) for v in series], dtype=object)


def feature_columns(train, ignore=()):
    """(numeric, categorical, dropped) feature columns of Train.csv; dropped: [(column, reason)]."""
    numeric, categorical, dropped = [], [], []
    n = len(train)
    for column in train.columns:
        if column in ID_COLUMNS or column == LABEL:
            continue
        if column in ignore:
            dropped.append((column, "ignored (configuration)"))
            continue
        series = train[column]
        missing = series.isna().mean()
        if missing > 0.5:
            dropped.append((column, f"{100 * missing:.0f}% missing values"))
        elif series.nunique(dropna=True) <= 1:
            dropped.append((column, "constant"))
        elif is_numeric(series):
            numeric.append(column)
        else:
            levels = series.astype(str).nunique()
            if levels > MAX_CATEGORIES and levels > 0.5 * n:
                dropped.append((column, f"identifier-like text ({levels} distinct values)"))
            elif levels > MAX_CATEGORIES:
                dropped.append((column, f"categorical with {levels} categories (at most {MAX_CATEGORIES})"))
            else:
                categorical.append(column)
    return numeric, categorical, dropped


def clean(frame, numeric, categorical):
    """The feature columns as the models expect them: numbers, and categories as text ("missing" for
    a missing value). Also used by the saved models on new data."""
    out = pd.DataFrame(index=frame.index)
    for column in numeric:
        out[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    for column in categorical:
        values = frame[column]
        out[column] = values.map(lambda v: MISSING if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out


# ---------------------------------------------------------------------------------------
# Summary (upload checks)
# ---------------------------------------------------------------------------------------

def summarize(train_path, test_path=None):
    """What the upload page and the configuration need, with the errors that block a run and the
    warnings to review."""
    errors, warnings = [], []
    try:
        train = read(train_path)
    except DataError as e:
        return {"errors": [str(e)], "warnings": []}
    test = None
    if test_path:
        try:
            test = read(test_path)
        except DataError as e:
            errors.append(str(e))
    n = len(train)
    if n < MIN_ROWS:
        errors.append(f"Train.csv has {n} rows: clustering needs at least {MIN_ROWS}.")
    if n > MAX_ROWS:
        errors.append(f"Train.csv has {n:,} rows: at most {MAX_ROWS:,}.")
    numeric, categorical, dropped = feature_columns(train)
    if not numeric and not categorical:
        errors.append("Train.csv has no usable feature column (besides ID and Target).")
    ident = id_column(train)
    if ident and train[ident].duplicated().any():
        warnings.append(f"{int(train[ident].duplicated().sum())} repeated {ident} value(s) in Train.csv: the rows are still clustered one by one.")
    if ident is None:
        warnings.append("No ID (or patient_id) column: the samples are identified by their row in Train.csv.")

    summary = {
        "train_rows": n, "test_rows": len(test) if test is not None else 0, "has_test": test is not None,
        "id_column": ident, "numeric": numeric, "categorical": categorical,
        "dropped": [{"column": c, "reason": r} for c, r in dropped],
        "features": numeric + categorical, "has_labels": False, "test_has_labels": False,
        "classes": [], "class_counts": {}, "missing_values": int(train[numeric + categorical].isna().sum().sum()) if n else 0,
    }
    if dropped:
        warnings.append("Columns left out: " + "; ".join(f"{c} ({r})" for c, r in dropped) + ".")
    if summary["missing_values"]:
        warnings.append(f"{summary['missing_values']} missing feature value(s) in Train.csv: imputed (median, or a 'missing' category).")

    if LABEL in train.columns:
        kind = label_kind(train[LABEL])
        if kind == "classes":
            labels = labels_of(train[LABEL])
            known = [v for v in labels if v is not None]
            counts = pd.Series(known).value_counts()
            summary.update(has_labels=True, classes=sorted(counts.index, key=_natural),
                           class_counts={str(k): int(v) for k, v in counts.items()})
            if len(counts) < 2:
                warnings.append("Target holds a single class: the external metrics need at least two.")
            if len(known) < n:
                warnings.append(f"{n - len(known)} Train.csv row(s) without a Target are clustered but not used by the external metrics.")
        elif kind == "continuous":
            warnings.append("Target holds continuous values, not classes: it is left out (neither a feature nor labels).")
        else:
            warnings.append("Target is empty: the clustering is evaluated without labels.")

    if test is not None:
        missing = [c for c in numeric + categorical if c not in test.columns]
        if missing:
            errors.append(f"Test.csv lacks the feature column(s) {', '.join(missing)} of Train.csv.")
        if len(test) < 2:
            errors.append("Test.csv needs at least 2 rows.")
        if len(test) > MAX_ROWS:
            errors.append(f"Test.csv has {len(test):,} rows: at most {MAX_ROWS:,}.")
        if summary["has_labels"]:
            if LABEL in test.columns and label_kind(test[LABEL]) == "classes":
                summary["test_has_labels"] = True
                unseen = sorted({v for v in labels_of(test[LABEL]) if v is not None} - set(summary["classes"]), key=_natural)
                if unseen:
                    warnings.append(f"Test.csv has classes not in Train.csv ({', '.join(unseen[:10])}): they count as other classes in the external metrics.")
            else:
                warnings.append("Test.csv has no Target: its clusters are evaluated without labels.")
        for column in categorical:
            if column in test.columns:
                new = set(test[column].dropna().astype(str)) - set(train[column].dropna().astype(str))
                if new:
                    warnings.append(f"Test.csv values of {column} not in Train.csv ({', '.join(sorted(new)[:5])}) are encoded as unknown.")
    k_cap = max(2, min(30, n // 3))
    summary["max_k"] = k_cap
    summary["suggested_k"] = min(len(summary["classes"]), k_cap) if summary["has_labels"] and len(summary["classes"]) >= 2 else None
    summary["max_folds"] = max(2, min(10, n // 5))
    summary["errors"], summary["warnings"] = errors, warnings
    return summary


def _natural(value):
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


# ---------------------------------------------------------------------------------------
# Preparation (pipeline)
# ---------------------------------------------------------------------------------------

SCALERS = ("standard", "robust", "minmax", "none")


def make_featurizer(numeric, categorical, scaling):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

    scaler = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler(),
              "none": FunctionTransformer()}[scaling]
    parts = []
    if numeric:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", scaler)]), numeric))
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def prepare(train_path, test_path, params):
    """The data of a run: original frames, labels, ids, and the features in the space the models
    use (``X_train``, ``X_test``: scaled, one-hot encoded, optionally reduced by PCA) and before the
    PCA (``F_train``, ``F_test``, named ``feature_names``: for the explanations)."""
    summary = summarize(train_path, test_path)
    if summary["errors"]:
        raise DataError(" ".join(summary["errors"]))
    train = read(train_path)
    test = read(test_path) if test_path else None
    numeric, categorical, dropped = feature_columns(train, ignore=set(params.get("ignore_columns") or []))
    if not numeric and not categorical:
        raise DataError("No feature column is left: do not ignore every column.")
    scaling = params.get("scaling", "standard")
    featurizer = make_featurizer(numeric, categorical, scaling)
    F_train = np.asarray(featurizer.fit_transform(clean(train, numeric, categorical)), dtype=float)
    names = [str(n) for n in featurizer.get_feature_names_out()]
    reducer = None
    if params.get("reduction", "none") == "pca" and F_train.shape[1] > 2:
        from sklearn.decomposition import PCA
        variance = float(params.get("pca_variance", 0.95))
        reducer = PCA(n_components=variance, svd_solver="full", random_state=int(params.get("seed", 42))).fit(F_train)
    X_train = reducer.transform(F_train) if reducer is not None else F_train
    ident = id_column(train)
    data = SimpleNamespace(
        train=train, test=test, id_column=ident, numeric=numeric, categorical=categorical, dropped=dropped,
        featurizer=featurizer, reducer=reducer, feature_names=names, scaling=scaling,
        F_train=F_train, X_train=np.ascontiguousarray(X_train, dtype=np.float32),
        ids_train=[str(v) for v in train[ident]] if ident else [str(i + 1) for i in range(len(train))],
        has_labels=summary["has_labels"], classes=summary["classes"], summary=summary,
        y_train=labels_of(train[LABEL]) if summary["has_labels"] else None,
        F_test=None, X_test=None, ids_test=None, y_test=None)
    if test is not None:
        F_test = np.asarray(featurizer.transform(clean(test, numeric, categorical)), dtype=float)
        data.F_test = F_test
        data.X_test = np.ascontiguousarray(reducer.transform(F_test) if reducer is not None else F_test, dtype=np.float32)
        test_ident = id_column(test)
        data.ids_test = [str(v) for v in test[test_ident]] if test_ident else [str(i + 1) for i in range(len(test))]
        data.y_test = labels_of(test[LABEL]) if summary["test_has_labels"] else None
    return data
