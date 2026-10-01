"""Time series of the forecasting automator: reading and checking Train.csv and Test.csv,
and building the data frames of neuralforecast.

Long format: one row per series and time point, with the columns
- ID (or patient_id): the series, e.g. a patient;
- Time: integer time steps or dates, at a regular frequency (missing time points are filled);
- Target: the value to forecast;
- any other column: a covariate. Columns constant within every series are static features
  (e.g. sex); the others vary over time and are either known in advance ("future", e.g. a
  scheduled dose or the month) or only observed up to the present ("past", e.g. another lab
  value). Categorical columns are one-hot encoded.

Test.csv: the last H time points (the horizon) of every series are forecast from the points
before them. A series whose ID is also in Train.csv continues it: its Train.csv points earlier
than its first Test.csv point are prepended to its history (temporal hold-out). A new ID is an
unseen series (e.g. a new patient), forecast from its own first points.
"""
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ID, TIME, TARGET = "ID", "Time", "Target"
ID_ALIASES = ("ID", "patient_id")
MAX_CATEGORIES = 20
# Seasonal period suggested for the seasonal naive baseline and MASE, by frequency
SEASONS = {"H": 24, "D": 7, "B": 5, "W": 52, "M": 12, "Q": 4, "A": 1, "Y": 1}
# Horizon suggested by frequency (one season ahead, within reason)
HORIZONS = {"H": 24, "D": 14, "B": 10, "W": 8, "M": 6, "Q": 4, "A": 2, "Y": 2}


class DataError(ValueError):
    """A problem with the uploaded files, explained to the user."""


# ---------------------------------------------------------------------------------------
# Reading and checks
# ---------------------------------------------------------------------------------------

def read_table(path, name):
    try:
        frame = pd.read_csv(path)
    except Exception as e:  # malformed CSV
        raise DataError(f"{name} could not be read as a CSV file ({e}).")
    frame.columns = [str(c).strip() for c in frame.columns]
    id_column = next((c for c in ID_ALIASES if c in frame.columns), None)
    missing = [c for c, ok in ((ID, id_column), (TIME, TIME in frame.columns), (TARGET, TARGET in frame.columns)) if not ok]
    if missing:
        raise DataError(f"{name} has no {' / '.join(missing)} column: the columns must include ID, Time and Target.")
    if not len(frame):
        raise DataError(f"{name} has no rows.")
    frame = frame.rename(columns={id_column: ID})
    frame[ID] = frame[ID].astype(str).str.strip()
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    invalid = frame[TARGET].notna() & target.isna()
    if invalid.any():
        raise DataError(f"The Target column of {name} must contain numbers (e.g. row {int(invalid.idxmax()) + 2}: "
                        f"{frame.loc[invalid.idxmax(), TARGET]!r}).")
    frame[TARGET] = target.astype(float)
    if frame[TIME].isna().any():
        raise DataError(f"The Time column of {name} has empty values.")
    return frame


def parse_time(train, test):
    """Integer steps or dates, the same in both files. Returns "integer" or "date"."""
    values = pd.concat([train[TIME], test[TIME]], ignore_index=True)
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.notna().all() and np.allclose(numbers, np.round(numbers)):
        for frame in (train, test):
            frame[TIME] = pd.to_numeric(frame[TIME]).round().astype("int64")
        return "integer"
    dates = pd.to_datetime(values.astype(str), errors="coerce")
    if dates.isna().any():
        bad = values[dates.isna()].iloc[0]
        raise DataError(f"The Time column must contain integer time steps or dates in both files (e.g. {bad!r} is neither).")
    for frame in (train, test):
        frame[TIME] = pd.to_datetime(frame[TIME].astype(str))
    return "date"


def check_duplicates(frame, name):
    duplicated = frame.duplicated([ID, TIME])
    if duplicated.any():
        row = frame[duplicated].iloc[0]
        raise DataError(f"{name} has several rows for the same series and time (ID {row[ID]}, Time {row[TIME]}).")


def _offset_unit(delta):
    seconds = int(delta.total_seconds())
    for unit, size in (("D", 86400), ("H", 3600), ("min", 60), ("S", 1)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return None


def infer_frequency(frames, kind):
    """The regular step of the series: an integer for integer time steps, else a pandas
    frequency alias (e.g. "MS", "D", "W-MON"). Missing time points are allowed."""
    series = [f.sort_values(TIME)[TIME].drop_duplicates() for frame in frames for _, f in frame.groupby(ID)]
    diffs = [s.diff().dropna() for s in series if len(s) > 1]
    if not diffs:
        raise DataError("Every series has a single time point: the series need several time points.")
    if kind == "integer":
        steps = np.concatenate([d.to_numpy() for d in diffs]).astype("int64")
        return int(np.gcd.reduce(steps[steps > 0])) or 1
    candidates = Counter(f for f in (pd.infer_freq(s) for s in series if len(s) >= 3) if f)
    ordered = [f for f, _ in candidates.most_common()]
    smallest = min(d.min() for d in diffs)
    ordered += ["MS", "M", "QS", "Q", "AS", "A"] + [u for u in [_offset_unit(smallest)] if u]
    for freq in ordered:
        try:
            if all(_on_grid(s, freq) for s in series):
                return freq
        except (ValueError, TypeError):
            continue
    raise DataError("The dates in the Time column are not at a regular frequency (e.g. daily, weekly, monthly). "
                    "Use regular dates or integer time steps (visit 1, 2, 3, ...).")


def _on_grid(dates, freq):
    grid = pd.date_range(dates.iloc[0], dates.iloc[-1], freq=freq)
    return len(grid) > 0 and grid[0] == dates.iloc[0] and dates.isin(grid).all()


def _grid(start, end, freq, kind):
    if kind == "integer":
        return pd.Index(np.arange(start, end + 1, freq), name=TIME)
    return pd.date_range(start, end, freq=freq, name=TIME)


def season_for(freq, kind):
    if kind == "integer":
        return 1
    base = "".join(ch for ch in str(freq).split("-")[0] if ch.isalpha()).upper()
    base = {"MS": "M", "QS": "Q", "AS": "A", "YS": "Y", "MIN": "H", "T": "H", "S": "H"}.get(base, base)
    return SEASONS.get(base, 1)


def horizon_for(freq, kind):
    if kind == "integer":
        return 6
    base = "".join(ch for ch in str(freq).split("-")[0] if ch.isalpha()).upper()
    base = {"MS": "M", "QS": "Q", "AS": "A", "YS": "Y"}.get(base, base)
    return HORIZONS.get(base, 6)


# ---------------------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------------------

def classify_covariates(train, test):
    """Static, time-varying and categorical covariates (decided on Train.csv)."""
    covariates = [c for c in train.columns if c not in (ID, TIME, TARGET)]
    missing = [c for c in covariates if c not in test.columns]
    if missing:
        raise DataError(f"Test.csv has no {', '.join(missing)} column{'s' if len(missing) > 1 else ''}: "
                        "both files need the same columns.")
    spec = {"static": [], "dynamic": [], "categorical": {}, "dropped": [], "warnings": []}
    for column in covariates:
        values = train[column]
        if values.isna().all():
            spec["dropped"].append(column)
            spec["warnings"].append(f"{column} is empty in Train.csv and is ignored.")
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        is_numeric = values.dtype != object or numeric[values.notna()].notna().all()
        if is_numeric:
            train[column] = numeric if values.dtype == object else values.astype(float)
            test[column] = pd.to_numeric(test[column], errors="coerce")
        else:
            levels = sorted(values.dropna().astype(str).unique())
            if len(levels) > MAX_CATEGORIES:
                spec["dropped"].append(column)
                spec["warnings"].append(f"{column} has {len(levels)} categories (at most {MAX_CATEGORIES}) and is ignored.")
                continue
            spec["categorical"][column] = levels
        constant = all((frame.groupby(ID)[column].nunique(dropna=True) <= 1).all() for frame in (train, test))
        spec["static" if constant else "dynamic"].append(column)
    return spec


def _safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def encoded_columns(spec, column):
    """The model columns of a covariate (one per category for a categorical one)."""
    if column in spec["categorical"]:
        return [f"{_safe_name(column)}_{_safe_name(level)}" for level in spec["categorical"][column]]
    return [column]


def encode(frame, spec):
    frame = frame.copy()
    for column, levels in spec["categorical"].items():
        values = frame[column].astype(str).where(frame[column].notna())
        for level, name in zip(levels, encoded_columns(spec, column)):
            frame[name] = (values == level).astype(float)
        frame = frame.drop(columns=column)
    return frame


# ---------------------------------------------------------------------------------------
# Regular series
# ---------------------------------------------------------------------------------------

def regularize(frame, freq, kind, dynamic):
    """One row per time step of every series: missing time points are added, with the
    Target interpolated (carried forward at the end) and the covariates carried forward.
    Returns the frame and the number of added points. ``observed`` marks the points with an
    observed Target (the only ones scored); leading points without a Target stay missing."""
    parts, added = [], 0
    for uid, part in frame.sort_values([ID, TIME]).groupby(ID, sort=False):
        part = part.set_index(TIME)
        grid = _grid(part.index[0], part.index[-1], freq, kind)
        full = part.reindex(grid)
        added += int(full[ID].isna().sum())
        full["observed"] = full[TARGET].notna()
        full[ID] = uid
        full[TARGET] = full[TARGET].interpolate(limit_area="inside").ffill()
        if dynamic:
            full[dynamic] = full[dynamic].ffill().bfill()
        parts.append(full.reset_index())
    return pd.concat(parts, ignore_index=True), added


# ---------------------------------------------------------------------------------------
# Upload summary
# ---------------------------------------------------------------------------------------

def load_files(train_path, test_path):
    train, test = read_table(train_path, "Train.csv"), read_table(test_path, "Test.csv")
    kind = parse_time(train, test)
    check_duplicates(train, "Train.csv")
    check_duplicates(test, "Test.csv")
    freq = infer_frequency([train, test], kind)
    return train, test, kind, freq


def summarize(train_path, test_path):
    """What the configuration page shows and needs: series, frequency, covariates and the
    largest possible horizon. ``errors`` lists the problems that block the run."""
    summary = {"errors": [], "warnings": []}
    try:
        train, test, kind, freq = load_files(train_path, test_path)
        spec = classify_covariates(train, test)
    except DataError as e:
        summary["errors"].append(str(e))
        return summary
    train = train.dropna(subset=[TARGET])
    lengths = train.groupby(ID)[TIME].agg(["min", "max", "count"])
    test_counts = test.groupby(ID)[TIME].count()
    continuing = sorted(set(test_counts.index) & set(lengths.index))
    new = sorted(set(test_counts.index) - set(lengths.index))
    # Horizon: every test series needs the H points to forecast plus at least one point of history
    limits = [int(test_counts[i]) for i in continuing] + [int(test_counts[i]) - 1 for i in new]
    max_horizon = max(0, min(limits)) if limits else 0
    # Series lengths on the regular grid (the time span, missing points included)
    spans = [len(_grid(row["min"], row["max"], freq, kind)) for _, row in lengths.iterrows()]
    if max_horizon < 1:
        summary["errors"].append("Every new series of Test.csv needs at least 2 time points (history and the points to forecast).")
    overlap = [i for i in continuing
               if test.loc[test[ID] == i, TIME].min() <= lengths.loc[i, "max"]]
    if overlap:
        summary["errors"].append(f"Test.csv repeats time points of Train.csv for {len(overlap)} series (e.g. ID {overlap[0]}): "
                                 "a series continued in Test.csv must start after its last Train.csv time point.")
    summary["warnings"] += spec["warnings"]
    if len(lengths) < 2:
        summary["warnings"].append("Train.csv has a single series: the models learn from many series best.")
    summary.update({
        "kind": kind,
        "freq": str(freq),
        "train_series": int(len(lengths)),
        "train_rows": int(len(train)),
        "test_series": int(len(test_counts)),
        "test_rows": int(len(test)),
        "continuing_series": len(continuing),
        "new_series": len(new),
        "length_min": int(min(spans)),
        "length_median": int(np.median(spans)),
        "length_max": int(max(spans)),
        "start": str(lengths["min"].min()),
        "end": str(max(lengths["max"].max(), test[TIME].max())),
        "static": spec["static"],
        "dynamic": spec["dynamic"],
        "categorical": sorted(spec["categorical"]),
        "dropped": spec["dropped"],
        "max_horizon": int(max_horizon),
        "suggested_horizon": int(max(1, min(horizon_for(freq, kind), max_horizon, max(1, min(spans) // 3)))),
        "season": season_for(freq, kind),
        "target_missing": int(train[TARGET].isna().sum()),
    })
    return summary


def max_folds(length_min, horizon):
    """Largest number of rolling-origin folds: every training series keeps at least one
    horizon of points before its first validation window."""
    return max(0, length_min // max(1, horizon) - 1)


# ---------------------------------------------------------------------------------------
# Frames for neuralforecast
# ---------------------------------------------------------------------------------------

@dataclass
class Prepared:
    """Data ready for neuralforecast (columns unique_id, ds, y and the covariates)."""
    train: pd.DataFrame
    test_history: pd.DataFrame
    test_horizon: pd.DataFrame          # the points to forecast: ``y`` actual value, ``observed`` False if filled in
    static: Optional[pd.DataFrame]      # unique_id and the standardised static features
    futr: List[str]
    hist: List[str]
    stat: List[str]
    freq: object
    kind: str
    horizon: int
    season: int
    spec: dict
    scaling: Dict[str, list] = field(default_factory=dict)
    added_points: int = 0
    continuing: List[str] = field(default_factory=list)
    new: List[str] = field(default_factory=list)


def _to_nf(frame):
    return frame.rename(columns={ID: "unique_id", TIME: "ds", TARGET: "y"})


def prepare(train_path, test_path, horizon, future_columns=(), season=None):
    """Train.csv and Test.csv as neuralforecast frames. ``future_columns`` are the
    time-varying covariates known in advance; the others are past covariates."""
    train, test, kind, freq = load_files(train_path, test_path)
    spec = classify_covariates(train, test)
    columns = [ID, TIME, TARGET] + spec["static"] + spec["dynamic"]
    train, test = encode(train[columns], spec), encode(test[columns], spec)
    futr = [n for c in spec["dynamic"] if c in future_columns for n in encoded_columns(spec, c)]
    hist = [n for c in spec["dynamic"] if c not in future_columns for n in encoded_columns(spec, c)]
    stat = [n for c in spec["static"] for n in encoded_columns(spec, c)]
    dynamic = futr + hist

    # Test series: history (with the Train.csv points of a continued series) and horizon
    train_ids = set(train[ID])
    histories, horizons, continuing, new = [], [], [], []
    for uid, part in test.sort_values([ID, TIME]).groupby(ID, sort=False):
        if uid in train_ids:
            continuing.append(uid)
            earlier = train[(train[ID] == uid) & (train[TIME] < part[TIME].iloc[0])]
            if (train.loc[train[ID] == uid, TIME] >= part[TIME].iloc[0]).any():
                raise DataError(f"Test.csv repeats time points of Train.csv for ID {uid}: a series continued in "
                                "Test.csv must start after its last Train.csv time point.")
            part = pd.concat([earlier, part], ignore_index=True)
        else:
            new.append(uid)
        full, _ = regularize(part, freq, kind, dynamic)
        if len(full) <= horizon:
            raise DataError(f"Series {uid} of Test.csv has {len(full)} time points: it needs more than the horizon "
                            f"({horizon}) to have a history to forecast from.")
        future = full.iloc[-horizon:].copy()  # ``observed`` False: no Target there, not scored
        history = full.iloc[:-horizon].copy()
        # The history is filled from its own points only: the horizon must not leak into it
        history[TARGET] = history[TARGET].where(history["observed"]).interpolate(limit_area="inside").ffill()
        histories.append(history)
        horizons.append(future)
    test_history = pd.concat(histories, ignore_index=True).dropna(subset=[TARGET])
    test_horizon = pd.concat(horizons, ignore_index=True)
    if test_history[TARGET].isna().all():
        raise DataError("The test series have no Target values before their horizon.")

    train, added = regularize(train, freq, kind, dynamic)
    train = train.dropna(subset=[TARGET])

    # Covariates: missing values filled with the training mean; static features standardised
    means = {c: float(train[c].mean()) if train[c].notna().any() else 0.0 for c in dynamic + stat}
    for frame in (train, test_history, test_horizon):
        for column in dynamic + stat:
            frame[column] = frame[column].fillna(means[column])
    scaling = {}
    static = None
    if stat:
        rows = pd.concat([train, test_history]).groupby(ID, sort=False)[stat].first()
        per_series = train.groupby(ID)[stat].first()
        for column in stat:
            mean, std = float(per_series[column].mean()), float(per_series[column].std(ddof=0))
            scaling[column] = [mean, std if math.isfinite(std) and std > 0 else 1.0]
            rows[column] = (rows[column] - scaling[column][0]) / scaling[column][1]
        static = rows.reset_index().rename(columns={ID: "unique_id"})

    keep = [ID, TIME, TARGET] + dynamic
    return Prepared(
        train=_to_nf(train[keep]), test_history=_to_nf(test_history[keep]),
        test_horizon=_to_nf(test_horizon[keep + ["observed"]]),
        static=static, futr=futr, hist=hist, stat=stat, freq=freq, kind=kind, horizon=int(horizon),
        season=int(season or season_for(freq, kind)), spec=spec, scaling=scaling, added_points=added,
        continuing=continuing, new=new,
    )
