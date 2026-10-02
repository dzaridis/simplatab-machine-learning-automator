"""Data of the survival analysis automator: Train.csv and Test.csv, one row per patient (or unit).

- ``Time``: the follow-up time, a positive number (days, months, years: any unit, the same in
  both files): the time of the event, or of the last follow-up when the event was not observed;
- ``Event``: 1 if the event (death, relapse, progression...) happened at ``Time``, 0 if the
  patient was censored (still event-free at the last follow-up);
- ``ID`` (or ``patient_id``): optional identifier, never a feature;
- every other column is a feature: numeric, or categorical (one-hot encoded). Missing values are
  imputed; constant, mostly missing or identifier-like text columns are left out (as for clustering).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from Helpers.clustering.data import ID_COLUMNS, clean, feature_columns, id_column, make_featurizer, read, DataError

TIME, EVENT = "Time", "Event"
MIN_EVENTS = 10
MAX_ROWS = 100_000
EVENT_VALUES = {"1": 1, "0": 0, "1.0": 1, "0.0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}

__all__ = ["DataError", "TIME", "EVENT", "summarize", "prepare", "events_of", "make_featurizer", "clean"]


def events_of(series):
    """Event indicators as 0/1, None for a value that is not one."""
    out = []
    for v in series:
        key = str(v).strip().lower()
        out.append(EVENT_VALUES.get(key))
    return out


def _check_outcome(frame, name, errors):
    for column in (TIME, EVENT):
        if column not in frame.columns:
            errors.append(f"{name} has no {column} column (follow-up time and event indicator are required).")
    if errors:
        return None, None
    time = pd.to_numeric(frame[TIME], errors="coerce")
    if time.isna().any():
        errors.append(f"{name}: {int(time.isna().sum())} Time value(s) are missing or not numbers.")
    elif (time <= 0).any():
        errors.append(f"{name}: Time must be positive ({int((time <= 0).sum())} value(s) are 0 or negative).")
    events = events_of(frame[EVENT])
    bad = sum(e is None for e in events)
    if bad:
        errors.append(f"{name}: Event must be 1 (event) or 0 (censored); {bad} value(s) are not.")
    return time.to_numpy(dtype=float), np.array([e if e is not None else 0 for e in events], dtype=int)


def km_median(time, event):
    """Kaplan-Meier median of the times (None when the curve stays above 0.5)."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    at_risk = len(t) - np.arange(len(t))
    surv = np.cumprod(1 - e / at_risk)
    below = np.where(surv <= 0.5)[0]
    return float(t[below[0]]) if len(below) else None


def summarize(train_path, test_path):
    errors, warnings = [], []
    try:
        train, test = read(train_path), read(test_path)
    except DataError as e:
        return {"errors": [str(e)], "warnings": []}
    time, event = _check_outcome(train, "Train.csv", errors)
    _check_outcome(test, "Test.csv", errors)
    features = train.drop(columns=[c for c in (TIME, EVENT) if c in train.columns])
    numeric, categorical, dropped = feature_columns(features)
    summary = {"train_rows": len(train), "test_rows": len(test), "id_column": id_column(train),
               "numeric": numeric, "categorical": categorical, "features": numeric + categorical,
               "dropped": [{"column": c, "reason": r} for c, r in dropped]}
    if not numeric and not categorical:
        errors.append("Train.csv has no usable feature column (besides ID, Time and Event).")
    if len(train) > MAX_ROWS:
        errors.append(f"Train.csv has {len(train):,} rows: at most {MAX_ROWS:,}.")
    missing = [c for c in numeric + categorical if c not in test.columns]
    if missing:
        errors.append(f"Test.csv lacks the feature column(s) {', '.join(missing)} of Train.csv.")
    if dropped:
        warnings.append("Columns left out: " + "; ".join(f"{c} ({r})" for c, r in dropped) + ".")
    if time is not None and not errors:
        n_events = int(event.sum())
        if n_events < MIN_EVENTS:
            errors.append(f"Train.csv has {n_events} events: survival models need at least {MIN_EVENTS}.")
        event_times = time[event == 1]
        quantiles = np.quantile(event_times, [0.25, 0.5, 0.75]) if n_events else np.array([])
        summary.update(
            events=n_events, censored=int(len(event) - n_events), censoring_rate=round(100 * (1 - event.mean()), 1),
            time_min=float(time.min()), time_max=float(time.max()),
            median_survival=km_median(time, event),
            median_follow_up=km_median(time, 1 - event),   # reverse Kaplan-Meier
            suggested_horizons=[float(_round(q)) for q in quantiles],
            max_horizon=float(np.quantile(event_times, 0.9)) if n_events else None,
            max_folds=int(max(2, min(10, n_events // 3))),
        )
        if summary["censoring_rate"] > 80:
            warnings.append(f"{summary['censoring_rate']}% of Train.csv is censored: estimates at late times are uncertain.")
        miss = int(features[numeric + categorical].isna().sum().sum())
        if miss:
            warnings.append(f"{miss} missing feature value(s): imputed (median, or a 'missing' category).")
        if summary["id_column"] is None:
            warnings.append("No ID (or patient_id) column: patients are identified by their row in Train.csv.")
    summary["errors"], summary["warnings"] = errors, warnings
    return summary


def _round(value):
    """A readable horizon: 2 significant digits."""
    if value <= 0:
        return value
    digits = int(np.floor(np.log10(value)))
    return round(value, max(0, 1 - digits))


def prepare(train_path, test_path, params):
    summary = summarize(train_path, test_path)
    if summary["errors"]:
        raise DataError(" ".join(summary["errors"]))
    train, test = read(train_path), read(test_path)
    ignore = set(params.get("ignore_columns") or [])
    numeric, categorical, dropped = feature_columns(train.drop(columns=[TIME, EVENT]), ignore=ignore)
    if not numeric and not categorical:
        raise DataError("No feature column is left: do not ignore every column.")
    ident = id_column(train)
    test_ident = id_column(test)
    return SimpleNamespace(
        train=train, test=test, numeric=numeric, categorical=categorical, dropped=dropped, summary=summary,
        F_train=clean(train, numeric, categorical), F_test=clean(test, numeric, categorical),
        time_train=pd.to_numeric(train[TIME]).to_numpy(float), event_train=np.array(events_of(train[EVENT]), int),
        time_test=pd.to_numeric(test[TIME]).to_numpy(float), event_test=np.array(events_of(test[EVENT]), int),
        ids_train=[str(v) for v in train[ident]] if ident else [str(i + 1) for i in range(len(train))],
        ids_test=[str(v) for v in test[test_ident]] if test_ident else [str(i + 1) for i in range(len(test))],
        id_column=ident, test_id_column=test_ident)
