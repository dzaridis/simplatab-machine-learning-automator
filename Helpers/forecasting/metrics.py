"""Forecast errors and the seasonal naive baseline.

- MAE, RMSE: in the unit of the Target, over all the forecast points.
- sMAPE: symmetric mean absolute percentage error (0-200%), defined when the values are 0.
- MASE: the MAE of each series divided by the in-sample MAE of the seasonal naive forecast on
  its history, averaged over the series; below 1, the model beats the seasonal naive forecast.
"""
import numpy as np
import pandas as pd

METRICS = ["MAE", "RMSE", "sMAPE", "MASE"]
BASELINE = "Seasonal naive"


def seasonal_naive(history, horizon, season):
    """Repeats the last season of the history (the last value if the history is shorter)."""
    values = np.asarray(history, dtype=float)
    season = season if len(values) >= season else 1
    last = values[-season:]
    return np.array([last[i % season] for i in range(horizon)])


def naive_scale(history, season):
    """In-sample MAE of the seasonal naive forecast (NaN if undefined)."""
    values = np.asarray(history, dtype=float)
    for m in (season, 1):
        if len(values) > m:
            scale = np.mean(np.abs(values[m:] - values[:-m]))
            if scale > 0:
                return float(scale)
    return np.nan


def baseline_forecasts(history, horizon_frame, season):
    """Seasonal naive forecasts for the rows of ``horizon_frame`` (unique_id, ds)."""
    out = []
    for uid, part in horizon_frame.groupby("unique_id", sort=False):
        past = history.loc[history["unique_id"] == uid, "y"].to_numpy()
        out.append(pd.Series(seasonal_naive(past, len(part), season), index=part.index))
    return pd.concat(out).reindex(horizon_frame.index)


def scales(history, season):
    return {uid: naive_scale(part["y"].to_numpy(), season) for uid, part in history.groupby("unique_id", sort=False)}


def score(actual, predicted, series, scale_by_series):
    """The four metrics. ``series`` gives the series of each point (for MASE)."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    ok = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted, series = actual[ok], predicted[ok], np.asarray(series)[ok]
    if not len(actual):
        return {m: np.nan for m in METRICS}
    error = np.abs(actual - predicted)
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.where(denominator > 0, 200 * error / np.where(denominator > 0, denominator, 1), 0.0)
    ratios = []
    for uid in pd.unique(series):
        scale = scale_by_series.get(uid, np.nan)
        if np.isfinite(scale):
            ratios.append(error[series == uid].mean() / scale)
    return {"MAE": float(error.mean()), "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "sMAPE": float(smape.mean()), "MASE": float(np.mean(ratios)) if ratios else np.nan}


def error_by_step(actual, predicted, steps):
    """MAE at each horizon step (1..H)."""
    frame = pd.DataFrame({"error": np.abs(np.asarray(actual, float) - np.asarray(predicted, float)), "step": steps})
    return frame.dropna().groupby("step")["error"].mean()
