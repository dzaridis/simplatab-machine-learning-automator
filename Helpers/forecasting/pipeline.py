"""The forecasting pipeline, following the other automators step by step:
1. data: Train.csv and Test.csv as regular series with their covariates (data.py);
2. rolling-origin (prequential) validation on Train.csv: K windows of H points at the end of
   every series; for each window, a network is trained on the points before it and forecasts
   it. With tuning, the configuration (lookback, learning rate, size) with the lowest mean MAE
   over the windows is kept;
3. final networks trained on all of Train.csv forecast the last H points of every Test.csv
   series from the points before them: errors, forecasts and figures, integrated gradients,
   forecasts beyond the data and the saved models.
Outputs go to ./Materials. Progress is printed in the format that web/jobs.py follows.
"""
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from . import metrics as fm
from . import plots
from .data import ID, TARGET, TIME, DataError, prepare
from .models import BY_KEY, build, search_space

MATERIALS = "Materials"
MAX_EXPLAINED_SERIES = 30
FORMAT = "simplatab-forecaster"


def _banner(text):
    print("------------- \n", f"{text} \n", "-------------")


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def _quiet():
    """Lightning reports every fit (seed, devices): keep the run log readable. Its loggers
    are configured when it is imported, so import it first."""
    import pytorch_lightning  # noqa: F401
    for name in ("lightning", "lightning.pytorch", "lightning.fabric", "lightning_fabric", "pytorch_lightning",
                 "neuralforecast", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def _short(error):
    text = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    return text[:300]


def _accelerator():
    import torch
    return "gpu" if torch.cuda.is_available() else "cpu"


def describe_device():
    import torch
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return f"CPU ({os.cpu_count()} cores)"


def run_forecasting_pipeline(input_folder, params):
    _quiet()
    try:
        return _run(input_folder, params)
    except DataError as e:
        print(f"Error: {e}")
        return f"Error: {e}"
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"


# ---------------------------------------------------------------------------------------
# Validation windows
# ---------------------------------------------------------------------------------------

def make_folds(train, folds, horizon):
    """Rolling-origin windows: fold k trains on every series up to (K - k + 1) horizons before
    its end and is scored on the next horizon. A series too short for a fold (less than one
    horizon of training points) is left out of it."""
    series = [part.reset_index(drop=True) for _, part in train.groupby("unique_id", sort=False)]
    windows = []
    for k in range(folds):
        cut = (folds - k) * horizon
        fit, score = [], []
        for part in series:
            if len(part) - cut >= horizon:
                fit.append(part.iloc[:len(part) - cut])
                score.append(part.iloc[len(part) - cut:len(part) - cut + horizon])
        if not fit:
            raise DataError(f"No training series is long enough for {folds} validation windows of {horizon} points: "
                            "reduce the number of folds or the horizon.")
        windows.append((pd.concat(fit, ignore_index=True), pd.concat(score, ignore_index=True)))
    return windows


def _static(data, frame):
    if data.static is None:
        return None
    return data.static[data.static["unique_id"].isin(frame["unique_id"].unique())].reset_index(drop=True)


def _future_frame(data, frame):
    return frame[["unique_id", "ds"] + data.futr] if data.futr else None


def _fit(model, config, data, train, max_steps):
    from neuralforecast import NeuralForecast
    network = build(model, data.horizon, config, data.futr, data.hist, data.stat, max_steps, accelerator=_accelerator())
    nf = NeuralForecast(models=[network], freq=data.freq)
    nf.fit(df=train, static_df=_static(data, train))
    return nf


def _forecast(nf, model, data, history, future):
    """Forecasts of the H points after each series of ``history`` (``future`` gives their
    timestamps and future covariates), aligned with the rows of ``future``."""
    predicted = nf.predict(df=history, static_df=_static(data, history), futr_df=_future_frame(data, future))
    predicted = predicted.reset_index() if "unique_id" not in predicted.columns else predicted
    merged = future[["unique_id", "ds"]].merge(predicted[["unique_id", "ds", model.name]], on=["unique_id", "ds"], how="left")
    return merged[model.name].to_numpy()


def _scores(frame, column, scale_by_series):
    mask = frame["observed"].to_numpy() if "observed" in frame else np.ones(len(frame), bool)
    return fm.score(frame["y"].to_numpy()[mask], frame[column].to_numpy()[mask],
                    frame["unique_id"].to_numpy()[mask], scale_by_series)


def _validate(model, config, data, windows, max_steps):
    results = []
    for fit, score in windows:
        nf = _fit(model, config, data, fit, max_steps)
        frame = score.copy()
        frame["forecast"] = _forecast(nf, model, data, fit, score)
        results.append(_scores(frame, "forecast", fm.scales(fit, data.season)))
    return results


def _describe(config):
    parts = [f"lookback {config['input_size']}", f"learning rate {config['learning_rate']:g}"]
    parts += [f"{k.replace('_', ' ')} {v}" for k, v in config.items() if k not in ("input_size", "learning_rate")]
    return ", ".join(parts)


def _mean_sd(results):
    table = pd.DataFrame(results)
    return {m: f"{table[m].mean():.3f} ± {table[m].std(ddof=0):.3f}" for m in fm.METRICS}


# ---------------------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------------------

def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _used(explanation, key, columns, ndim):
    """Attributions of the covariates the network uses (others are absent or empty)."""
    return bool(columns) and key in explanation and _array(explanation[key]).ndim == ndim


def summarize_attributions(explanation, lookback, horizon, data):
    """Mean |integrated gradients| per input over time, and each input's share of the total.
    Arrays: [series, horizon, n_series, outputs, time, features] (insample: target, mask)."""
    temporal_cols = list(range(-lookback, horizon))
    rows, importance = {}, {}
    target = np.abs(_array(explanation["insample"]))[..., 0].mean(axis=(0, 1, 2, 3))
    rows["Target (history)"] = np.concatenate([target[-lookback:], np.full(horizon, np.nan)])
    if _used(explanation, "hist_exog", data.hist, 6):
        values = np.abs(_array(explanation["hist_exog"])).mean(axis=(0, 1, 2, 3))
        for j, column in enumerate(data.hist):
            rows[f"{column} (past)"] = np.concatenate([values[-lookback:, j], np.full(horizon, np.nan)])
    if _used(explanation, "futr_exog", data.futr, 6):
        values = np.abs(_array(explanation["futr_exog"])).mean(axis=(0, 1, 2, 3))
        for j, column in enumerate(data.futr):
            rows[f"{column} (future)"] = values[-(lookback + horizon):, j]
    for name, row in rows.items():
        importance[name] = float(np.nansum(row))
    if _used(explanation, "stat_exog", data.stat, 5):
        values = np.abs(_array(explanation["stat_exog"])).mean(axis=(0, 1, 2, 3))
        for j, column in enumerate(data.stat):
            importance[f"{column} (static)"] = float(values[j])
    total = sum(importance.values()) or 1.0
    importance = pd.Series({k: 100 * v / total for k, v in importance.items()})
    temporal = pd.DataFrame(rows, index=temporal_cols).T
    return importance, temporal


def explain(nf, model, config, data, history, future, folder):
    ids = plots.pick_series(history["unique_id"].unique(), MAX_EXPLAINED_SERIES)
    part = history[history["unique_id"].isin(ids)]
    ahead = future[future["unique_id"].isin(ids)]
    _, explanations = nf.explain(df=part, static_df=_static(data, part), futr_df=_future_frame(data, ahead),
                                 explainer="IntegratedGradients", verbose=False)
    importance, temporal = summarize_attributions(explanations[model.name], config["input_size"], data.horizon, data)
    stem = os.path.join(folder, _safe(model.name))
    importance.sort_values(ascending=False).rename("share_of_attribution_percent").to_csv(f"{stem}_feature_importance.csv",
                                                                                           index_label="input")
    plots.attributions(importance, temporal.to_numpy(), list(temporal.index), [str(c) for c in temporal.columns],
                       f"{model.name}: integrated gradients on {len(ids)} test series", f"{stem}_integrated_gradients.png")


# ---------------------------------------------------------------------------------------
# Saved models
# ---------------------------------------------------------------------------------------

def model_info(model, config, data):
    spec = data.spec
    return {
        "format": FORMAT,
        "version": 1,
        "model": model.name,
        "horizon": data.horizon,
        "freq": data.freq if data.kind == "integer" else str(data.freq),
        "time": data.kind,
        "season": data.season,
        "config": config,
        "future_covariates": data.futr,
        "past_covariates": data.hist,
        "static_features": data.stat,
        # One-hot encoded columns: original column -> [[category, model column], ...]
        "categories": {column: [[level, f"{_name(column)}_{_name(level)}"] for level in levels]
                       for column, levels in spec["categorical"].items()},
        # Static features are standardised: (value - mean) / std
        "static_scaling": data.scaling,
    }


def _name(text):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def save_model(nf, model, config, data, path):
    """The neuralforecast model folder and simplatab.json, zipped (``<name>/`` inside)."""
    workdir = tempfile.mkdtemp()
    try:
        folder = os.path.join(workdir, _safe(model.name))
        nf.save(folder, save_dataset=False, overwrite=True)
        with open(os.path.join(folder, "simplatab.json"), "w") as f:
            json.dump(model_info(model, config, data), f, indent=2)
        shutil.make_archive(path[:-4], "zip", workdir, _safe(model.name))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------

def _run(input_folder, params):
    started = time.time()
    _banner("Loading Data")
    horizon, folds = int(params["horizon"]), int(params["k_folds"])
    data = prepare(os.path.join(input_folder, "Train.csv"), os.path.join(input_folder, "Test.csv"), horizon,
                   params.get("future_columns", []), params.get("season"))
    models = [BY_KEY[key] for key in params["models"]]
    train_ids = data.train["unique_id"].nunique()
    print(f"Train.csv: {train_ids} series · Test.csv: {len(data.continuing)} continued and {len(data.new)} new series · "
          f"frequency {data.freq} · horizon {horizon} · seasonal period {data.season}")
    print(f"Covariates: future {data.futr or 'none'} · past {data.hist or 'none'} · static {data.stat or 'none'}")
    if data.added_points:
        print(f"{data.added_points} missing time points of Train.csv were filled in (Target interpolated)")
    print(f"Device: {describe_device()}")
    for folder in ("Models", "Forecasts", "Forecast_Plots", "Metrics_Plots", "Explainability"):
        os.makedirs(os.path.join(MATERIALS, folder), exist_ok=True)

    # ---- Rolling-origin validation ------------------------------------------------------
    windows = make_folds(data.train, folds, horizon)
    lengths = windows[0][0].groupby("unique_id").size()
    longest = int(lengths.median())  # lookback candidates up to the typical history of the first window
    _banner(f"Training on K-Fold cross validation (rolling origin: {folds} windows of {horizon} points)")
    max_steps, trials = int(params.get("max_steps", 500)), int(params.get("trials", 0))
    kfold, configs, skipped = {}, {}, []
    for model in models:
        print(f"{model.name} is starting")
        try:
            t0 = time.time()
            candidates = search_space(model, horizon, longest, params.get("lookback") or None, trials, seed=0)
            best = None
            for i, config in enumerate(candidates):
                results = _validate(model, config, data, windows, max_steps)
                mae = float(np.nanmean([r["MAE"] for r in results]))
                if len(candidates) > 1:
                    print(f"{model.name}: configuration {i + 1}/{len(candidates)} ({_describe(config)}): validation MAE {mae:.4g}")
                if best is None or mae < best[0]:
                    best = (mae, config, results)
            configs[model.name], kfold[model.name] = best[1], best[2]
            print(f"{model.name}: {_describe(best[1])} · {time.time() - t0:.0f} s")
            print(f"{model.name} is completed successfully")
        except Exception as e:
            print(f"{model.name} failed and was skipped: {_short(e)}")
            skipped.append({"model": model.name, "reason": _short(e)})
    if not kfold:
        return "Error: every model failed during the validation (see the log)."
    kfold[fm.BASELINE] = []
    for fit, score in windows:
        frame = score.copy()
        frame["forecast"] = fm.baseline_forecasts(fit, score, data.season)
        kfold[fm.BASELINE].append(_scores(frame, "forecast", fm.scales(fit, data.season)))
    pd.DataFrame({name: _mean_sd(results) for name, results in kfold.items()}).T[fm.METRICS] \
        .to_excel(os.path.join(MATERIALS, f"{folds}_fold_results.xlsx"))
    pd.DataFrame([dict(model=name, window=i + 1, **r) for name, results in kfold.items() for i, r in enumerate(results)]) \
        .to_csv(os.path.join(MATERIALS, "Forecasts", "validation_windows.csv"), index=False)
    _banner("Training on K-Fold cross validation completed successfully")

    # ---- Final models on Test.csv ---------------------------------------------------------
    _banner("Evaluating algorithms on Test.csv")
    history, future = data.test_history, data.test_horizon.copy()
    future["step"] = future.groupby("unique_id").cumcount() + 1
    future[fm.BASELINE] = fm.baseline_forecasts(history, future, data.season)
    test_scales = fm.scales(history, data.season)
    beyond = None
    if not data.futr:  # forecasts after the last observed point (future covariates are unknown there)
        full = pd.concat([history, data.test_horizon[history.columns]], ignore_index=True).sort_values(["unique_id", "ds"])
        beyond = []
    notes = []
    trained = [m for m in models if m.name in configs]
    for model in trained:
        print(f"{model.name} is starting")
        try:
            config = configs[model.name]
            nf = _fit(model, config, data, data.train, max_steps)
            future[model.name] = _forecast(nf, model, data, history, future)
            if beyond is not None:
                predicted = nf.predict(df=full, static_df=_static(data, full))
                predicted = predicted.reset_index() if "unique_id" not in predicted.columns else predicted
                beyond.append(predicted[["unique_id", "ds", model.name]])
            save_model(nf, model, config, data, os.path.join(MATERIALS, "Models", f"{_safe(model.name)}.zip"))
            try:
                explain(nf, model, config, data, history, future, os.path.join(MATERIALS, "Explainability"))
            except Exception as e:
                notes.append(f"{model.name}: integrated gradients could not be computed ({_short(e)})")
                print(notes[-1])
            print(f"{model.name} is completed successfully")
        except Exception as e:
            print(f"{model.name} failed and was skipped: {_short(e)}")
            skipped.append({"model": model.name, "reason": _short(e)})
    done = [m.name for m in trained if m.name in future.columns]
    if not done:
        return "Error: every model failed on the test set (see the log)."

    names = done + [fm.BASELINE]
    test = pd.DataFrame({name: _scores(future, name, test_scales) for name in names}).T[fm.METRICS]
    test.to_excel(os.path.join(MATERIALS, "test_results.xlsx"))
    best = test.loc[done, "MAE"].idxmin()

    # ---- Forecasts, figures ---------------------------------------------------------------
    table = future.rename(columns={"unique_id": ID, "ds": TIME, "y": TARGET})
    table.loc[~table["observed"], TARGET] = np.nan  # filled-in points have no observed value
    table[[ID, TIME, "step", TARGET] + names].to_csv(os.path.join(MATERIALS, "Forecasts", "test_forecasts.csv"), index=False)
    if beyond:
        merged = beyond[0]
        for frame in beyond[1:]:
            merged = merged.merge(frame, on=["unique_id", "ds"], how="outer")
        merged["step"] = merged.groupby("unique_id").cumcount() + 1
        merged.rename(columns={"unique_id": ID, "ds": TIME})[[ID, TIME, "step"] + [n for n in done if n in merged]] \
            .to_csv(os.path.join(MATERIALS, "Forecasts", "future_forecasts.csv"), index=False)
    ids = plots.pick_series(future["unique_id"].unique())
    context = min(60, max(3 * horizon, max(c["input_size"] for c in configs.values())))
    for name in done:
        plots.forecast_grid(history, future, name, ids, f"{name}: forecasts of the test series vs. observed values",
                            os.path.join(MATERIALS, "Forecast_Plots", f"{_safe(name)}_test_forecasts.png"), context)
        if beyond:
            ahead = merged[merged["unique_id"].isin(ids)][["unique_id", "ds", name]]
            plots.forecast_grid(full, ahead, name, ids, f"{name}: forecasts beyond the last observed point",
                                os.path.join(MATERIALS, "Forecast_Plots", f"{_safe(name)}_future_forecasts.png"), context)
    plots.metric_bars(test, best, os.path.join(MATERIALS, "Metrics_Plots", "test_metrics.png"), fm.BASELINE)
    steps = pd.DataFrame({name: fm.error_by_step(future.loc[future["observed"], "y"], future.loc[future["observed"], name],
                                                 future.loc[future["observed"], "step"]) for name in names}).T
    steps.to_csv(os.path.join(MATERIALS, "Metrics_Plots", "mae_by_horizon_step.csv"), index_label="model")
    plots.heatmap(steps.to_numpy(), list(steps.index), [str(c) for c in steps.columns],
                  "Test MAE at each horizon step", os.path.join(MATERIALS, "Metrics_Plots", "error_by_horizon.png"),
                  xlabel="Horizon step")

    info = {
        "automator": "time-series-forecasting", "horizon": horizon, "k_folds": folds, "freq": str(data.freq),
        "time": data.kind, "season": data.season, "train_series": int(train_ids),
        "continuing_series": len(data.continuing), "new_series": len(data.new), "added_points": data.added_points,
        "future_covariates": data.futr, "past_covariates": data.hist, "static_features": data.stat,
        "future_columns": [c for c in data.spec["dynamic"] if c in params.get("future_columns", [])],
        "categorical": sorted(data.spec["categorical"]), "best_model": best, "configs": configs,
        "max_steps": max_steps, "trials": trials, "device": describe_device(), "skipped": skipped,
        "notes": notes, "future_forecasts": bool(beyond), "minutes": round((time.time() - started) / 60, 1),
    }
    with open(os.path.join(MATERIALS, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"Best model on Test.csv (MAE): {best}")
    print("Pipeline completed successfully.")
    return "Pipeline completed successfully"
