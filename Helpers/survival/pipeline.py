"""The survival analysis pipeline, following the other automators step by step:
1. data: Train.csv and Test.csv with Time, Event and the features (data.py);
2. stratified K-fold cross-validation on Train.csv (folds stratified on the event indicator; the
   preprocessing is fitted inside each fold): C-index, Uno's C-index, integrated Brier score and
   the time-dependent AUC and Brier score at the horizons, on the held-out fold;
3. final models trained on all of Train.csv, evaluated on Test.csv, with a Kaplan-Meier
   (no covariates) reference: metrics, risk groups (tertiles of the training risk) with their
   Kaplan-Meier curves and log-rank test, calibration at a horizon, predicted survival curves,
   permutation importance of every feature and the saved models.
Outputs go to ./Materials. Progress is printed in the format that web/jobs.py follows.
"""
import json
import os
import time as clock
import traceback
import warnings

import numpy as np
import pandas as pd

from . import metrics as sm
from . import plots
from .data import DataError, EVENT, TIME, make_featurizer, prepare
from .models import BY_KEY, SimplatabSurvival, build, requirements, save
from Helpers.splits import index_rows, write_splits

MATERIALS = "Materials"
KM_NAME = "Kaplan-Meier (no covariates)"
IMPORTANCE_REPEATS = 3


def _banner(text):
    print("------------- \n", f"{text} \n", "-------------")


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def _short(error):
    text = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    return text[:300]


def describe_device():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)} (the networks run on CPU: tabular data)"
    except Exception:
        pass
    return f"CPU ({os.cpu_count()} cores)"


def run_survival_pipeline(input_folder, params):
    for category in (UserWarning, FutureWarning, DeprecationWarning, RuntimeWarning):
        warnings.filterwarnings("ignore", category=category)
    try:
        return _run(input_folder, params)
    except DataError as e:
        print(f"Error: {e}")
        return f"Error: {e}"
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"


def _fit(key, params, frame, time, event, numeric, categorical):
    featurizer = make_featurizer(numeric, categorical, "standard")
    X = np.asarray(featurizer.fit_transform(frame), float)
    return featurizer, build(key, params).fit(X, time, event)


def _evaluate(model, X, time, event, censoring, grid, horizons):
    return sm.score(lambda times: model.predict_survival(X, times), model.predict_risk(X), time, event, censoring,
                    grid, horizons)


def _mean_sd(results, columns):
    table = pd.DataFrame(results)
    return {m: (f"{table[m].mean():.3f} ± {table[m].std(ddof=0):.3f}" if m in table and table[m].notna().any() else "–")
            for m in columns}


class _KaplanMeier:
    """The no-covariate reference: the same Kaplan-Meier curve for every patient."""

    def fit(self, X, time, event):
        self.times_, self.surv_ = sm.kaplan_meier(time, event)
        return self

    def predict_risk(self, X):
        return np.zeros(len(X))

    def predict_survival(self, X, times):
        return np.tile(sm.step(self.times_, self.surv_, np.asarray(times, float)), (len(X), 1))


def permutation_importance(model, featurizer, frame, time, event, columns, seed):
    """Drop of the C-index when each original column is shuffled (mean, SD over repeats)."""
    rng = np.random.default_rng(seed)
    base = sm.harrell_c(time, event, model.predict_risk(np.asarray(featurizer.transform(frame), float)))
    rows = {}
    for column in columns:
        drops = []
        for _ in range(IMPORTANCE_REPEATS):
            shuffled = frame.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            risk = model.predict_risk(np.asarray(featurizer.transform(shuffled), float))
            drops.append(base - sm.harrell_c(time, event, risk))
        rows[column] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return pd.DataFrame(rows).T.sort_values("mean", ascending=False)


def _run(input_folder, params):
    started = clock.time()
    _banner("Loading Data")
    data = prepare(os.path.join(input_folder, "Train.csv"), os.path.join(input_folder, "Test.csv"), params)
    seed = int(params.get("seed", 42))
    summary = data.summary
    horizons = sorted(float(h) for h in (params.get("horizons") or summary["suggested_horizons"]))
    models = [BY_KEY[key] for key in params["models"]]
    t_tr, e_tr, t_te, e_te = data.time_train, data.event_train, data.time_test, data.event_test
    print(f"Train.csv: {len(t_tr)} patients, {int(e_tr.sum())} events ({100 * (1 - e_tr.mean()):.0f}% censored) · "
          f"Test.csv: {len(t_te)} patients, {int(e_te.sum())} events · {len(data.numeric)} numeric and "
          f"{len(data.categorical)} categorical features")
    print(f"Horizons: {', '.join(f'{h:g}' for h in horizons)} · Device: {describe_device()}")
    for folder in ("Models", "Survival_Plots", "Metrics_Plots", "Predictions", "Explainability"):
        os.makedirs(os.path.join(MATERIALS, folder), exist_ok=True)

    # ---- Stratified K-fold cross-validation ---------------------------------------------------
    from sklearn.model_selection import StratifiedKFold
    k = int(params.get("k_folds", 5))
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(t_tr)), e_tr))
    write_splits(index_rows(folds, data.ids_train, {"row": list(range(1, len(t_tr) + 1)), "event": list(e_tr)}),
                 materials=MATERIALS, kind="kfold",
                 description=f"{k}-fold cross-validation stratified on Event (seed {seed}); the preprocessing and every "
                             "model are fitted on the train samples of a fold and scored on its validation samples. "
                             "id: the ID of Train.csv (else the row); row: the line of Train.csv.")
    _banner(f"Training on K-Fold cross validation ({k} folds)")
    columns = sm.METRICS + sm.horizon_metrics(horizons)
    kfold, skipped = {}, []
    for model in models:
        print(f"{model.name} is starting")
        try:
            t0 = clock.time()
            results = []
            for train_idx, valid_idx in folds:
                featurizer, fitted = _fit(model.key, params, data.F_train.iloc[train_idx], t_tr[train_idx], e_tr[train_idx],
                                          data.numeric, data.categorical)
                X_valid = np.asarray(featurizer.transform(data.F_train.iloc[valid_idx]), float)
                censoring = sm.Censoring(t_tr[train_idx], e_tr[train_idx])
                grid = sm.evaluation_grid(t_tr[train_idx], e_tr[train_idx], t_tr[valid_idx])
                results.append(_evaluate(fitted, X_valid, t_tr[valid_idx], e_tr[valid_idx], censoring, grid, horizons))
            kfold[model.name] = results
            print(f"{model.name}: C-index {np.nanmean([r['C-index'] for r in results]):.3f}, "
                  f"IBS {np.nanmean([r['IBS'] for r in results]):.3f} · {clock.time() - t0:.0f} s")
            print(f"{model.name} is completed successfully")
        except Exception as e:
            print(f"{model.name} failed and was skipped: {_short(e)}")
            skipped.append({"model": model.name, "reason": _short(e)})
    if not kfold:
        return "Error: every model failed during the cross-validation (see the log)."
    pd.DataFrame({name: _mean_sd(results, columns) for name, results in kfold.items()}).T[columns] \
        .to_excel(os.path.join(MATERIALS, f"{k}_fold_results.xlsx"))
    pd.DataFrame([dict(model=name, fold=i + 1, **r) for name, results in kfold.items() for i, r in enumerate(results)]) \
        .to_csv(os.path.join(MATERIALS, "Metrics_Plots", "validation_folds.csv"), index=False)
    _banner("Training on K-Fold cross validation completed successfully")

    # ---- Final models on Test.csv ---------------------------------------------------------------
    _banner("Evaluating algorithms on Test.csv")
    censoring = sm.Censoring(t_tr, e_tr)
    grid = sm.evaluation_grid(t_tr, e_tr, t_te)
    curve_grid = sm.evaluation_grid(t_tr, e_tr, t_te, points=25)
    km_model = _KaplanMeier().fit(None, t_tr, e_tr)
    X_dummy = np.zeros((len(t_te), 1))
    test = {KM_NAME: _evaluate(km_model, X_dummy, t_te, e_te, censoring, grid, horizons)}
    over_auc, over_brier = {}, {}
    km_brier = [sm.brier(t, km_model.predict_survival(X_dummy, [t])[:, 0], t_te, e_te, censoring) for t in curve_grid]
    predictions = pd.DataFrame({data.test_id_column or "row": data.ids_test, TIME: t_te, EVENT: e_te})
    km_test = sm.kaplan_meier(t_te, e_te)
    middle = horizons[len(horizons) // 2]
    info_models, notes = {}, []
    trained = [m for m in models if m.name in kfold]
    for model in trained:
        print(f"{model.name} is starting")
        try:
            safe = _safe(model.name)
            featurizer, fitted = _fit(model.key, params, data.F_train, t_tr, e_tr, data.numeric, data.categorical)
            X_test = np.asarray(featurizer.transform(data.F_test), float)
            X_train = np.asarray(featurizer.transform(data.F_train), float)
            test[model.name] = _evaluate(fitted, X_test, t_te, e_te, censoring, grid, horizons)
            risk = fitted.predict_risk(X_test)
            surv_h = fitted.predict_survival(X_test, horizons)
            predictions[f"{model.name} risk"] = np.round(risk, 5)
            for j, h in enumerate(horizons):
                predictions[f"{model.name} S({h:g})"] = np.round(surv_h[:, j], 4)
            over_auc[model.name] = [sm.td_auc(t, risk, t_te, e_te, censoring) for t in curve_grid]
            surv_curve = fitted.predict_survival(X_test, curve_grid)
            over_brier[model.name] = [sm.brier(t, surv_curve[:, j], t_te, e_te, censoring) for j, t in enumerate(curve_grid)]
            # Risk groups: tertiles of the training risk applied to the test patients
            cuts = np.quantile(fitted.predict_risk(X_train), [1 / 3, 2 / 3])
            groups = np.array(["Low risk", "Intermediate risk", "High risk"])[np.searchsorted(cuts, risk, side="right")]
            chi2, dof, p_value = sm.logrank(t_te, e_te, groups)
            plots.km_groups(t_te, e_te, groups, f"{model.name}: test patients by risk group (training tertiles)",
                            os.path.join(MATERIALS, "Survival_Plots", f"{safe}_risk_groups.png"), p_value)
            calibration_points = sm.calibration(middle, fitted.predict_survival(X_test, [middle])[:, 0], t_te, e_te)
            plots.calibration(calibration_points, middle, f"{model.name}: calibration at {middle:g} (Test.csv)",
                              os.path.join(MATERIALS, "Metrics_Plots", f"{safe}_calibration.png"))
            examples = np.argsort(risk)[np.linspace(0, len(risk) - 1, 5).round().astype(int)]
            plot_grid = np.linspace(0, curve_grid[-1], 60)
            curves = fitted.predict_survival(X_test[examples], np.maximum(plot_grid, 1e-6))
            plots.patient_curves(plot_grid, curves, [f"Patient {data.ids_test[i]} (risk rank {r + 1}/5)" for r, i in enumerate(examples)],
                                 km_test, f"{model.name}: predicted survival of five test patients",
                                 os.path.join(MATERIALS, "Survival_Plots", f"{safe}_patients.png"))
            if params.get("explain", True):
                try:
                    importance = permutation_importance(fitted, featurizer, data.F_test, t_te, e_te,
                                                        data.numeric + data.categorical, seed)
                    importance.to_csv(os.path.join(MATERIALS, "Explainability", f"{safe}_permutation_importance.csv"),
                                      index_label="feature")
                    plots.importance(importance, f"{model.name}: permutation importance (Test.csv)",
                                     os.path.join(MATERIALS, "Explainability", f"{safe}_importance.png"))
                except Exception as e:
                    notes.append(f"{model.name}: the permutation importance could not be computed ({_short(e)})")
                    print(notes[-1])
            if model.key in ("coxph", "weibull_aft", "lognormal_aft"):
                names = [str(n) for n in featurizer.get_feature_names_out()]
                coef = pd.DataFrame({"feature": names, "coefficient": fitted.coef_})
                coef["hazard_ratio" if model.key == "coxph" else "time_ratio"] = np.exp(coef["coefficient"])
                coef.to_csv(os.path.join(MATERIALS, "Explainability", f"{safe}_coefficients.csv"), index=False)
            save(SimplatabSurvival(model.name, model.key, data.numeric, data.categorical, featurizer, fitted, horizons),
                 os.path.join(MATERIALS, "Models", f"{safe}.pkl"))
            info_models[model.name] = {"key": model.key, "file": f"Models/{safe}.pkl", "requirements": requirements(model.key),
                                       "logrank_p": p_value, "risk_cuts": [float(c) for c in cuts]}
            print(f"{model.name} is completed successfully")
        except Exception as e:
            traceback.print_exc()
            print(f"{model.name} failed and was skipped: {_short(e)}")
            skipped.append({"model": model.name, "reason": _short(e)})
    done = [m.name for m in trained if m.name in info_models]
    if not done:
        return "Error: every model failed on the test set (see the log)."

    names = done + [KM_NAME]
    table = pd.DataFrame({n: test[n] for n in names}).T[columns]
    table.to_excel(os.path.join(MATERIALS, "test_results.xlsx"))
    predictions.to_csv(os.path.join(MATERIALS, "Predictions", "test_predictions.csv"), index=False)
    metric = params.get("selection_metric", "C-index")
    validation_means = {n: float(np.nanmean([r[metric] for r in kfold[n]])) for n in done}
    best = (min if metric in sm.LOWER_IS_BETTER else max)(done, key=lambda n: validation_means[n])
    bars = ["C-index", "Uno C-index", "IBS"] + [f"AUC@{h:g}" for h in horizons]
    plots.metric_bars(table, bars, best, sm.LOWER_IS_BETTER | {f"Brier@{h:g}" for h in horizons}, "Test.csv",
                      os.path.join(MATERIALS, "Metrics_Plots", "test_metrics.png"))
    plots.over_time(curve_grid, over_auc, "Time-dependent AUC", "Time-dependent AUC on Test.csv (higher is better)",
                    os.path.join(MATERIALS, "Metrics_Plots", "auc_over_time.png"))
    plots.over_time(curve_grid, over_brier, "Brier score", "Brier score on Test.csv (lower is better)",
                    os.path.join(MATERIALS, "Metrics_Plots", "brier_over_time.png"), reference=(KM_NAME, km_brier))

    info = {
        "automator": "survival-analysis", "train_patients": int(len(t_tr)), "train_events": int(e_tr.sum()),
        "test_patients": int(len(t_te)), "test_events": int(e_te.sum()), "horizons": horizons, "k_folds": k,
        "validation_file": f"{k}_fold_results.xlsx", "id_column": data.id_column, "numeric": data.numeric,
        "categorical": data.categorical, "dropped": [{"column": c, "reason": r} for c, r in data.dropped],
        "selection_metric": metric, "best_model": best, "validation_means": validation_means, "models": info_models,
        "calibration_horizon": middle, "ibs_range": [float(grid[0]), float(grid[-1])], "device": describe_device(),
        "skipped": skipped, "notes": notes, "minutes": round((clock.time() - started) / 60, 1),
    }
    with open(os.path.join(MATERIALS, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"Best model (validation {metric}): {best}")
    print("Pipeline completed successfully.")
    return "Pipeline completed successfully"
