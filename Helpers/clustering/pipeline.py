"""The clustering pipeline, following the other automators step by step:
1. data: Train.csv (and an optional Test.csv) preprocessed as the configuration says (data.py);
2. every selected algorithm clusters Train.csv. Its number of clusters is given, the number of
   Target classes, or chosen in a range by a criterion (silhouette by default); density-based and
   Dirichlet-process algorithms find it themselves;
3. K-fold validation: each algorithm is fitted again on K-1 folds and assigns the held-out fold;
   the held-out clusters are scored (internal metrics, external metrics with a Target) and compared
   with the clusters of the model fitted on all of Train.csv (stability);
4. the models fitted on Train.csv assign the Test.csv samples (scored the same way); cluster
   profiles, 2D projections, clusters vs. classes, silhouettes, SHAP explanations of the clusters
   (through a surrogate random forest) and the saved models.
Outputs go to ./Materials. Progress is printed in the format that web/jobs.py follows.
"""
import json
import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from . import metrics as cm
from . import plots
from .data import ID_COLUMNS, LABEL, DataError, prepare
from .models import BY_KEY, OPTICS_XI, Assigner, SimplatabClusterer, build, fit_labels, order_mapping, raw_predict, remap, \
    requirements, save, candidates
from Helpers.splits import index_rows, write_splits

MATERIALS = "Materials"
PROJECTION_SAMPLE = 3000
SHAP_SAMPLE = 400
SURROGATE_SAMPLE = 3000


def _banner(text):
    print("------------- \n", f"{text} \n", "-------------")


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def _short(error):
    text = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    return text[:300]


def _quiet():
    for category in (UserWarning, FutureWarning, DeprecationWarning, RuntimeWarning):
        warnings.filterwarnings("ignore", category=category)
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass


def describe_device():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return f"CPU ({os.cpu_count()} cores)"


def run_clustering_pipeline(input_folder, params):
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
# Number of clusters, fitting, validation
# ---------------------------------------------------------------------------------------

def k_range(params, n):
    low = max(2, int(params.get("k_min", 2)))
    high = max(low, min(int(params.get("k_max", 10)), n - 1))
    return list(range(low, high + 1))


MAX_NOISE = 0.5   # settings leaving more of the samples as noise are not chosen


def _choose_setting(algorithm, X, params):
    """Algorithms that find the number of clusters themselves: their settings tried on X, the one
    with the best criterion kept (at most half of the samples as noise)."""
    criterion = params.get("k_criterion", "silhouette")
    seed = int(params.get("seed", 42))
    best, first = None, None
    if algorithm.key == "optics":
        from sklearn.cluster import cluster_optics_xi
        estimator = build(algorithm.key, None, X, params).fit(X)
        options = []
        for xi in OPTICS_XI:
            labels, _ = cluster_optics_xi(reachability=estimator.reachability_, predecessor=estimator.predecessor_,
                                          ordering=estimator.ordering_, min_samples=estimator.min_samples,
                                          min_cluster_size=estimator.min_cluster_size, xi=xi)
            options.append((f"xi {xi}", xi, labels))
        for description, xi, labels in options:
            value = cm.criterion(X, labels, criterion, seed) if np.mean(labels == cm.NOISE) <= MAX_NOISE else np.nan
            first = first or (description, xi, labels)
            if best is None or cm.better(criterion, value, best[3]):
                best = (description, xi, labels, value)
        description, xi, labels = (best if best[3] == best[3] else first)[:3]
        estimator.set_params(xi=xi)
        estimator.labels_ = labels
        return estimator, np.asarray(labels, dtype=int), description
    for description, estimator in candidates(algorithm.key, X, params):
        try:
            labels = fit_labels(estimator, X)
        except Exception:
            continue
        value = cm.criterion(X, labels, criterion, seed) if np.mean(labels == cm.NOISE) <= MAX_NOISE else np.nan
        first = first or (description, estimator, labels)
        if best is None or cm.better(criterion, value, best[3]):
            best = (description, estimator, labels, value)
    if first is None:
        raise ValueError("no setting could be fitted")
    description, estimator, labels = (best if best[3] == best[3] else first)[:3]
    return estimator, labels, description


def fit_algorithm(algorithm, X, params, k_mode, classes):
    """Fits an algorithm on X: (estimator, raw labels, k, curve of the criterion [(k, value)],
    description of the setting)."""
    criterion = params.get("k_criterion", "silhouette")
    seed = int(params.get("seed", 42))
    if not algorithm.uses_k:
        estimator, labels, description = _choose_setting(algorithm, X, params)
        return estimator, labels, None, [], description
    if k_mode == "classes":
        k = len(classes)
    elif k_mode != "auto":
        k = int(k_mode)
    else:
        k = None
    if k is not None:
        k = max(1, min(k, len(X) - 1))
        estimator = build(algorithm.key, k, X, params)
        return estimator, fit_labels(estimator, X), k, [], f"k = {k}"
    ks = k_range(params, len(X))
    curve, best = [], None
    if algorithm.family == "deep" and algorithm.key != "som":
        # One representation for every k: the pretrained network's embedding, clustered by k-means
        from sklearn.cluster import KMeans
        network = build(algorithm.key, ks[0], X, params).pretrain(X)
        z = network.transform(X)
        for k in ks:
            labels = KMeans(k, n_init=10, random_state=seed).fit_predict(z)
            value = cm.criterion(X, labels, criterion, seed)
            curve.append((k, value))
            if best is None or cm.better(criterion, value, best[1]):
                best = (k, value)
        network.n_clusters = best[0]
        return network, fit_labels(network, X), best[0], curve, f"k = {best[0]} ({cm.CRITERIA[criterion]} {best[1]:.3g})"
    for k in ks:
        estimator = build(algorithm.key, k, X, params)
        labels = fit_labels(estimator, X)
        value = cm.criterion(X, labels, criterion, seed)
        curve.append((k, value))
        if best is None or cm.better(criterion, value, best[1]):
            best = (k, value, estimator, labels)
    if best is None or best[1] != best[1]:
        raise ValueError(f"no number of clusters between {ks[0]} and {ks[-1]} gave a valid {criterion}")
    return best[2], best[3], best[0], curve, f"k = {best[0]} ({cm.CRITERIA[criterion]} {best[1]:.3g})"


def make_folds(n, y, k, seed):
    from sklearn.model_selection import KFold, StratifiedKFold
    if y is not None:
        known = np.array([v if v is not None else "__none__" for v in y])
        counts = pd.Series(known).value_counts()
        if counts.min() >= k:
            return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n), known))
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n)))


def validate(algorithm, X, y, labels, k, folds, params, fitted=None):
    """Scores of each fold: the model fitted on the training part (same number of clusters, or same
    setting) assigns the held-out part."""
    from sklearn.base import clone
    results = []
    for train_idx, valid_idx in folds:
        if algorithm.uses_k or fitted is None:
            estimator = build(algorithm.key, k, X[train_idx], params)
        else:
            estimator = clone(fitted)
        raw = fit_labels(estimator, X[train_idx])
        assigner = None if algorithm.native_predict else Assigner(X[train_idx], raw)
        predicted = raw_predict(algorithm, estimator, assigner, X[valid_idx])
        scores = cm.score(X[valid_idx], predicted, y[valid_idx] if y is not None else None, int(params.get("seed", 42)))
        scores[cm.STABILITY] = cm.stability(labels[valid_idx], predicted)
        results.append(scores)
    return results


def _mean_sd(results, columns):
    table = pd.DataFrame(results)
    out = {}
    for m in columns:
        values = table[m].astype(float) if m in table else pd.Series(dtype=float)
        digits = 1 if m in ("Clusters", "Noise %", "Calinski-Harabasz") else 3
        out[m] = f"{values.mean():.{digits}f} ± {values.std(ddof=0):.{digits}f}" if values.notna().any() else "–"
    return out


# ---------------------------------------------------------------------------------------
# Description of the clusters
# ---------------------------------------------------------------------------------------

def projections(X, seed, use_tsne):
    """2D projections of a sample of the training data: {"PCA": P, "t-SNE": P or None}, indices."""
    rng = np.random.default_rng(seed)
    n = len(X)
    index = np.sort(rng.choice(n, min(n, PROJECTION_SAMPLE), replace=False))
    if X.shape[1] >= 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=seed).fit(X)
        P_pca = pca.transform(X[index])
        label = f"PCA ({100 * pca.explained_variance_ratio_.sum():.0f}% of the variance)"
    else:
        P_pca = np.c_[X[index, 0], rng.normal(0, 0.05, len(index))]
        label = "Feature (with vertical jitter)"
    out = {label: P_pca}
    if use_tsne and len(index) >= 10 and X.shape[1] >= 2:
        from sklearn.manifold import TSNE
        perplexity = float(max(2, min(30, (len(index) - 1) / 3)))
        out["t-SNE"] = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X[index])
    return out, index


def profile(frame, labels, numeric, categorical):
    """(table of every cluster: size, means / most frequent values; z-score matrix of the clusters
    (noise left out) for the heatmap: [clusters x features], clusters, sizes, feature names)."""
    clusters = sorted(set(labels.tolist()) - {-1}) + ([-1] if (labels == -1).any() else [])
    num = frame[numeric].apply(pd.to_numeric, errors="coerce") if numeric else pd.DataFrame(index=frame.index)
    rows = []
    for c in clusters:
        mask = labels == c
        row = {"cluster": "noise" if c == -1 else c, "size": int(mask.sum()), "share %": round(100 * mask.mean(), 1)}
        for column in numeric:
            row[f"{column} (mean)"] = num.loc[mask, column].mean()
            row[f"{column} (median)"] = num.loc[mask, column].median()
        for column in categorical:
            values = frame.loc[mask, column].astype(str)
            if len(values):
                top = values.value_counts()
                row[f"{column} (most frequent)"] = f"{top.index[0]} ({100 * top.iloc[0] / len(values):.0f}%)"
        rows.append(row)
    table = pd.DataFrame(rows)
    # z-scores: numeric means, and the share of the most common levels of the categorical columns
    columns, names = [], []
    for column in numeric:
        values = num[column]
        std = values.std(ddof=0)
        columns.append(((values - values.mean()) / std if std > 0 else values * 0).to_numpy())
        names.append(column)
    for column in categorical:
        values = frame[column].astype(str)
        for level in values.value_counts().index[:5]:
            share = (values == level).astype(float)
            std = share.std(ddof=0)
            if std > 0:
                columns.append(((share - share.mean()) / std).to_numpy())
                names.append(f"{column}={level}")
    kept = [c for c in clusters if c != -1]
    if not columns or not kept:
        return table, None, kept, [], []
    Z = np.column_stack(columns)
    matrix = np.array([np.nanmean(Z[labels == c], axis=0) for c in kept])
    order = np.argsort(-np.nanmax(np.abs(matrix), axis=0))[:25]
    sizes = [int((labels == c).sum()) for c in kept]
    return table, matrix[:, order], kept, sizes, [names[i] for i in order]


def explain(F, labels, names, seed):
    """Mean |SHAP| of each feature for each cluster, from a random forest trained to recognise the
    clusters (noise left out), and how well the forest reproduces them (cross-validated accuracy)."""
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    keep = np.where(labels != -1)[0]
    clusters = sorted(set(labels[keep].tolist()))
    if len(clusters) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    index = rng.choice(keep, min(len(keep), SURROGATE_SAMPLE), replace=False)
    Fs, ls = F[index], labels[index]
    forest = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    fidelity = None
    smallest = pd.Series(ls).value_counts().min()
    if smallest >= 2:
        folds = StratifiedKFold(n_splits=int(min(3, smallest)), shuffle=True, random_state=seed)
        fidelity = float(cross_val_score(forest, Fs, ls, cv=folds).mean())
    forest.fit(Fs, ls)
    sample = Fs[rng.choice(len(Fs), min(len(Fs), SHAP_SAMPLE), replace=False)]
    values = shap.TreeExplainer(forest).shap_values(sample, check_additivity=False)
    if not isinstance(values, list):   # newer shap: (samples, features, classes)
        values = [values[..., i] for i in range(values.shape[-1])]
    importance = pd.DataFrame({f"cluster_{c}": np.abs(v).mean(axis=0) for c, v in zip(forest.classes_, values)},
                              index=names)
    importance.insert(0, "total", importance.sum(axis=1))
    return importance.sort_values("total", ascending=False), fidelity


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------

def _run(input_folder, params):
    started = time.time()
    _banner("Loading Data")
    test_path = os.path.join(input_folder, "Test.csv")
    data = prepare(os.path.join(input_folder, "Train.csv"), test_path if os.path.exists(test_path) else None, params)
    seed = int(params.get("seed", 42))
    algorithms = [BY_KEY[key] for key in params["models"]]
    X, y = data.X_train, data.y_train
    n = len(X)
    k_mode = params.get("n_clusters", "auto")
    if k_mode == "classes" and not data.has_labels:
        k_mode = "auto"
    print(f"Train.csv: {n} samples · {len(data.numeric)} numeric and {len(data.categorical)} categorical features · "
          f"{data.F_train.shape[1]} after encoding" + (f" · PCA: {X.shape[1]} components" if data.reducer is not None else ""))
    if data.has_labels:
        print(f"Target: {len(data.classes)} classes ({', '.join(data.classes[:10])}): used only to evaluate the clusters")
    else:
        print("No Target: unsupervised evaluation (internal metrics)")
    if data.X_test is not None:
        print(f"Test.csv: {len(data.X_test)} samples" + (" with Target" if data.y_test is not None else ""))
    print(f"Number of clusters: {'chosen by ' + params.get('k_criterion', 'silhouette') + ' between ' + str(k_range(params, n)[0]) + ' and ' + str(k_range(params, n)[-1]) if k_mode == 'auto' else ('the number of classes (' + str(len(data.classes)) + ')' if k_mode == 'classes' else k_mode)}")
    print(f"Device: {describe_device()}")
    for folder in ("Models", "Clusters", "Cluster_Profiles", "Embeddings", "Metrics_Plots", "Explainability"):
        os.makedirs(os.path.join(MATERIALS, folder), exist_ok=True)

    validation = params.get("validation", "kfold")
    k_folds = int(params.get("k_folds", 5))
    folds = make_folds(n, y, k_folds, seed) if validation == "kfold" else []
    if folds:
        extra = {"row": list(range(1, n + 1))}
        if y is not None:
            extra["class"] = [v if v is not None else "" for v in y]
        write_splits(index_rows(folds, data.ids_train, extra), materials=MATERIALS, kind="kfold",
                     description=f"{k_folds}-fold validation of the clusterings"
                                 + (" (stratified by Target)" if y is not None else "")
                                 + ": each algorithm is fitted on the train samples of a fold and assigns its validation "
                                   "samples. id: the ID of Train.csv (else the row); row: the line of Train.csv.")
        _banner(f"Training on K-Fold cross validation ({k_folds} folds)")
    else:
        _banner("Clustering Train.csv (no validation)")

    # ---- Clustering of Train.csv and validation -----------------------------------------
    fitted, skipped, curves, chosen = {}, [], {}, {}
    train_scores, kfold = {}, {}
    for algorithm in algorithms:
        name = algorithm.name
        print(f"{name} is starting")
        try:
            if n > algorithm.max_samples:
                raise ValueError(f"at most {algorithm.max_samples:,} samples (Train.csv has {n:,})")
            t0 = time.time()
            estimator, raw, k, curve, setting = fit_algorithm(algorithm, X, params, k_mode, data.classes)
            mapping = order_mapping(raw)
            labels = remap(raw, mapping)
            if cm.n_clusters(labels) < 1:
                raise ValueError("every sample was left as noise")
            if curve:
                curves[name], chosen[name] = curve, k
            assigner = None if algorithm.native_predict else Assigner(X, raw)
            train_scores[name] = cm.score(X, labels, y, seed)
            if folds:
                kfold[name] = validate(algorithm, X, y, labels, k, folds, params, estimator)
            fitted[name] = {"algorithm": algorithm, "estimator": estimator, "raw": raw, "labels": labels,
                            "mapping": mapping, "assigner": assigner, "k": k, "setting": setting}
            s = train_scores[name]
            print(f"{name}: {s['Clusters']} clusters" + (f", {s['Noise %']:.1f}% noise" if s["Noise %"] else "")
                  + (f", silhouette {s['Silhouette']:.3f}" if s["Silhouette"] == s["Silhouette"] else "")
                  + (f", ARI {s['ARI']:.3f}" if "ARI" in s and s["ARI"] == s["ARI"] else "")
                  + (f" · {setting}" if setting else "") + f" · {time.time() - t0:.0f} s")
            print(f"{name} is completed successfully")
        except Exception as e:
            print(f"{name} failed and was skipped: {_short(e)}")
            skipped.append({"model": name, "reason": _short(e)})
    if not fitted:
        return "Error: every algorithm failed (see the log)."
    columns = cm.COUNTS + cm.INTERNAL + (cm.EXTERNAL if y is not None else [])
    pd.DataFrame(train_scores).T[columns].to_excel(os.path.join(MATERIALS, "train_results.xlsx"))
    if kfold:
        kcols = columns + [cm.STABILITY]
        pd.DataFrame({name: _mean_sd(results, kcols) for name, results in kfold.items()}).T[kcols] \
            .to_excel(os.path.join(MATERIALS, f"{k_folds}_fold_results.xlsx"))
        pd.DataFrame([dict(model=name, fold=i + 1, **r) for name, results in kfold.items() for i, r in enumerate(results)]) \
            .to_csv(os.path.join(MATERIALS, "Clusters", "validation_folds.csv"), index=False)
        _banner("Training on K-Fold cross validation completed successfully")
    if curves:
        pd.DataFrame([{"model": name, "k": k, params.get("k_criterion", "silhouette"): v}
                      for name, curve in curves.items() for k, v in curve]) \
            .to_csv(os.path.join(MATERIALS, "Clusters", "k_selection.csv"), index=False)
        plots.k_selection(curves, chosen, cm.CRITERIA[params.get("k_criterion", "silhouette")],
                          os.path.join(MATERIALS, "Metrics_Plots", "k_selection.png"))

    # ---- Test.csv, descriptions, explanations, models ----------------------------------------
    has_test = data.X_test is not None
    _banner("Evaluating algorithms on Test.csv" if has_test else "Evaluating algorithms on Train.csv (cluster descriptions)")
    projected, sample = projections(X, seed, bool(params.get("tsne", True)))
    pd.DataFrame(dict({"id": [data.ids_train[i] for i in sample]},
                      **{f"{label.split(' ')[0]}_{j + 1}": P[:, j] for label, P in projected.items() for j in range(2)})) \
        .to_csv(os.path.join(MATERIALS, "Embeddings", "projection.csv"), index=False)
    if y is not None:
        classes_sample = np.array([v if v is not None else "(none)" for v in y[sample]], dtype=object)
        plots.embedding(projected, classes_sample, "Train.csv samples coloured by Target class",
                        os.path.join(MATERIALS, "Embeddings", "Target_classes.png"), names=lambda v: str(v))
    test_scores, notes, test_labels = {}, [], {}
    explain_models = bool(params.get("explain", True))
    model_files = {}
    for name, fit in fitted.items():
        print(f"{name} is starting")
        try:
            algorithm, labels = fit["algorithm"], fit["labels"]
            safe = _safe(name)
            if has_test:
                predicted = remap(raw_predict(algorithm, fit["estimator"], fit["assigner"], data.X_test), fit["mapping"])
                test_labels[name] = predicted
                test_scores[name] = cm.score(data.X_test, predicted, data.y_test, seed)
            table, matrix, kept, sizes, features = profile(data.train, labels, data.numeric, data.categorical)
            table.to_csv(os.path.join(MATERIALS, "Cluster_Profiles", f"{safe}_profile.csv"), index=False)
            if matrix is not None and len(kept) >= 1:
                plots.profile_heatmap(matrix, kept, sizes, features, f"{name}: cluster profiles (Train.csv)",
                                      os.path.join(MATERIALS, "Cluster_Profiles", f"{safe}_profile.png"))
            plots.embedding(projected, labels[sample], f"{name}: clusters of Train.csv",
                            os.path.join(MATERIALS, "Embeddings", f"{safe}_clusters.png"))
            if y is not None:
                known = np.array([v is not None for v in y])
                table_c, row_names, class_names = cm.contingency(y[known].astype(str), labels[known])
                plots.contingency_heatmap(table_c, row_names, class_names, f"{name}: clusters vs. Target classes (Train.csv)",
                                          os.path.join(MATERIALS, "Metrics_Plots", f"{safe}_contingency.png"))
            if cm.n_clusters(labels) >= 2:
                from sklearn.metrics import silhouette_samples
                keep = np.where(labels != -1)[0]
                rng = np.random.default_rng(seed)
                part = np.sort(rng.choice(keep, min(len(keep), 4000), replace=False))
                if len(set(labels[part].tolist())) >= 2:
                    values = silhouette_samples(X[part], labels[part])
                    plots.silhouette_plot(values, labels[part], float(values.mean()), f"{name}: silhouette of the Train.csv samples",
                                          os.path.join(MATERIALS, "Metrics_Plots", f"{safe}_silhouette.png"))
            if explain_models:
                try:
                    importance, fidelity = explain(data.F_train, labels, data.feature_names, seed)
                    if importance is not None:
                        importance.to_csv(os.path.join(MATERIALS, "Explainability", f"{safe}_feature_importance.csv"),
                                          index_label="feature")
                        plots.shap_bars(importance, sorted(c for c in set(labels.tolist()) if c != -1),
                                        f"{name}: features that define the clusters (SHAP)"
                                        + (f"; surrogate accuracy {100 * fidelity:.0f}%" if fidelity is not None else ""),
                                        os.path.join(MATERIALS, "Explainability", f"{safe}_shap.png"))
                        fit["fidelity"] = fidelity
                except Exception as e:
                    notes.append(f"{name}: SHAP explanations could not be computed ({_short(e)})")
                    print(notes[-1])
            estimator = fit["estimator"]
            if hasattr(estimator, "to_cpu"):
                estimator.to_cpu()
            model = SimplatabClusterer(name, algorithm.key, data.numeric, data.categorical, data.featurizer, data.reducer,
                                       estimator, fit["assigner"], fit["mapping"], algorithm.native_predict,
                                       classes=data.classes or None)
            path = os.path.join(MATERIALS, "Models", f"{safe}.pkl")
            save(model, path)
            model_files[name] = f"Models/{safe}.pkl"
            print(f"{name} is completed successfully")
        except Exception as e:
            traceback.print_exc()
            print(f"{name} failed and was skipped: {_short(e)}")
            skipped.append({"model": name, "reason": _short(e)})
    done = [name for name in fitted if name in model_files]
    if not done:
        return "Error: every algorithm failed while describing the clusters (see the log)."

    # ---- Tables and the best model -----------------------------------------------------------
    ident = data.id_column or "row"
    train_table = pd.DataFrame({ident: data.ids_train})
    if y is not None:
        train_table[LABEL] = data.train[LABEL].to_numpy()
    for name in done:
        train_table[name] = fitted[name]["labels"]
    train_table.to_csv(os.path.join(MATERIALS, "Clusters", "train_clusters.csv"), index=False)
    if has_test:
        test_scores = {name: s for name, s in test_scores.items() if name in done}
        test_columns = cm.COUNTS + cm.INTERNAL + (cm.EXTERNAL if data.y_test is not None else [])
        pd.DataFrame(test_scores).T[test_columns].to_excel(os.path.join(MATERIALS, "test_results.xlsx"))
        test_ident = next((c for c in ID_COLUMNS if c in data.test.columns), None) or "row"
        test_table = pd.DataFrame({test_ident: data.ids_test})
        if data.y_test is not None:
            test_table[LABEL] = data.test[LABEL].to_numpy()
        for name in done:
            test_table[name] = test_labels[name]
        test_table.to_csv(os.path.join(MATERIALS, "Clusters", "test_clusters.csv"), index=False)

    metric = params.get("selection_metric") or ("ARI" if y is not None else "Silhouette")
    if kfold:
        source, scores = "validation", {name: {m: float(np.nanmean([r[m] for r in kfold[name]]))
                                               for m in kfold[name][0]} for name in done if name in kfold}
    elif has_test:
        source, scores = "test", test_scores
    else:
        source, scores = "train", {name: train_scores[name] for name in done}
    eligible = [name for name in done if name in scores and scores[name].get("Noise %", 0) <= 50
                and scores[name].get(metric) == scores[name].get(metric)] or \
        [name for name in done if name in scores and scores[name].get(metric) == scores[name].get(metric)] or done
    pick = min if metric in cm.LOWER_IS_BETTER else max
    best = pick(eligible, key=lambda name: scores.get(name, {}).get(metric, -np.inf) if metric not in cm.LOWER_IS_BETTER
                else scores.get(name, {}).get(metric, np.inf))
    lower = cm.LOWER_IS_BETTER
    bars = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"] + (["ARI", "AMI", "Accuracy"] if y is not None else [])
    plots.metric_bars(pd.DataFrame(train_scores).T.loc[done], bars, best, lower, "Clustering of Train.csv",
                      os.path.join(MATERIALS, "Metrics_Plots", "train_metrics.png"))
    if has_test:
        test_bars = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"] + (["ARI", "AMI", "Accuracy"] if data.y_test is not None else [])
        plots.metric_bars(pd.DataFrame(test_scores).T.loc[done], test_bars, best, lower, "Test.csv samples assigned to the clusters",
                          os.path.join(MATERIALS, "Metrics_Plots", "test_metrics.png"))

    info = {
        "automator": "clustering", "train_samples": n, "test_samples": len(data.X_test) if has_test else 0,
        "id_column": data.id_column, "numeric": data.numeric, "categorical": data.categorical,
        "dropped": [{"column": c, "reason": r} for c, r in data.dropped], "encoded_features": int(data.F_train.shape[1]),
        "model_dimensions": int(X.shape[1]), "scaling": data.scaling, "reduction": "pca" if data.reducer is not None else "none",
        "supervised": bool(data.has_labels), "classes": data.classes, "test_labels": data.y_test is not None,
        "n_clusters": k_mode, "k_range": k_range(params, n) if k_mode == "auto" else None,
        "k_criterion": params.get("k_criterion", "silhouette"), "chosen_k": chosen,
        "clusters": {name: int(train_scores[name]["Clusters"]) for name in done},
        "noise": {name: round(float(train_scores[name]["Noise %"]), 2) for name in done},
        "selection_metric": metric, "selection_source": source, "best_model": best,
        "validation": "kfold" if kfold else "none", "k_folds": k_folds if kfold else None,
        "validation_file": f"{k_folds}_fold_results.xlsx" if kfold else None,
        "models": {name: {"key": fitted[name]["algorithm"].key, "file": model_files[name],
                          "requirements": requirements(fitted[name]["algorithm"].key),
                          "surrogate_accuracy": fitted[name].get("fidelity"),
                          "setting": fitted[name].get("setting")} for name in done},
        "device": describe_device(), "skipped": skipped, "notes": notes, "tsne": "t-SNE" in projected,
        "minutes": round((time.time() - started) / 60, 1),
    }
    with open(os.path.join(MATERIALS, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"Best model ({metric} on {source}): {best}")
    print("Pipeline completed successfully.")
    return "Pipeline completed successfully"
