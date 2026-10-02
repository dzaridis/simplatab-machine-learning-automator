"""Metrics of a clustering.

Internal (no labels needed; computed on the clustered samples, noise left out):
- Silhouette (-1 to 1, higher is better): how much closer each sample is to its own cluster than
  to the nearest other cluster;
- Calinski-Harabasz (higher is better): between-cluster over within-cluster dispersion;
- Davies-Bouldin (0 or more, lower is better): average similarity of each cluster with its most
  similar one.

External (with the Target classes; noise counts as one more cluster):
- ARI: adjusted Rand index (chance level 0, perfect 1);
- AMI / NMI: (adjusted / normalised) mutual information between clusters and classes;
- Homogeneity (each cluster holds a single class), Completeness (each class falls in a single
  cluster) and their harmonic mean, the V-measure;
- FMI: Fowlkes-Mallows index; Purity: share of samples in the majority class of their cluster;
- Accuracy: share of samples whose cluster maps to their class, clusters and classes matched
  one-to-one by the Hungarian algorithm.

Stability (validation): ARI between the clusters a model fitted on a fold gives to the held-out
samples and the clusters of the model fitted on all of Train.csv.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as skm

INTERNAL = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]
EXTERNAL = ["ARI", "AMI", "NMI", "V-measure", "Homogeneity", "Completeness", "FMI", "Purity", "Accuracy"]
COUNTS = ["Clusters", "Noise %"]
STABILITY = "Stability (ARI)"
LOWER_IS_BETTER = {"Davies-Bouldin"}
NOISE = -1
SILHOUETTE_SAMPLE = 5000


def n_clusters(labels):
    return int(len(set(np.asarray(labels).tolist()) - {NOISE}))


def internal(X, labels, seed=0, sample=SILHOUETTE_SAMPLE):
    labels = np.asarray(labels)
    keep = labels != NOISE
    out = {m: np.nan for m in INTERNAL}
    k = n_clusters(labels)
    if k < 2 or k >= keep.sum():
        return out
    Xk, lk = X[keep], labels[keep]
    size = min(sample, len(lk)) if len(lk) > sample else None
    out["Silhouette"] = float(skm.silhouette_score(Xk, lk, sample_size=size, random_state=seed))
    out["Calinski-Harabasz"] = float(skm.calinski_harabasz_score(Xk, lk))
    out["Davies-Bouldin"] = float(skm.davies_bouldin_score(Xk, lk))
    return out


def contingency(classes, labels):
    """(matrix clusters x classes, cluster values, class values)."""
    clusters = sorted(set(labels.tolist()))
    names = sorted(set(classes.tolist()), key=str)
    index_c = {c: i for i, c in enumerate(clusters)}
    index_y = {y: j for j, y in enumerate(names)}
    table = np.zeros((len(clusters), len(names)), dtype=int)
    for c, y in zip(labels, classes):
        table[index_c[c], index_y[y]] += 1
    return table, clusters, names


def matched_accuracy(classes, labels):
    table, _, _ = contingency(classes, labels)
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / table.sum())


def external(classes, labels):
    """External metrics on the samples that have a class (None: no class)."""
    classes = np.asarray(classes, dtype=object)
    labels = np.asarray(labels)
    known = np.array([c is not None for c in classes])
    out = {m: np.nan for m in EXTERNAL}
    if known.sum() < 2:
        return out
    y, c = classes[known].astype(str), labels[known]
    h, comp, v = skm.homogeneity_completeness_v_measure(y, c)
    table, _, _ = contingency(y, c)
    out.update({
        "ARI": float(skm.adjusted_rand_score(y, c)),
        "AMI": float(skm.adjusted_mutual_info_score(y, c)),
        "NMI": float(skm.normalized_mutual_info_score(y, c)),
        "V-measure": float(v), "Homogeneity": float(h), "Completeness": float(comp),
        "FMI": float(skm.fowlkes_mallows_score(y, c)),
        "Purity": float(table.max(axis=1).sum() / table.sum()),
        "Accuracy": matched_accuracy(y, c),
    })
    return out


def score(X, labels, classes=None, seed=0):
    """Every metric of a clustering: counts, internal and (with classes) external."""
    labels = np.asarray(labels)
    out = {"Clusters": n_clusters(labels), "Noise %": float(100 * np.mean(labels == NOISE))}
    out.update(internal(X, labels, seed))
    if classes is not None:
        out.update(external(classes, labels))
    return out


def stability(reference, labels):
    return float(skm.adjusted_rand_score(np.asarray(reference), np.asarray(labels)))


# ---------------------------------------------------------------------------------------
# Choosing the number of clusters
# ---------------------------------------------------------------------------------------

CRITERIA = {"silhouette": "Silhouette", "calinski_harabasz": "Calinski-Harabasz", "davies_bouldin": "Davies-Bouldin"}


def criterion(X, labels, name="silhouette", seed=0):
    """The value of a criterion for choosing the number of clusters, and whether higher is better."""
    labels = np.asarray(labels)
    keep = labels != NOISE
    if n_clusters(labels) < 2 or n_clusters(labels) >= keep.sum():
        return np.nan
    Xk, lk = X[keep], labels[keep]
    if name == "silhouette":
        size = 3000 if len(lk) > 3000 else None
        return float(skm.silhouette_score(Xk, lk, sample_size=size, random_state=seed))
    if name == "calinski_harabasz":
        return float(skm.calinski_harabasz_score(Xk, lk))
    return float(skm.davies_bouldin_score(Xk, lk))


def better(name, a, b):
    """Whether criterion value a is better than b (NaN never is)."""
    if a is None or np.isnan(a):
        return False
    if b is None or np.isnan(b):
        return True
    return a < b if name == "davies_bouldin" else a > b


def summary_metric(metric, values):
    """Best value of a metric among models (direction aware)."""
    clean = [v for v in values if v == v]
    if not clean:
        return np.nan
    return min(clean) if metric in LOWER_IS_BETTER else max(clean)
