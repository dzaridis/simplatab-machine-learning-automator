"""The clustering algorithms of the automator and the saved model.

Classical (scikit-learn): centroid (K-Means, Bisecting K-Means), probabilistic (Gaussian mixture,
Dirichlet-process Bayesian Gaussian mixture), hierarchical (Ward agglomerative, BIRCH), graph
(spectral clustering, affinity propagation) and density-based (DBSCAN, HDBSCAN, OPTICS, Mean Shift)
algorithms. Deep and neural (PyTorch, MiniSom; Helpers/clustering/deep.py): DEC, IDEC, DCN, VaDE,
SCARF + k-means and a self-organising map.

Algorithms with ``uses_k`` take a number of clusters (given, the number of classes, or chosen on
Train.csv by a criterion); the others find it themselves. A sample outside the training data gets
a cluster from the model itself (``predict``) or, for the algorithms without one (agglomerative,
spectral, DBSCAN, HDBSCAN, OPTICS), from its nearest training samples.
"""
from dataclasses import dataclass

import numpy as np

NOISE = -1


@dataclass(frozen=True)
class Algorithm:
    key: str
    name: str
    family: str          # centroid | probabilistic | hierarchical | graph | density | deep
    description: str
    uses_k: bool          # takes the number of clusters
    default: bool
    native_predict: bool  # assigns new samples itself (else: nearest training samples)
    max_samples: int = 100_000
    slow: bool = False
    noise: bool = False   # can leave samples unclustered (noise, cluster -1)

    @property
    def group(self):
        return "deep_learning" if self.family == "deep" else "classical"


ALGORITHMS = [
    Algorithm("kmeans", "K-Means", "centroid",
              "k-means++ initialisation, 10 restarts: compact, spherical clusters of similar size.", True, True, True),
    Algorithm("bisecting_kmeans", "Bisecting K-Means", "centroid",
              "Splits the largest cluster in two until k clusters: a hierarchy of k-means clusterings.", True, False, True),
    Algorithm("gmm", "Gaussian Mixture", "probabilistic",
              "Mixture of full-covariance Gaussians (EM): elliptical clusters, soft memberships.", True, True, True),
    Algorithm("bayesian_gmm", "Bayesian Gaussian Mixture", "probabilistic",
              "Dirichlet-process mixture: switches off the components it does not need (number of clusters found).",
              False, False, True),
    Algorithm("agglomerative", "Agglomerative (Ward)", "hierarchical",
              "Bottom-up merging that minimises the within-cluster variance.", True, True, False, max_samples=20_000),
    Algorithm("birch", "BIRCH", "hierarchical",
              "Clustering-feature tree summarising the data, then agglomerative clustering: scales to large tables.",
              True, False, True),
    Algorithm("spectral", "Spectral Clustering", "graph",
              "Clusters of the k-nearest-neighbour graph (eigenvectors of its Laplacian): non-convex shapes.",
              True, True, False, max_samples=10_000, slow=True),
    Algorithm("affinity_propagation", "Affinity Propagation", "graph",
              "Samples exchange messages to elect exemplars (number of clusters found).", False, False, True,
              max_samples=5_000, slow=True),
    Algorithm("dbscan", "DBSCAN", "density",
              "Dense regions separated by sparse ones; radius from the k-distance curve, outliers left as noise.",
              False, False, False, max_samples=50_000, noise=True),
    Algorithm("hdbscan", "HDBSCAN", "density",
              "Hierarchical DBSCAN: clusters of varying density, the most stable ones kept, outliers as noise.",
              False, True, False, max_samples=50_000, noise=True),
    Algorithm("optics", "OPTICS", "density",
              "Reachability ordering of the samples; clusters extracted where the density changes steeply.",
              False, False, False, max_samples=20_000, slow=True, noise=True),
    Algorithm("mean_shift", "Mean Shift", "density",
              "Shifts points to the modes of a kernel density estimate (number of clusters found).", False, False, True,
              max_samples=20_000, slow=True),
    Algorithm("dec", "DEC", "deep",
              "Deep Embedded Clustering: pretrained autoencoder, then encoder and centres refined together.",
              True, True, True),
    Algorithm("idec", "IDEC", "deep",
              "Improved DEC: keeps the reconstruction loss while clustering, preserving the local structure.",
              True, True, True),
    Algorithm("dcn", "DCN", "deep",
              "Deep Clustering Network: autoencoder trained jointly with a k-means loss in its latent space.",
              True, False, True),
    Algorithm("vade", "VaDE", "deep",
              "Variational Deep Embedding: a variational autoencoder with a Gaussian-mixture prior, one component per cluster.",
              True, True, True),
    Algorithm("scarf", "SCARF + K-Means", "deep",
              "Contrastive self-supervised embeddings (random feature corruption), clustered with k-means.",
              True, False, True, slow=True),
    Algorithm("som", "Self-Organizing Map", "deep",
              "Kohonen map of prototypes (MiniSom), the prototypes grouped by Ward linkage.", True, False, True),
]
BY_KEY = {a.key: a for a in ALGORITHMS}
FAMILIES = [
    ("centroid", "Centroid-based", "Fast; clusters around centres."),
    ("probabilistic", "Probabilistic (mixtures)", "Soft memberships; elliptical clusters."),
    ("hierarchical", "Hierarchical", "Nested clusters merged bottom-up."),
    ("graph", "Graph-based", "Clusters of a similarity graph; non-convex shapes."),
    ("density", "Density-based", "Arbitrary shapes; outliers left as noise."),
    ("deep", "Deep learning and neural", "Learn a representation and the clusters together (PyTorch, SOM)."),
]


def _knee(values):
    """Index of the knee of an increasing curve (largest gap below the chord)."""
    y = np.sort(np.asarray(values, dtype=float))
    if len(y) < 3 or y[-1] == y[0]:
        return len(y) // 2, y
    x = np.linspace(0, 1, len(y))
    yn = (y - y[0]) / (y[-1] - y[0])
    return int(np.argmax(x - yn)), y


def _neighbour_distances(X, k, seed):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    sample = X[rng.choice(len(X), min(len(X), 5000), replace=False)]
    k = max(2, min(k, len(sample) - 1))
    dist, _ = NearestNeighbors(n_neighbors=k).fit(sample).kneighbors(sample)
    return dist[:, -1]


def density_settings(X):
    """min_samples and the DBSCAN radius from the knee of the k-distance curve."""
    n, d = X.shape
    min_samples = int(min(20, max(4, 2 * d), max(2, n // 10)))
    index, sorted_dist = _knee(_neighbour_distances(X, min_samples, 0))
    eps = float(max(sorted_dist[index], 1e-6))
    return min_samples, eps


def build(key, k, X, params):
    """An unfitted estimator of an algorithm for the data X (k: number of clusters)."""
    from sklearn import cluster, mixture
    seed = int(params.get("seed", 42))
    n, d = X.shape
    if key == "kmeans":
        return cluster.KMeans(n_clusters=k, n_init=10, random_state=seed)
    if key == "bisecting_kmeans":
        return cluster.BisectingKMeans(n_clusters=k, n_init=3, random_state=seed)
    if key == "gmm":
        # reg_covar 1e-2: one-hot columns are constant within many clusters (singular covariances)
        return mixture.GaussianMixture(n_components=k, covariance_type="full", n_init=3, reg_covar=1e-2,
                                       random_state=seed)
    if key == "bayesian_gmm":
        components = max(2, min(int(params.get("k_max", 10)), n - 1))
        return mixture.BayesianGaussianMixture(n_components=components, weight_concentration_prior_type="dirichlet_process",
                                               weight_concentration_prior=1.0 / components, covariance_type="full",
                                               reg_covar=1e-2, max_iter=500, n_init=1, random_state=seed)
    if key == "agglomerative":
        return cluster.AgglomerativeClustering(n_clusters=k, linkage="ward")
    if key == "birch":
        threshold = float(np.median(_neighbour_distances(X, 5, seed)))
        return cluster.Birch(n_clusters=k, threshold=max(threshold, 1e-6))
    if key == "spectral":
        return cluster.SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=min(10, n - 1),
                                          assign_labels="kmeans", random_state=seed)
    if key == "affinity_propagation":
        return cluster.AffinityPropagation(damping=0.9, max_iter=500, random_state=seed)
    if key == "dbscan":
        min_samples, eps = density_settings(X)
        return cluster.DBSCAN(eps=eps, min_samples=min_samples)
    if key == "hdbscan":
        return cluster.HDBSCAN(min_cluster_size=int(max(5, round(0.02 * n))))
    if key == "optics":
        return cluster.OPTICS(min_samples=int(max(5, min(50, round(0.01 * n)))), xi=0.05,
                              min_cluster_size=max(5, int(round(0.02 * n))))
    if key == "mean_shift":
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.2, n_samples=min(n, 1000), random_state=seed)
        return cluster.MeanShift(bandwidth=max(float(bandwidth), 1e-6), bin_seeding=True)
    from . import deep
    if key == "som":
        return deep.SelfOrganizingMap(n_clusters=k, seed=seed)
    networks = {"dec": deep.DEC, "idec": deep.IDEC, "dcn": deep.DCN, "vade": deep.VaDE, "scarf": deep.SCARF}
    return networks[key](n_clusters=k, latent_dim=int(params.get("latent_dim", 10)),
                         pretrain_epochs=int(params.get("pretrain_epochs", 100)), epochs=int(params.get("epochs", 100)),
                         batch_size=int(params.get("batch_size", 256)), learning_rate=float(params.get("learning_rate", 1e-3)),
                         seed=seed)


def candidates(key, X, params):
    """The settings tried for an algorithm that finds the number of clusters itself:
    [(description, unfitted estimator)]; the pipeline keeps the one with the best criterion among
    those leaving at most half of the samples as noise. OPTICS is fitted once and its clusters
    extracted for several xi (see the pipeline)."""
    from sklearn import cluster
    seed = int(params.get("seed", 42))
    n = len(X)
    if key == "dbscan":
        min_samples, knee = density_settings(X)
        distances = np.sort(_neighbour_distances(X, min_samples, seed))
        radii = sorted({round(float(r), 6) for r in [knee] + list(np.quantile(distances, [0.3, 0.45, 0.6, 0.75, 0.85, 0.95]))
                        if r > 0})
        return [(f"eps {r:.3g}, min_samples {min_samples}", cluster.DBSCAN(eps=r, min_samples=min_samples)) for r in radii]
    if key == "hdbscan":
        sizes = sorted({max(5, int(round(f * n))) for f in (0.01, 0.02, 0.05, 0.1)})
        return [(f"min_cluster_size {s}, {method}", cluster.HDBSCAN(min_cluster_size=s, cluster_selection_method=method))
                for s in sizes for method in ("eom", "leaf")]
    if key == "mean_shift":
        out = []
        for q in (0.05, 0.1, 0.2, 0.3):
            bandwidth = float(cluster.estimate_bandwidth(X, quantile=q, n_samples=min(n, 1000), random_state=seed))
            if bandwidth > 0:
                out.append((f"bandwidth {bandwidth:.3g} (quantile {q})", cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)))
        return out or [("default bandwidth", cluster.MeanShift(bin_seeding=True))]
    if key == "affinity_propagation":
        rng = np.random.default_rng(seed)
        sample = X[rng.choice(n, min(n, 1500), replace=False)]
        similarity = -((sample[:, None, :] - sample[None, :, :]) ** 2).sum(2)[np.triu_indices(len(sample), 1)]
        out = []
        for label, value in (("median", np.median(similarity)), ("10th percentile", np.quantile(similarity, 0.1)),
                             ("minimum", similarity.min())):
            out.append((f"preference {label} of the similarities ({value:.3g})",
                        cluster.AffinityPropagation(damping=0.9, max_iter=500, preference=float(value), random_state=seed)))
        return out
    return [("", build(key, None, X, params))]


OPTICS_XI = (0.01, 0.02, 0.05, 0.1)


def fit_labels(estimator, X):
    """Fits an estimator and returns the cluster of every training sample."""
    if hasattr(estimator, "fit_predict"):
        labels = estimator.fit_predict(X)
    else:
        labels = estimator.fit(X).predict(X)
    return np.asarray(labels, dtype=int)


def order_mapping(labels):
    """Clusters renumbered by decreasing size (0 is the largest); noise stays -1."""
    values, counts = np.unique(labels[labels != NOISE], return_counts=True)
    ranked = values[np.lexsort((values, -counts))]
    mapping = {int(v): i for i, v in enumerate(ranked)}
    mapping[NOISE] = NOISE
    return mapping


def remap(labels, mapping):
    return np.array([mapping.get(int(v), NOISE) for v in labels], dtype=int)


class Assigner:
    """Assigns new samples to the clusters of the training samples (weighted vote of the 5 nearest),
    for the algorithms without a predict method; noise is a possible answer."""

    def __init__(self, X, labels, neighbours=5):
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=max(1, min(neighbours, len(X))), weights="distance").fit(X, labels)

    def predict(self, X):
        return self.model.predict(X)


def raw_predict(algorithm, estimator, assigner, X):
    if algorithm.native_predict and hasattr(estimator, "predict"):
        return np.asarray(estimator.predict(X), dtype=int)
    return np.asarray(assigner.predict(X), dtype=int)


class SimplatabClusterer:
    """A trained clustering model, from the columns of Train.csv to a cluster number (0 is the largest
    cluster of Train.csv; -1 is noise): preprocessing (imputation, scaling, one-hot encoding, PCA),
    the algorithm, and the assignment of new samples. Stored with cloudpickle: loading it needs
    numpy, pandas, scikit-learn and cloudpickle (and torch for the deep networks), not Simplatab."""

    def __init__(self, name, key, numeric, categorical, featurizer, reducer, estimator, assigner, mapping,
                 native_predict, classes=None):
        self.name = name
        self.key = key
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.featurizer = featurizer
        self.reducer = reducer
        self.estimator = estimator
        self.assigner = assigner
        self.mapping = dict(mapping)
        self.native_predict = native_predict
        self.n_clusters = len([v for v in self.mapping.values() if v != NOISE])
        self.classes = classes

    @property
    def columns(self):
        return self.numeric + self.categorical

    def transform(self, frame):
        """The samples in the space the algorithm clusters (after preprocessing and PCA)."""
        import pandas as pd
        out = pd.DataFrame(index=frame.index)
        for column in self.numeric:
            out[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        for column in self.categorical:
            out[column] = frame[column].map(lambda v: "missing" if v is None or (isinstance(v, float) and v != v) else str(v))
        X = np.asarray(self.featurizer.transform(out), dtype=float)
        if self.reducer is not None:
            X = self.reducer.transform(X)
        return np.ascontiguousarray(X, dtype=np.float32)

    def predict(self, frame):
        """The cluster of every row (columns of Train.csv; ID and Target are not needed)."""
        X = self.transform(frame)
        if self.native_predict and hasattr(self.estimator, "predict"):
            raw = np.asarray(self.estimator.predict(X), dtype=int)
        else:
            raw = np.asarray(self.assigner.predict(X), dtype=int)
        return np.array([self.mapping.get(int(v), NOISE) for v in raw], dtype=int)

    def predict_proba(self, frame):
        """Cluster memberships (columns: clusters 0, 1, ...), for the probabilistic and deep models."""
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.name} gives hard assignments only (use predict).")
        p = np.asarray(self.estimator.predict_proba(self.transform(frame)))
        out = np.zeros((len(p), self.n_clusters))
        for raw, new in self.mapping.items():
            if new != NOISE and 0 <= raw < p.shape[1]:
                out[:, new] += p[:, raw]
        return out


def save(model, path):
    import cloudpickle
    from . import deep
    import sys
    modules = [deep, sys.modules[__name__]]
    for module in modules:
        cloudpickle.register_pickle_by_value(module)
    try:
        with open(path, "wb") as f:
            cloudpickle.dump(model, f)
    finally:
        for module in modules:
            cloudpickle.unregister_pickle_by_value(module)


REQUIREMENTS = ["numpy==1.23.5", "pandas==2.0.3", "scikit-learn==1.3.1", "cloudpickle==3.1.2"]


def requirements(key):
    return REQUIREMENTS + (["torch==2.8.0"] if key in ("dec", "idec", "dcn", "vade", "scarf") else [])
