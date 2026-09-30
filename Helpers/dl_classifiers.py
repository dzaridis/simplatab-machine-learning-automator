"""Deep-learning classifiers for tabular data, exposed as scikit-learn estimators.

Every class here follows the scikit-learn estimator API (``get_params``/``set_params``,
``fit``, ``predict_proba``, ``predict``, ``classes_``), so the models plug into the
existing flow unchanged: ``MLPipeline`` (feature selection -> preprocessing -> model),
``GridSearchCV``/``RandomizedSearchCV``, the K-fold threshold optimisation, the
external test, the ROC/PR curves, SHAP and the pickled pipelines.

* ``TabPFNv2Classifier`` - TabPFN v2 prior-fitted transformer (Hollmann et al.,
  Nature 2025) via the ``tabpfn`` package. Pretrained, in-context learning.
* ``TabICLClassifier`` - TabICL tabular foundation model (Qu et al., ICML 2025)
  via the ``tabicl`` package. Pretrained, in-context learning.
* ``TabTransformerClassifier`` - TabTransformer (Huang et al., 2020), trained from scratch.
* ``TabRClassifier`` - TabR retrieval-augmented network (Gorishniy et al., ICLR 2024),
  trained from scratch.

PyTorch and the model packages are imported lazily, only when a model is fitted.
The pretrained checkpoints of TabPFN v2 and TabICL are downloaded from the
HuggingFace Hub on first use (see ``download_pretrained_weights``).
"""
import contextlib
import copy
import importlib
import os

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


def _import(module, attribute, package):
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"This classifier requires the '{package}' package. "
            f"Install the deep learning dependencies with: pip install -r requirements.txt"
        ) from e
    return getattr(mod, attribute)


def _as_dense_float(X, allow_nan=False):
    """The preprocessing ColumnTransformer may return a sparse matrix (one-hot encoding)."""
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array, got an array of shape {X.shape}.")
    if not allow_nan and not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinity.")
    return X


def _torch_device(device):
    import torch
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _discrete_columns(X, max_cardinality):
    """Indices and sorted unique values of the columns with at most ``max_cardinality`` values
    (e.g. the one-hot encoded columns produced by the preprocessing step)."""
    columns, values = [], []
    for j in range(X.shape[1]):
        uniques = np.unique(X[:, j])
        if len(uniques) <= max_cardinality:
            columns.append(j)
            values.append(uniques)
    return columns, values


@contextlib.contextmanager
def _env(name, value):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


# ---------------------------------------------------------------------------
# Pretrained tabular foundation models (in-context learning, no gradient training)
# ---------------------------------------------------------------------------

class TabPFNv2Classifier(ClassifierMixin, BaseEstimator):
    """TabPFN v2 (``tabpfn.TabPFNClassifier``).

    The model's own limits apply (10,000 training samples, 500 features, 10 classes);
    only TabPFN's guard against running more than 1,000 samples on CPU is lifted, as the
    tool usually runs on CPU (it is slower, not less accurate).

    Args:
        n_estimators: number of ensemble members (forward passes on perturbed inputs).
        softmax_temperature: temperature applied to the predicted logits.
        balance_probabilities: rebalance the probabilities by the training class frequencies.
        average_before_softmax: average the ensemble logits instead of the probabilities.
        device: "auto" (CUDA if available), "cpu" or "cuda".
        random_state: seed of the ensemble perturbations.
        model_path: "auto" (download/cache the official checkpoint) or a checkpoint path.
    """

    def __init__(self, n_estimators=4, softmax_temperature=0.9, balance_probabilities=False,
                 average_before_softmax=False, device="auto", random_state=0, model_path="auto"):
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.device = device
        self.random_state = random_state
        self.model_path = model_path

    def fit(self, X, y):
        TabPFNClassifier = _import("tabpfn", "TabPFNClassifier", "tabpfn")
        X = _as_dense_float(X, allow_nan=True)
        self.model_ = TabPFNClassifier(
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            balance_probabilities=self.balance_probabilities,
            average_before_softmax=self.average_before_softmax,
            device=self.device,
            random_state=self.random_state,
            model_path=self.model_path,
        )
        with _env("TABPFN_ALLOW_CPU_LARGE_DATASET", "1"):
            self.model_.fit(X, np.asarray(y))
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(_as_dense_float(X, allow_nan=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TabICLClassifier(ClassifierMixin, BaseEstimator):
    """TabICL (``tabicl.TabICLClassifier``).

    The fitted model is pickled together with its weights, so the saved pipelines can be
    loaded without access to the checkpoint.

    Args:
        n_estimators: number of ensemble members (feature / class shuffles and normalisations).
        softmax_temperature: temperature applied to the predicted logits.
        average_logits: average the ensemble logits instead of the probabilities.
        checkpoint_version: pretrained checkpoint to use (see the ``tabicl`` package).
        model_path: local checkpoint path; ``None`` downloads/caches it from the HuggingFace Hub.
        device: "auto" (CUDA if available), "cpu" or "cuda".
        random_state: seed of the ensemble.
    """

    def __init__(self, n_estimators=8, softmax_temperature=0.9, average_logits=True,
                 checkpoint_version="tabicl-classifier-v2-20260212.ckpt", model_path=None,
                 device="auto", random_state=42):
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.checkpoint_version = checkpoint_version
        self.model_path = model_path
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        TabICL = _import("tabicl", "TabICLClassifier", "tabicl")
        X = _as_dense_float(X, allow_nan=True)
        self.model_ = TabICL(
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            average_logits=self.average_logits,
            checkpoint_version=self.checkpoint_version,
            model_path=self.model_path,
            device=None if self.device == "auto" else self.device,
            random_state=self.random_state,
        )
        self.model_.fit(X, np.asarray(y))
        self._keep_weights_when_pickled()
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def _keep_weights_when_pickled(self):
        # tabicl drops the weights when pickled and reloads the checkpoint on unpickling.
        self.model_._save_model_weights = True

    def __setstate__(self, state):
        super().__setstate__(state)
        if hasattr(self, "model_"):
            self._keep_weights_when_pickled()

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(_as_dense_float(X, allow_nan=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Deep learning models trained from scratch
# ---------------------------------------------------------------------------

class _TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    """Training loop shared by the PyTorch models trained from scratch.

    AdamW + cross-entropy, with early stopping on the validation loss of a stratified
    ``validation_fraction`` split of the training data; the best epoch is restored.

    Subclasses implement ``_setup`` (learn the column layout from the training rows),
    ``_build_network``, ``_to_tensors`` and ``_logits``.
    """

    _PREDICT_CHUNK = 2048

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F

        X = _as_dense_float(X)
        if X.shape[1] == 0:
            raise ValueError("No input features: check the feature selection and preprocessing steps.")
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) < 2:
            raise ValueError("The training data must contain at least two classes.")
        self.n_features_in_ = X.shape[1]
        device = _torch_device(self.device)
        train_idx, val_idx = self._validation_split(y)

        seed = np.random.RandomState(self.random_state).randint(np.iinfo(np.int32).max)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)

            self._setup(X[train_idx])
            network = self._build_network(len(self.classes_)).to(device)
            tensors = self._to_tensors(X, device)
            y_t = torch.as_tensor(y, dtype=torch.long, device=device)
            train_t = torch.as_tensor(train_idx, device=device)
            val_t = torch.as_tensor(val_idx, device=device)
            pool = self._pool(tensors, y_t, train_t)

            optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
            best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
            for epoch in range(self.max_epochs):
                network.train()
                positions = torch.randperm(len(train_idx), generator=generator).to(device)
                batches = list(positions.split(self.batch_size))
                if len(batches) > 1 and len(batches[-1]) == 1:
                    batches.pop()  # batch normalisation needs more than one row per batch
                for batch in batches:
                    rows = train_t[batch]
                    logits = self._logits(network, [t[rows] for t in tensors], pool, self_positions=batch)
                    loss = F.cross_entropy(logits, y_t[rows])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if len(val_idx) == 0:
                    best_epoch = epoch + 1
                    continue
                network.eval()
                with torch.no_grad():
                    val_loss = F.cross_entropy(
                        self._batched_logits(network, [t[val_t] for t in tensors], pool), y_t[val_t]
                    ).item()
                if val_loss < best_loss:
                    best_loss, best_epoch, stale = val_loss, epoch + 1, 0
                    best_state = copy.deepcopy(network.state_dict())
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
            if best_state is not None:
                network.load_state_dict(best_state)

        network.eval()
        self.network_ = network.cpu()
        self.best_epoch_ = best_epoch
        self._finalize(X, y)
        return self

    def _validation_split(self, y):
        indices = np.arange(len(y))
        n_classes = len(np.unique(y))
        n_val = int(round(self.validation_fraction * len(y)))
        if n_val < n_classes or len(y) - n_val < n_classes or np.bincount(y).min() < 2:
            return indices, indices[:0]
        return train_test_split(indices, test_size=n_val, stratify=y, random_state=self.random_state)

    def _batched_logits(self, network, tensors, pool):
        import torch
        pool = self._eval_pool(network, pool)
        n = tensors[0].shape[0]
        return torch.cat([
            self._logits(network, [t[i:i + self._PREDICT_CHUNK] for t in tensors], pool)
            for i in range(0, n, self._PREDICT_CHUNK)
        ])

    def predict_proba(self, X):
        import torch
        check_is_fitted(self, "network_")
        X = _as_dense_float(X)
        device = _torch_device(self.device)
        network = self.network_.to(device)
        with torch.no_grad():
            logits = self._batched_logits(network, self._to_tensors(X, device), self._predict_pool(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        self.network_ = network.cpu()
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # Hooks -----------------------------------------------------------------
    def _pool(self, tensors, y, train_rows):
        """Retrieval pool used during training (TabR only)."""
        return None

    def _predict_pool(self, device):
        return None

    def _eval_pool(self, network, pool):
        """Pool used with the network frozen (validation / prediction)."""
        return pool

    def _finalize(self, X, y):
        pass


class TabTransformerClassifier(_TorchTabularClassifier):
    """TabTransformer (Huang et al., 2020).

    Columns with at most ``max_categorical_cardinality`` distinct training values (e.g.
    the one-hot encoded categorical features of the preprocessing step) are treated as
    categorical tokens and contextualised by the transformer; the remaining columns are
    continuous (batch-normalised). As in the original architecture, when a dataset has no
    categorical column the model reduces to an MLP over the continuous features.

    Args:
        dim: embedding size of the categorical tokens.
        depth: number of transformer blocks.
        heads: number of attention heads (must divide ``dim``).
        attn_dropout / ff_dropout: dropout of the attention and feed-forward sublayers.
        mlp_hidden_mults: hidden layer sizes of the MLP, as multiples of its input size.
        mlp_dropout: dropout of the MLP hidden layers.
        max_categorical_cardinality: columns with at most this many values are categorical.
        learning_rate / weight_decay: AdamW parameters.
        batch_size, max_epochs, patience, validation_fraction: training and early stopping.
        device: "auto" (CUDA if available), "cpu" or "cuda".
        random_state: seed of the weight initialisation, batching and validation split.
    """

    def __init__(self, dim=32, depth=6, heads=8, attn_dropout=0.1, ff_dropout=0.1,
                 mlp_hidden_mults=(4, 2), mlp_dropout=0.0, max_categorical_cardinality=10,
                 learning_rate=1e-3, weight_decay=1e-5, batch_size=64, max_epochs=200,
                 patience=16, validation_fraction=0.15, device="auto", random_state=42):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.mlp_hidden_mults = mlp_hidden_mults
        self.mlp_dropout = mlp_dropout
        self.max_categorical_cardinality = max_categorical_cardinality
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state

    def _setup(self, X):
        self.categorical_columns_, self.categories_ = _discrete_columns(X, self.max_categorical_cardinality)
        categorical = set(self.categorical_columns_)
        self.continuous_columns_ = [j for j in range(X.shape[1]) if j not in categorical]

    def _build_network(self, n_classes):
        from Helpers.dl_networks import TabTransformerNet
        return TabTransformerNet(
            n_classes=n_classes,
            cat_cardinalities=[len(v) for v in self.categories_],
            n_continuous=len(self.continuous_columns_),
            dim=self.dim, depth=self.depth, heads=self.heads,
            attn_dropout=self.attn_dropout, ff_dropout=self.ff_dropout,
            mlp_hidden_mults=tuple(self.mlp_hidden_mults), mlp_dropout=self.mlp_dropout,
        )

    def _to_tensors(self, X, device):
        import torch
        codes = np.zeros((X.shape[0], len(self.categorical_columns_)), dtype=np.int64)
        for k, (j, values) in enumerate(zip(self.categorical_columns_, self.categories_)):
            position = np.clip(np.searchsorted(values, X[:, j]), 0, len(values) - 1)
            known = np.isclose(values[position], X[:, j], rtol=1e-5, atol=1e-6)
            codes[:, k] = np.where(known, position + 1, 0)  # 0 = unseen category
        return [torch.as_tensor(codes, device=device),
                torch.as_tensor(X[:, self.continuous_columns_], device=device)]

    def _logits(self, network, batch, pool, self_positions=None):
        return network(batch[0], batch[1])


class TabRClassifier(_TorchTabularClassifier):
    """TabR (Gorishniy et al., ICLR 2024).

    The defaults are close to the paper's TabR-S default configuration. At prediction
    time the retrieval pool is the whole training data, which is therefore stored in the
    fitted model (and in the pickled pipeline).

    Args:
        d_main: width of the representations.
        d_multiplier: hidden size of the residual blocks, as a multiple of ``d_main``.
        encoder_n_blocks / predictor_n_blocks: residual blocks before / after retrieval.
        context_size: number of retrieved neighbours.
        context_dropout: dropout on the neighbour attention weights.
        dropout0 / dropout1: dropout inside / at the output of the residual blocks.
        num_embeddings: None, or "plr" for periodic embeddings of the continuous columns
            (columns with more than two distinct values).
        plr_n_frequencies / plr_frequency_scale / plr_d_embedding: PLR embedding parameters.
        learning_rate / weight_decay: AdamW parameters.
        batch_size, max_epochs, patience, validation_fraction: training and early stopping.
        device: "auto" (CUDA if available), "cpu" or "cuda".
        random_state: seed of the weight initialisation, batching and validation split.
    """

    def __init__(self, d_main=265, d_multiplier=2.0, encoder_n_blocks=0, predictor_n_blocks=1,
                 context_size=96, context_dropout=0.39, dropout0=0.39, dropout1=0.0,
                 num_embeddings=None, plr_n_frequencies=48, plr_frequency_scale=0.01,
                 plr_d_embedding=16, learning_rate=3e-4, weight_decay=1e-6, batch_size=64,
                 max_epochs=200, patience=16, validation_fraction=0.15, device="auto",
                 random_state=42):
        self.d_main = d_main
        self.d_multiplier = d_multiplier
        self.encoder_n_blocks = encoder_n_blocks
        self.predictor_n_blocks = predictor_n_blocks
        self.context_size = context_size
        self.context_dropout = context_dropout
        self.dropout0 = dropout0
        self.dropout1 = dropout1
        self.num_embeddings = num_embeddings
        self.plr_n_frequencies = plr_n_frequencies
        self.plr_frequency_scale = plr_frequency_scale
        self.plr_d_embedding = plr_d_embedding
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state

    def _setup(self, X):
        if self.num_embeddings not in (None, "plr"):
            raise ValueError(f"num_embeddings must be None or 'plr', got {self.num_embeddings!r}.")
        binary, _ = _discrete_columns(X, 2)
        self.continuous_columns_ = [j for j in range(X.shape[1]) if j not in set(binary)]

    def _build_network(self, n_classes):
        from Helpers.dl_networks import TabRNet
        return TabRNet(
            n_features=self.n_features_in_, n_classes=n_classes,
            continuous_idx=self.continuous_columns_, d_main=self.d_main,
            d_multiplier=self.d_multiplier, encoder_n_blocks=self.encoder_n_blocks,
            predictor_n_blocks=self.predictor_n_blocks, context_dropout=self.context_dropout,
            dropout0=self.dropout0, dropout1=self.dropout1, num_embeddings=self.num_embeddings,
            plr_n_frequencies=self.plr_n_frequencies, plr_frequency_scale=self.plr_frequency_scale,
            plr_d_embedding=self.plr_d_embedding,
        )

    # Each object attends to ``context_size`` neighbours: keep prediction chunks small.
    _PREDICT_CHUNK = 256

    def _to_tensors(self, X, device):
        import torch
        return [torch.as_tensor(X, device=device)]

    def _pool(self, tensors, y, train_rows):
        return {"x": tensors[0][train_rows], "y": y[train_rows]}

    def _finalize(self, X, y):
        # All the training data (including the early-stopping split) is retrieved from at prediction.
        self.candidate_x_ = X
        self.candidate_y_ = y

    def _predict_pool(self, device):
        import torch
        return {"x": torch.as_tensor(self.candidate_x_, device=device),
                "y": torch.as_tensor(self.candidate_y_, dtype=torch.long, device=device)}

    def _eval_pool(self, network, pool):
        # The network is frozen: encode the candidates' keys once for all the chunks.
        return {**pool, "k": network.candidate_keys(pool["x"])}

    def _logits(self, network, batch, pool, self_positions=None):
        n_candidates = pool["x"].shape[0] - (1 if self_positions is not None else 0)
        context_size = max(1, min(self.context_size, n_candidates))
        return network(batch[0], pool["x"], pool["y"], context_size,
                       self_positions=self_positions, candidate_k=pool.get("k"))


DEEP_LEARNING_CLASSIFIERS = (
    TabPFNv2Classifier,
    TabICLClassifier,
    TabTransformerClassifier,
    TabRClassifier,
)


def download_pretrained_weights():
    """Download and cache the TabPFN v2 and TabICL checkpoints (e.g. while building the
    Docker image), by fitting both models once on a tiny dataset."""
    rng = np.random.RandomState(0)
    X, y = rng.normal(size=(20, 3)), np.arange(20) % 2
    for model in (TabPFNv2Classifier(n_estimators=1), TabICLClassifier(n_estimators=1)):
        model.fit(X, y)
        print(f"{type(model).__name__}: pretrained weights available")


if __name__ == "__main__":
    download_pretrained_weights()
