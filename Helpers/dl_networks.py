"""PyTorch networks for the deep-learning classifiers trained from scratch.

The scikit-learn compatible wrappers live in ``Helpers/dl_classifiers.py``; this
module is only imported when one of them is fitted, so the rest of the tool keeps
working even if PyTorch is not installed.

* ``TabTransformerNet`` - Huang et al., "TabTransformer: Tabular Data Modeling
  Using Contextual Embeddings" (2020).
* ``TabRNet`` - Gorishniy et al., "TabR: Tabular Deep Learning Meets Nearest
  Neighbors" (ICLR 2024).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block with separate attention / feed-forward dropout."""

    def __init__(self, dim, heads, attn_dropout, ff_dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4 * 2),
            _GEGLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ff_norm(x))


class TabTransformerNet(nn.Module):
    """TabTransformer.

    Categorical columns are embedded (one embedding table per column, index 0 of each
    column is reserved for unseen categories) and contextualised by a stack of
    transformer blocks. The flattened contextual embeddings are concatenated with the
    normalised continuous columns and fed to an MLP. The continuous columns are
    batch-normalised (as in the PyTorch Tabular implementation): the paper's per-row
    layer normalisation discards information when there are few continuous columns.

    Args:
        n_classes (int): number of output classes.
        cat_cardinalities (list[int]): number of known categories of each categorical column.
        n_continuous (int): number of continuous columns.
    """

    def __init__(self, n_classes, cat_cardinalities, n_continuous, dim=32, depth=6, heads=8,
                 attn_dropout=0.1, ff_dropout=0.1, mlp_hidden_mults=(4, 2), mlp_dropout=0.0):
        super().__init__()
        self.n_categorical = len(cat_cardinalities)
        self.n_continuous = n_continuous

        if self.n_categorical > 0:
            table_sizes = [c + 1 for c in cat_cardinalities]  # +1: unseen category
            offsets = torch.tensor([0] + table_sizes[:-1]).cumsum(0)
            self.register_buffer("cat_offsets", offsets, persistent=True)
            self.cat_embedding = nn.Embedding(sum(table_sizes), dim)
            self.transformer = nn.ModuleList(
                [_TransformerBlock(dim, heads, attn_dropout, ff_dropout) for _ in range(depth)]
            )
        self.cont_norm = nn.BatchNorm1d(n_continuous) if n_continuous > 0 else None

        d_in = self.n_categorical * dim + n_continuous
        layers, d = [], d_in
        for mult in mlp_hidden_mults:
            layers += [nn.Linear(d, d_in * mult), nn.ReLU(), nn.Dropout(mlp_dropout)]
            d = d_in * mult
        layers.append(nn.Linear(d, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont):
        parts = []
        if self.n_categorical > 0:
            tokens = self.cat_embedding(x_cat + self.cat_offsets)
            for block in self.transformer:
                tokens = block(tokens)
            parts.append(tokens.flatten(1))
        if self.cont_norm is not None:
            parts.append(self.cont_norm(x_cont))
        return self.mlp(torch.cat(parts, dim=1))


class PeriodicEmbeddings(nn.Module):
    """PLR ("lite") numerical embeddings: Periodic -> Linear -> ReLU.

    Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning" (2022).
    """

    def __init__(self, n_features, n_frequencies=48, frequency_scale=0.01, d_embedding=16):
        super().__init__()
        self.frequencies = nn.Parameter(torch.normal(0.0, frequency_scale, (n_features, n_frequencies)))
        self.linear = nn.Linear(2 * n_frequencies, d_embedding)

    def forward(self, x):
        x = 2 * math.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return F.relu(self.linear(x)).flatten(1)


class TabRNet(nn.Module):
    """TabR: a feed-forward network with a retrieval (nearest-neighbours attention) module.

    For each object, the ``context_size`` nearest training objects are retrieved in a
    learned key space, and their labels plus the key differences are aggregated with a
    softmax over the (negative squared L2) similarities.

    Args:
        n_features (int): number of input columns.
        n_classes (int): number of output classes.
        continuous_idx (list[int]): columns that go through the numerical embeddings
            (only used when ``num_embeddings == "plr"``).
    """

    def __init__(self, n_features, n_classes, continuous_idx, d_main=265, d_multiplier=2.0,
                 encoder_n_blocks=0, predictor_n_blocks=1, context_dropout=0.39, dropout0=0.39,
                 dropout1=0.0, num_embeddings=None, plr_n_frequencies=48, plr_frequency_scale=0.01,
                 plr_d_embedding=16):
        super().__init__()
        continuous_idx = list(continuous_idx) if num_embeddings == "plr" else []
        other_idx = [i for i in range(n_features) if i not in set(continuous_idx)]
        self.register_buffer("continuous_idx", torch.tensor(continuous_idx, dtype=torch.long))
        self.register_buffer("other_idx", torch.tensor(other_idx, dtype=torch.long))
        if continuous_idx:
            self.num_embeddings = PeriodicEmbeddings(
                len(continuous_idx), plr_n_frequencies, plr_frequency_scale, plr_d_embedding
            )
            d_in = len(other_idx) + len(continuous_idx) * plr_d_embedding
        else:
            self.num_embeddings = None
            d_in = n_features

        d_block = int(d_main * d_multiplier)

        def make_block(prenorm):
            return nn.Sequential(
                *([nn.LayerNorm(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                nn.ReLU(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        # Encoder
        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList([make_block(i > 0) for i in range(encoder_n_blocks)])

        # Retrieval module
        self.normalization = nn.LayerNorm(d_main) if encoder_n_blocks > 0 else None
        self.label_encoder = nn.Embedding(n_classes, d_main)
        nn.init.uniform_(self.label_encoder.weight, -1.0, 1.0)
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            nn.ReLU(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.context_dropout = nn.Dropout(context_dropout)

        # Predictor
        self.blocks1 = nn.ModuleList([make_block(True) for _ in range(predictor_n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_main), nn.ReLU(), nn.Linear(d_main, n_classes))

    def encode(self, x):
        """Return the hidden representation and the retrieval key of ``x``."""
        if self.num_embeddings is not None:
            x = torch.cat([x[:, self.other_idx], self.num_embeddings(x[:, self.continuous_idx])], dim=1)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def candidate_keys(self, candidate_x, chunk_size=4096):
        with torch.no_grad():
            return torch.cat([self.encode(c)[1] for c in candidate_x.split(chunk_size)])

    def forward(self, x, candidate_x, candidate_y, context_size, self_positions=None, candidate_k=None):
        """
        Args:
            x: (batch, n_features) objects to score.
            candidate_x / candidate_y: the retrieval pool (training objects and labels).
            self_positions: during training, the position of each object of ``x`` inside
                the pool, so that an object never retrieves itself.
        """
        if candidate_k is None:
            candidate_k = self.candidate_keys(candidate_x)
        h, k = self.encode(x)
        batch_size = k.shape[0]

        with torch.no_grad():
            distances = (
                k.square().sum(1, keepdim=True)
                - 2 * k @ candidate_k.T
                + candidate_k.square().sum(1)[None]
            )
            if self_positions is not None:
                distances[torch.arange(batch_size, device=k.device), self_positions] = torch.inf
            context_idx = distances.topk(context_size, dim=1, largest=False).indices

        if torch.is_grad_enabled():
            # Re-encode the retrieved objects with gradients (the "memory efficient" TabR
            # training: identical gradients without back-propagating through the whole pool).
            context_k = self.encode(candidate_x[context_idx].flatten(0, 1))[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        similarities = -(k[:, None] - context_k).square().sum(-1)
        probs = self.context_dropout(F.softmax(similarities, dim=-1))
        values = self.label_encoder(candidate_y[context_idx]) + self.T(k[:, None] - context_k)
        h = h + (probs[..., None] * values).sum(1)

        for block in self.blocks1:
            h = h + block(h)
        return self.head(h)
