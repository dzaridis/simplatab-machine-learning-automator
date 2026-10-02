"""Figures of the clustering automator: the clusters in 2D projections of the data, cluster
profiles, clusters vs. classes, silhouettes, the choice of the number of clusters, metrics and
SHAP explanations.

Clusters get the eight categorical colours in a fixed order (cluster 0, the largest, first) and a
marker shape each, so that they stay distinguishable without colour; clusters past the eighth are
grey ("other clusters") and noise is a light grey cross.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
OTHER = "#9a9993"
NOISE_COLOR = "#c9c8c2"
BEST = "#2a78d6"
BAR = "#b9b8b3"
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#1c5cab", "#86b6ef", "#f1f0ec", "#f3a582", "#c2410c"])
BLUES = LinearSegmentedColormap.from_list("blues", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _style(ax, grid=True):
    ax.grid(grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter_groups(ax, P, groups, names=None):
    """One scatter per group: the eight first in colour and shape, the others grey, noise (-1) as crosses."""
    values = [g for g in dict.fromkeys(groups.tolist())]
    ordered = sorted([v for v in values if v != -1], key=lambda v: (-np.sum(groups == v), str(v)))
    size = 9 if len(P) > 2000 else 16
    for i, value in enumerate(ordered):
        mask = groups == value
        label = names(value) if names else f"Cluster {value}"
        if i < len(PALETTE):
            ax.scatter(P[mask, 0], P[mask, 1], s=size, color=PALETTE[i], marker=MARKERS[i], alpha=0.8,
                       edgecolors="white", linewidths=0.3, label=label)
            center = np.median(P[mask], axis=0)
            ax.annotate(str(value) if not names else names(value), center, fontsize=8, fontweight="bold", color=INK,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=PALETTE[i], linewidth=1, alpha=0.9))
        else:
            ax.scatter(P[mask, 0], P[mask, 1], s=size, color=OTHER, marker="o", alpha=0.6, edgecolors="none",
                       label="Other clusters" if i == len(PALETTE) else None)
    if -1 in values:
        mask = groups == -1
        ax.scatter(P[mask, 0], P[mask, 1], s=size, color=NOISE_COLOR, marker="x", linewidths=0.8, label="Noise")


def embedding(projections, groups, title, path, names=None):
    """The samples in each 2D projection ({"PCA": P, "t-SNE": P}), coloured by group."""
    items = [(k, v) for k, v in projections.items() if v is not None]
    fig, axes = plt.subplots(1, len(items), figsize=(5.4 * len(items), 4.8), squeeze=False)
    for ax, (label, P) in zip(axes.flat, items):
        _style(ax, grid=False)
        _scatter_groups(ax, P, np.asarray(groups), names)
        ax.set_title(label, fontsize=9, color=MUTED, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False, fontsize=8, markerscale=1.4)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 0.84, 0.94))
    _save(fig, path)


def profile_heatmap(z, clusters, sizes, features, title, path):
    """Cluster means of the features as z-scores (difference from the overall mean, in standard
    deviations), clipped at +-2.5."""
    z = np.asarray(z, dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.5, 0.55 * len(features) + 3.0), max(3.6, 0.5 * len(clusters) + 2.2)))
    image = ax.imshow(np.clip(z, -2.5, 2.5), cmap=DIVERGING, vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {c} (n={s})" for c, s in zip(clusters, sizes)], fontsize=8, color=INK)
    if z.size <= 400:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if z[i, j] == z[i, j]:
                    ax.text(j, i, f"{z[i, j]:+.1f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(z[i, j]) > 1.6 else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("Cluster mean vs. overall (SD)", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def contingency_heatmap(table, clusters, classes, title, path):
    """Samples of each class in each cluster (row shares as colour, counts as text)."""
    table = np.asarray(table)
    share = table / np.maximum(1, table.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(max(7.5, 0.8 * len(classes) + 3.0), max(3.6, 0.5 * len(clusters) + 2.2)))
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([str(c) for c in classes], rotation=30, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(["Noise" if c == -1 else f"Cluster {c}" for c in clusters], fontsize=8, color=INK)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, str(table[i, j]), ha="center", va="center", fontsize=7.5,
                    color="white" if share[i, j] > 0.55 else INK)
    ax.set_xlabel("Target class", fontsize=8, color=MUTED)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def silhouette_plot(values, labels, average, title, path):
    """Silhouette of every sample, sorted within its cluster (noise left out)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _style(ax)
    y = 0
    clusters = sorted(c for c in set(labels.tolist()) if c != -1)
    for i, cluster in enumerate(clusters):
        part = np.sort(values[labels == cluster])
        color = PALETTE[i] if i < len(PALETTE) else OTHER
        ax.fill_betweenx(np.arange(y, y + len(part)), 0, part, color=color, alpha=0.85, linewidth=0)
        ax.text(-0.04, y + len(part) / 2, str(cluster), fontsize=7.5, color=INK, ha="right", va="center")
        y += len(part) + max(2, len(values) // 100)
    ax.axvline(average, color=INK, linestyle="--", linewidth=1)
    ax.text(average, y, f" mean {average:.2f}", fontsize=8, color=INK, va="bottom")
    ax.set_yticks([])
    ax.set_xlabel("Silhouette", fontsize=8, color=MUTED)
    ax.set_xlim(min(-0.1, float(np.min(values)) - 0.05) if len(values) else -0.1, 1)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def k_selection(curves, chosen, criterion, path):
    """The criterion for each number of clusters, one panel per model, the chosen k marked."""
    names = list(curves)
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 2.8 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        _style(ax)
        ks = [k for k, v in curves[name] if v == v]
        vs = [v for k, v in curves[name] if v == v]
        ax.plot(ks, vs, color=MUTED, linewidth=2, marker="o", markersize=4)
        if name in chosen and chosen[name] in ks:
            v = vs[ks.index(chosen[name])]
            ax.plot([chosen[name]], [v], marker="o", markersize=9, color=BEST, markeredgecolor="white")
            ax.annotate(f"k = {chosen[name]}", (chosen[name], v), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Number of clusters", fontsize=8, color=MUTED)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle(f"Choice of the number of clusters ({criterion}; "
                 f"{'lower' if criterion == 'Davies-Bouldin' else 'higher'} is better)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def metric_bars(table, metrics, best, lower, title, path):
    """One panel per metric: a bar per model, the best model highlighted."""
    metrics = [m for m in metrics if m in table.columns and table[m].notna().any()]
    if not metrics:
        return
    cols = min(3, len(metrics))
    rows = int(np.ceil(len(metrics) / cols))
    height = 0.36 * len(table) + 1.2
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, height * rows), squeeze=False)
    names = list(table.index)
    for ax, metric in zip(axes.flat, metrics):
        _style(ax)
        ax.grid(False, axis="y")
        values = table[metric].astype(float).to_numpy()
        colors = [BEST if n == best else BAR for n in names]
        ax.barh(range(len(names)), np.nan_to_num(values), color=colors, height=0.66)
        for i, v in enumerate(values):
            if v == v:
                ax.text(v, i, f" {v:.3g}", va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_title(f"{metric} ({'lower' if metric in lower else 'higher'} is better)", fontsize=9, color=INK, loc="left")
        ax.margins(x=0.25)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def shap_bars(importance, clusters, title, path, top=15):
    """Mean |SHAP| of the top features, stacked by cluster (the surrogate classifier explains why a
    sample is put in each cluster)."""
    data = importance.head(top)
    fig, ax = plt.subplots(figsize=(9.5, 0.4 * len(data) + 2.0))
    _style(ax)
    ax.grid(False, axis="y")
    left = np.zeros(len(data))
    for i, cluster in enumerate(clusters):
        column = f"cluster_{cluster}"
        if column not in data:
            continue
        values = data[column].to_numpy()
        color = PALETTE[i] if i < len(PALETTE) else OTHER
        ax.barh(range(len(data)), values, left=left, color=color, height=0.66, edgecolor="white", linewidth=1,
                label=f"Cluster {cluster}" if i < len(PALETTE) else ("Other clusters" if i == len(PALETTE) else None))
        left += values
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(list(data.index), fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (stacked over clusters)", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)
