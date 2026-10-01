"""Figures of the detection automator."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, BLUE_LIGHT, GRAY = "#2a78d6", "#9ec5f4", "#c9c8c3"
TP, FP, FN = "#2a78d6", "#e34948", "#eda100"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _box(ax, box, color, text=None, dashed=False):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.6,
                           linestyle="--" if dashed else "-"))
    if text:
        ax.text(x1, y1 - 1, text, color="white", fontsize=6.5, va="bottom",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2, "alpha": 0.9})


def detections_figure(image, detections, matched, missed, classes, title, path):
    """The image with its detections above the threshold (blue: correct, red: false alarm) and
    the missed boxes (yellow, dashed). ``detections`` (boxes, scores, labels) and ``missed``
    (boxes, labels), 2D in the pixels of ``image``."""
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(4.6, 4.6 * height / max(width, 1) + 0.4))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    for box, label in zip(missed[0], missed[1]):
        _box(ax, box, FN, f"missed {classes[label]}", dashed=True)
    for box, score, label, ok in zip(*detections, matched):
        _box(ax, box, TP if ok else FP, f"{classes[label]} {score:.2f}")
    ax.set_axis_off()
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def drise_figure(image, saliency, box, label, score, title, path):
    """D-RISE saliency over the image, with the explained detection."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax in axes:
        ax.imshow(image)
        ax.set_axis_off()
    axes[1].imshow(saliency, cmap="inferno", alpha=0.55, extent=(0, image.shape[1], image.shape[0], 0), vmin=0, vmax=1)
    for ax in axes:
        _box(ax, box, BLUE, f"{label} {score:.2f}")
    axes[0].set_title("Detection", fontsize=8, color=INK, loc="left")
    axes[1].set_title("D-RISE saliency (bright: needed by the detection)", fontsize=8, color=INK, loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curves_grid(curves, best, xlabel, ylabel, title, path, xlog=False, xlim=None):
    """Small multiples: one panel per model (blue), the other models in gray behind."""
    names = list(curves)
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False, sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        _style(ax)
        for other in names:
            x, y = curves[other]
            if other != name and len(x):
                ax.plot(x, y, color=GRAY, linewidth=1)
        x, y = curves[name]
        if len(x):
            ax.plot(x, y, color=BLUE, linewidth=2)
        ax.set_title(name + (" (best)" if name == best else ""), fontsize=8.5, color=INK, loc="left")
        if xlog:
            ax.set_xscale("log")
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.02)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metric_bars(table, metrics, best, path):
    """One panel per metric (higher is better), the best model highlighted."""
    metrics = [m for m in metrics if m in table and table[m].notna().any()]
    cols = min(4, len(metrics))
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, (0.36 * len(table) + 1.2) * rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        values = table[metric].sort_values()
        ax.barh(range(len(values)), values.to_numpy(), color=[BLUE if n == best else BLUE_LIGHT for n in values.index], height=0.7)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(values.index, fontsize=7.5, color=INK)
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1.12)
        for i, value in enumerate(values.to_numpy()):
            if np.isfinite(value):
                ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=7, color=MUTED)
        ax.set_title(metric, fontsize=9, color=INK, loc="left")
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def class_heatmap(values, models, classes, title, path):
    """AP per model (rows) and class (columns)."""
    values = np.asarray(values, float)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(classes) + 3), 0.42 * len(models) + 1.5))
    image = ax.imshow(values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8, color=MUTED, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    for (i, j), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.5 else INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7, colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
