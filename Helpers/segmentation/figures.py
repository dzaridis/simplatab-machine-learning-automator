"""Figures of the segmentation automator: metric bars, Dice per case and class, and overlays of
the reference and predicted masks with the uncertainty map."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from Helpers.detection.plots import BLUE, BLUE_LIGHT, INK, MUTED, _style, class_heatmap, metric_bars  # noqa: F401,E402

# Fixed order of the class colours (class 1 first), never cycled within a figure
CLASS_COLOURS = ["#2a78d6", "#e34948", "#eda100", "#1d9a6c", "#8e5bd8", "#e0619c", "#3fb8c4", "#8c6d31"]


def colours(num_classes):
    return [CLASS_COLOURS[(i - 1) % len(CLASS_COLOURS)] for i in range(1, num_classes)]


def distance_bars(table, metrics, best, path, unit):
    """HD95 and ASSD per model (lower is better), the best model highlighted."""
    metrics = [m for m in metrics if m in table and table[m].notna().any()]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.8 * len(metrics), 0.36 * len(table) + 1.3), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        values = table[metric].sort_values(ascending=False)
        ax.barh(range(len(values)), values.to_numpy(), height=0.7,
                color=[BLUE if n == best else BLUE_LIGHT for n in values.index])
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(values.index, fontsize=7.5, color=INK)
        _style(ax)
        ax.grid(axis="y", visible=False)
        top = np.nanmax(values.to_numpy()) if np.isfinite(values.to_numpy()).any() else 1
        ax.set_xlim(0, top * 1.25 or 1)
        for i, value in enumerate(values.to_numpy()):
            if np.isfinite(value):
                ax.text(value + top * 0.02, i, f"{value:.2f}", va="center", fontsize=7, color=MUTED)
        ax.set_title(f"{metric} ({unit}, lower is better)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dice_per_case(case_dice, best, path):
    """Distribution of the per-case Dice of each model on the test set (points and box)."""
    models = sorted(case_dice, key=lambda m: np.nanmedian(case_dice[m]) if np.isfinite(case_dice[m]).any() else -1)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(models) + 1.4))
    rng = np.random.default_rng(0)
    for i, model in enumerate(models):
        values = np.asarray(case_dice[model], float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        colour = BLUE if model == best else "#86b6ef"
        ax.boxplot([values], positions=[i], vert=False, widths=0.55, showfliers=False,
                   medianprops={"color": INK}, boxprops={"color": MUTED}, whiskerprops={"color": MUTED},
                   capprops={"color": MUTED})
        ax.scatter(values, i + rng.uniform(-0.18, 0.18, len(values)), s=10, color=colour, alpha=0.75, zorder=3,
                   edgecolors="white", linewidths=0.4)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7.5, color=INK)
    ax.set_xlim(-0.02, 1.02)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Dice per test case (mean over classes)", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _display(image, rgb):
    """(C, y, x) normalised or raw channels -> an image to show."""
    if rgb and image.shape[0] == 3:
        x = np.moveaxis(image, 0, -1).astype(np.float32)
        x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
        return x, None
    x = image[0].astype(np.float32)
    low, high = np.percentile(x, [0.5, 99.5])
    return np.clip((x - low) / max(high - low, 1e-6), 0, 1), "gray"


def best_slice(reference, prediction):
    """The slice with the most reference voxels (else predicted voxels, else the middle)."""
    area = (reference > 0).reshape(reference.shape[0], -1).sum(1)
    if not area.any():
        area = (prediction > 0).reshape(prediction.shape[0], -1).sum(1)
    return int(np.argmax(area)) if area.any() else reference.shape[0] // 2


def overlay_figure(image, reference, prediction, uncertainty, classes, rgb, title, path):
    """Image, reference, prediction and uncertainty of one case (3D: the most informative slice).
    image (C, z, y, x); reference, prediction, uncertainty (z, y, x)."""
    z = best_slice(reference, prediction)
    shown, cmap = _display(image[:, z], rgb)
    k = len(classes)
    palette = ListedColormap(["none"] + colours(k))
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.5))
    panels = [("Image", None), ("Reference", reference[z]), ("Prediction", prediction[z]), ("Uncertainty", None)]
    for ax, (name, mask) in zip(axes, panels):
        # under the uncertainty, a dimmed image: bright tissue must not read as high uncertainty
        ax.imshow(shown * 0.35 if name == "Uncertainty" else shown, cmap=cmap, vmin=0, vmax=1)
        if mask is not None:
            masked = np.ma.masked_equal(mask, 0)
            ax.imshow(masked, cmap=palette, vmin=0, vmax=k - 1, alpha=0.5, interpolation="nearest")
            for c in range(1, k):
                if (mask == c).any():
                    ax.contour(mask == c, levels=[0.5], colors=[colours(k)[c - 1]], linewidths=1.0)
        if name == "Uncertainty":
            heat = ax.imshow(np.ma.masked_less(uncertainty[z], 0.05), cmap="magma", vmin=0, vmax=1, alpha=0.8)
            fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6, colors=MUTED)
        ax.set_title(name + (f" (slice {z + 1}/{reference.shape[0]})" if reference.shape[0] > 1 and name == "Image" else ""),
                     fontsize=9, color=INK, loc="left")
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=c, lw=4) for c in colours(k)]
    fig.legend(handles, classes[1:], loc="lower center", ncol=min(len(handles), 6), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
