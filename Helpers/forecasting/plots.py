"""Figures of the forecasting automator: forecasts against the observed series, test errors,
error by horizon step and integrated-gradients attributions."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
FORECAST = "#2a78d6"
FORECAST_LIGHT = "#9ec5f4"
BASELINE = "#b9b8b3"
HORIZON_SHADE = "#f3f2ef"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def pick_series(ids, count=6):
    """Up to ``count`` series spread over the sorted IDs (the same ones in every figure)."""
    ids = sorted(ids, key=str)
    if len(ids) <= count:
        return ids
    return [ids[int(i)] for i in np.linspace(0, len(ids) - 1, count)]


def forecast_grid(history, future, column, ids, title, path, context):
    """For each series: the observed values (ink) and the forecast (blue) over the horizon.
    ``future`` has unique_id, ds, the forecasts in ``column`` and, for the test set, the
    observed values in ``y``."""
    n = len(ids)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.9 * rows), squeeze=False)
    for ax, uid in zip(axes.flat, ids):
        past = history[history["unique_id"] == uid].tail(context)
        ahead = future[future["unique_id"] == uid]
        _style(ax)
        if len(ahead):
            ax.axvspan(ahead["ds"].iloc[0], ahead["ds"].iloc[-1], color=HORIZON_SHADE, zorder=0)
        observed_x, observed_y = list(past["ds"]), list(past["y"])
        if "y" in ahead:
            observed_x += list(ahead["ds"])
            observed_y += list(ahead["y"])
        ax.plot(observed_x, observed_y, color=INK, linewidth=1.6, label="Observed")
        if len(past) and len(ahead):
            # Connect the forecast to the last observed point
            ax.plot([past["ds"].iloc[-1], ahead["ds"].iloc[0]], [past["y"].iloc[-1], ahead[column].iloc[0]],
                    color=FORECAST, linewidth=1.6, linestyle=":")
        ax.plot(ahead["ds"], ahead[column], color=FORECAST, linewidth=2, marker="o", markersize=3.5, label="Forecast")
        ax.set_title(f"Series {uid}", fontsize=9, color=INK, loc="left")
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metric_bars(table, best, path, baseline_name):
    """One panel per metric: a bar per model (best model highlighted, baseline in gray)."""
    metrics = [m for m in table.columns if table[m].notna().any()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 0.42 * len(table) + 1.4), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        values = table[metric].sort_values(ascending=False)
        colors = [BASELINE if name == baseline_name else FORECAST if name == best else FORECAST_LIGHT for name in values.index]
        ax.barh(range(len(values)), values.to_numpy(), color=colors, height=0.7)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(values.index, fontsize=8, color=INK)
        _style(ax)
        ax.grid(axis="y", visible=False)
        top = np.nanmax(values.to_numpy()) if len(values) else 1
        for i, value in enumerate(values.to_numpy()):
            if np.isfinite(value):
                ax.text(value + 0.01 * top, i, f"{value:.3g}", va="center", fontsize=7.5, color=MUTED)
        ax.set_xlim(0, top * 1.18 if np.isfinite(top) and top > 0 else 1)
        if metric == "MASE":
            ax.axvline(1, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(f"{metric} (lower is better)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(values, row_labels, col_labels, title, path, fmt="{:.3g}", xlabel=None):
    """Sequential heatmap with the values written in the cells (when they fit)."""
    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(col_labels) + 2.5), 0.42 * len(row_labels) + 1.6))
    image = ax.imshow(values, cmap=BLUES, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, color=MUTED)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    if len(col_labels) <= 24:
        finite = values[np.isfinite(values)]
        middle = (finite.min() + finite.max()) / 2 if finite.size else 0
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value):
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=7,
                        color="white" if value > middle else INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7, colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def attributions(importance, temporal, temporal_rows, temporal_cols, title, path):
    """Integrated gradients: share of the attribution per input (left) and over time (right)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, max(3.2, 0.42 * max(len(importance), len(temporal_rows)) + 1.6)),
                                      gridspec_kw={"width_ratios": [1, 1.6]})
    ordered = importance.sort_values()
    left.barh(range(len(ordered)), ordered.to_numpy(), color=FORECAST, height=0.7)
    left.set_yticks(range(len(ordered)))
    left.set_yticklabels(ordered.index, fontsize=8, color=INK)
    _style(left)
    left.grid(axis="y", visible=False)
    for i, value in enumerate(ordered.to_numpy()):
        left.text(value + 0.5, i, f"{value:.1f}%", va="center", fontsize=7.5, color=MUTED)
    left.set_xlim(0, max(ordered.max() * 1.2, 1))
    left.set_title("Share of the attribution (%)", fontsize=9, color=INK, loc="left")

    image = right.imshow(temporal, cmap=BLUES, aspect="auto")
    step = max(1, len(temporal_cols) // 16)
    right.set_xticks(range(0, len(temporal_cols), step))
    right.set_xticklabels(temporal_cols[::step], fontsize=7.5, color=MUTED)
    right.set_yticks(range(len(temporal_rows)))
    right.set_yticklabels(temporal_rows, fontsize=8, color=INK)
    right.set_xlabel("Time step relative to the forecast start (0 = first forecast point)", fontsize=8, color=MUTED)
    for side in right.spines.values():
        side.set_visible(False)
    right.set_title("Mean |attribution| over time", fontsize=9, color=INK, loc="left")
    fig.colorbar(image, ax=right, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
