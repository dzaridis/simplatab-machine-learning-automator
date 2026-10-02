"""Figures of the survival automator: Kaplan-Meier curves of risk groups, predicted survival
curves, calibration, time-dependent AUC and Brier score over time, metrics and feature importance.
Series take the categorical colours in a fixed order (the same model, the same colour in every
figure); lines also differ by marker so that they stay distinguishable without colour."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from Helpers.clustering.plots import BAR, BEST, GRID, INK, MARKERS, MUTED, OTHER, PALETTE, _save, _style, metric_bars  # noqa: E402,F401
from .metrics import kaplan_meier  # noqa: E402

GROUP_COLOURS = {"Low risk": PALETTE[2], "Intermediate risk": PALETTE[3], "High risk": PALETTE[7]}


def _colour(i):
    return PALETTE[i] if i < len(PALETTE) else OTHER


def km_groups(time, event, groups, title, path, p_value=None, unit=""):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _style(ax)
    for i, label in enumerate(["Low risk", "Intermediate risk", "High risk"]):
        mask = groups == label
        if not mask.any():
            continue
        times, surv = kaplan_meier(time[mask], event[mask])
        colour = GROUP_COLOURS.get(label, _colour(i))
        ax.step(np.concatenate([[0], times]), np.concatenate([[1], surv]), where="post", color=colour, linewidth=2,
                label=f"{label} (n={int(mask.sum())}, {int(event[mask].sum())} events)")
        censored = mask & (event == 0)
        if censored.any():
            ax.plot(time[censored], np.interp(time[censored], times, surv), "|", color=colour, markersize=6, alpha=0.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"Time{f' ({unit})' if unit else ''}", fontsize=8, color=MUTED)
    ax.set_ylabel("Event-free probability (Kaplan-Meier)", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    if p_value is not None and p_value == p_value:
        ax.text(0.99, 0.97, f"log-rank p {'< 0.001' if p_value < 0.001 else f'= {p_value:.3f}'}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def patient_curves(grid, curves, labels, km, title, path):
    """Predicted survival of a few patients (lines) and the Kaplan-Meier curve of all (grey steps)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _style(ax)
    ax.step(np.concatenate([[0], km[0]]), np.concatenate([[1], km[1]]), where="post", color=BAR, linewidth=2.5,
            label="Kaplan-Meier, all test patients")
    for i, (curve, label) in enumerate(zip(curves, labels)):
        ax.plot(grid, curve, color=_colour(i), linewidth=2, label=label)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_xlabel("Time", fontsize=8, color=MUTED)
    ax.set_ylabel("Predicted event-free probability", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def calibration(points, horizon, title, path):
    """Predicted vs. observed survival at the horizon, by group of predicted survival."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    _style(ax)
    ax.plot([0, 1], [0, 1], color=BAR, linestyle="--", linewidth=1.2, label="Perfect calibration")
    pred = [p for p, _, _ in points]
    obs = [o for _, o, _ in points]
    ax.plot(pred, obs, color=BEST, marker="o", markersize=8, linewidth=2, label="Groups of patients")
    for p, o, n in points:
        ax.annotate(f"n={n}", (p, o), textcoords="offset points", xytext=(6, -10), fontsize=7, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Predicted event-free probability at {horizon:g}", fontsize=8, color=MUTED)
    ax.set_ylabel("Observed (Kaplan-Meier)", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def over_time(grid, series, ylabel, title, path, reference=None):
    """A line per model over time (e.g. AUC(t) or Brier(t)); ``reference``: a grey dashed line."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    _style(ax)
    if reference is not None:
        ax.plot(grid, reference[1], color=BAR, linestyle="--", linewidth=2, label=reference[0])
    for i, (name, values) in enumerate(series.items()):
        ax.plot(grid, values, color=_colour(i), linewidth=2, marker=MARKERS[i % len(MARKERS)], markevery=max(1, len(grid) // 8),
                markersize=5, label=name)
    ax.set_xlabel("Time", fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def importance(table, title, path, top=15):
    """Drop of the C-index when a feature is shuffled (mean and SD over repeats)."""
    data = table.head(top)
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(data) + 1.6))
    _style(ax)
    ax.grid(False, axis="y")
    ax.barh(range(len(data)), data["mean"], xerr=data["std"], color=BEST, height=0.62,
            error_kw={"ecolor": MUTED, "elinewidth": 1})
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(list(data.index), fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color=GRID)
    ax.set_xlabel("Drop of the test C-index when the feature is shuffled", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)
