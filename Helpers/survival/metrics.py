"""Metrics of survival models, with censoring handled by inverse probability of censoring
weights (IPCW, the censoring distribution G estimated by Kaplan-Meier on the training data):

- C-index (Harrell): the share of comparable patient pairs whose predicted risks are in the
  order of their event times (0.5: chance, 1: perfect);
- Uno's C-index: the same, IPCW-weighted up to a time tau, consistent under censoring;
- time-dependent AUC at t (cumulative/dynamic, Uno 2007): how well the risk separates the
  patients with an event before t from those still event-free at t;
- Brier score at t (Graf 1999): squared error of the predicted survival probability at t;
  integrated Brier score (IBS): its average over time (0 is perfect; Kaplan-Meier is the
  no-covariate reference);
- Kaplan-Meier curves, the log-rank test between risk groups and calibration at a horizon.
"""
import numpy as np
from scipy import stats

METRICS = ["C-index", "Uno C-index", "IBS"]
LOWER_IS_BETTER = {"IBS"}


def kaplan_meier(time, event):
    """(distinct times, survival just after each)."""
    time, event = np.asarray(time, float), np.asarray(event, int)
    times = np.unique(time)
    at_risk = np.array([(time >= t).sum() for t in times]) if len(times) < 2000 else \
        len(time) - np.searchsorted(np.sort(time), times, side="left")
    deaths = np.array([((time == t) & (event == 1)).sum() for t in times]) if len(times) < 2000 else \
        np.bincount(np.searchsorted(times, time[event == 1]), minlength=len(times))
    surv = np.cumprod(1 - deaths / np.maximum(at_risk, 1))
    return times, surv


def step(times, values, at, before=False, start=1.0):
    """A right-continuous step function (times, values) evaluated at ``at``; ``before``: its left
    limit (the value just before each point)."""
    idx = np.searchsorted(times, np.asarray(at, float), side="left" if before else "right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], start)
    return out


class Censoring:
    """G(t) = P(C > t), the Kaplan-Meier estimate of the censoring distribution."""

    def __init__(self, time, event):
        self.times, self.surv = kaplan_meier(time, 1 - np.asarray(event, int))

    def __call__(self, t, before=False):
        return np.maximum(step(self.times, self.surv, t, before=before), 1e-8)


def harrell_c(time, event, risk):
    time, event, risk = np.asarray(time, float), np.asarray(event, int), np.asarray(risk, float)
    concordant = permissible = 0.0
    for i in np.where(event == 1)[0]:
        later = time > time[i]
        if not later.any():
            continue
        permissible += later.sum()
        concordant += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return float(concordant / permissible) if permissible else np.nan


def uno_c(time, event, risk, censoring, tau=None):
    time, event, risk = np.asarray(time, float), np.asarray(event, int), np.asarray(risk, float)
    tau = np.max(time) if tau is None else tau
    num = den = 0.0
    for i in np.where((event == 1) & (time < tau))[0]:
        later = time > time[i]
        if not later.any():
            continue
        w = 1.0 / censoring(time[i], before=True) ** 2
        den += w * later.sum()
        num += w * ((risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum())
    return float(num / den) if den else np.nan


def brier(t, surv_t, time, event, censoring):
    """IPCW Brier score at time t of the predicted survival probabilities ``surv_t``."""
    time, event, surv_t = np.asarray(time, float), np.asarray(event, int), np.asarray(surv_t, float)
    died = (time <= t) & (event == 1)
    alive = time > t
    score = np.zeros(len(time))
    score[died] = surv_t[died] ** 2 / censoring(time[died], before=True)
    score[alive] = (1 - surv_t[alive]) ** 2 / censoring(t)
    return float(score.mean())


def integrated_brier(grid, surv_grid, time, event, censoring):
    """Integrated Brier score over ``grid`` (surv_grid: samples x grid)."""
    scores = np.array([brier(t, surv_grid[:, j], time, event, censoring) for j, t in enumerate(grid)])
    if len(grid) < 2:
        return float(scores.mean())
    return float(np.trapz(scores, grid) / (grid[-1] - grid[0]))


def td_auc(t, risk, time, event, censoring):
    """Cumulative/dynamic AUC at t: cases had the event by t, controls are event-free after t."""
    time, event, risk = np.asarray(time, float), np.asarray(event, int), np.asarray(risk, float)
    cases = np.where((time <= t) & (event == 1))[0]
    controls = risk[time > t]
    if not len(cases) or not len(controls):
        return np.nan
    w = 1.0 / censoring(time[cases], before=True)
    hits = np.array([(risk[i] > controls).mean() + 0.5 * (risk[i] == controls).mean() for i in cases])
    return float((w * hits).sum() / w.sum())


def evaluation_grid(time_train, event_train, time_eval, points=50):
    """Times for the integrated Brier score: from the 10th to the 90th percentile of the training
    event times, within the follow-up of the evaluated data."""
    events = np.asarray(time_train)[np.asarray(event_train) == 1]
    low, high = np.quantile(events, [0.1, 0.9])
    high = min(high, np.max(time_eval) * 0.999)
    if high <= low:
        high = low * 1.001 + 1e-9
    return np.linspace(low, high, points)


def score(model_survival, risk, time, event, censoring, grid, horizons):
    """Every metric of a model on (time, event): model_survival(times) gives samples x times."""
    surv_grid = model_survival(grid)
    out = {"C-index": harrell_c(time, event, risk),
           "Uno C-index": uno_c(time, event, risk, censoring, tau=grid[-1]),
           "IBS": integrated_brier(grid, surv_grid, time, event, censoring)}
    surv_h = model_survival(np.asarray(horizons, float))
    for j, h in enumerate(horizons):
        out[f"AUC@{_label(h)}"] = td_auc(h, risk, time, event, censoring)
        out[f"Brier@{_label(h)}"] = brier(h, surv_h[:, j], time, event, censoring)
    return out


def _label(value):
    return f"{value:g}"


def horizon_metrics(horizons):
    return [f"{kind}@{_label(h)}" for h in horizons for kind in ("AUC", "Brier")]


def logrank(time, event, groups):
    """Log-rank test across groups: (chi-square, degrees of freedom, p-value)."""
    time, event, groups = np.asarray(time, float), np.asarray(event, int), np.asarray(groups)
    labels = [g for g in np.unique(groups)]
    if len(labels) < 2:
        return np.nan, 0, np.nan
    times = np.unique(time[event == 1])
    k = len(labels)
    observed_minus_expected = np.zeros(k)
    variance = np.zeros((k, k))
    for t in times:
        at_risk = np.array([((time >= t) & (groups == g)).sum() for g in labels], float)
        deaths = np.array([((time == t) & (event == 1) & (groups == g)).sum() for g in labels], float)
        n, d = at_risk.sum(), deaths.sum()
        if n < 2:
            continue
        expected = d * at_risk / n
        observed_minus_expected += deaths - expected
        factor = d * (n - d) / (n * n * (n - 1))
        variance += factor * (np.diag(at_risk * n) - np.outer(at_risk, at_risk))
    o, v = observed_minus_expected[:-1], variance[:-1, :-1]
    try:
        chi2 = float(o @ np.linalg.solve(v, o))
    except np.linalg.LinAlgError:
        return np.nan, k - 1, np.nan
    return chi2, k - 1, float(stats.chi2.sf(chi2, k - 1))


def calibration(t, surv_t, time, event, bins=5):
    """Predicted vs. observed (Kaplan-Meier) survival at t in quantile groups of the prediction:
    [(mean predicted, observed, patients)]."""
    surv_t = np.asarray(surv_t, float)
    edges = np.unique(np.quantile(surv_t, np.linspace(0, 1, bins + 1)))
    groups = np.clip(np.searchsorted(edges, surv_t, side="right") - 1, 0, max(0, len(edges) - 2))
    out = []
    for g in np.unique(groups):
        mask = groups == g
        times, surv = kaplan_meier(np.asarray(time)[mask], np.asarray(event)[mask])
        out.append((float(surv_t[mask].mean()), float(step(times, surv, t)), int(mask.sum())))
    return out
