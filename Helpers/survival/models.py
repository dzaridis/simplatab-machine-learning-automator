"""Survival models with one interface: ``fit(X, time, event)``, ``predict_risk(X)`` (higher means
an earlier event) and ``predict_survival(X, times)`` (samples x times, the probability of being
event-free at each time).

- Cox proportional hazards (ridge-penalised partial likelihood, Breslow baseline);
- Weibull and log-normal accelerated failure time (AFT) models (maximum likelihood);
- XGBoost with the Cox objective (gradient-boosted trees, Breslow baseline) and the AFT objective;
- DeepSurv (Katzman et al., 2018): a neural network trained on the Cox partial likelihood;
- DeepHit (Lee et al., 2018): a network predicting the distribution of the event time over a
  time grid, trained on the likelihood and a ranking loss;
- Logistic-Hazard / Nnet-survival (Gensheimer & Narasimhan, 2019; Kvamme & Borgan, 2021): a
  network predicting the hazard of every interval of a time grid.

The saved models store this module by value (cloudpickle): it only imports numpy, scipy and, for
the models that use them, xgboost or torch.
"""
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    family: str          # statistical | boosting | deep
    description: str
    default: bool
    slow: bool = False


MODELS = [
    Model("coxph", "Cox PH", "statistical", "The reference model: proportional hazards, a hazard ratio per feature (ridge penalty).", True),
    Model("weibull_aft", "Weibull AFT", "statistical", "Parametric accelerated failure time model: features stretch or shrink the time to event.", True),
    Model("lognormal_aft", "Log-normal AFT", "statistical", "AFT model with a log-normal time distribution: hazards that rise then fall.", False),
    Model("xgb_cox", "XGBoost Cox", "boosting", "Gradient-boosted trees on the Cox likelihood: non-linear effects and interactions.", True),
    Model("xgb_aft", "XGBoost AFT", "boosting", "Gradient-boosted trees on the AFT likelihood: predicts the time to event directly.", False),
    Model("deepsurv", "DeepSurv", "deep", "Neural network trained on the Cox partial likelihood (Katzman et al. 2018).", True),
    Model("deephit", "DeepHit", "deep", "Neural network predicting the event-time distribution, no proportional hazards (Lee et al. 2018).", True),
    Model("logistic_hazard", "Logistic-Hazard", "deep", "Discrete-time neural hazard model, Nnet-survival (Gensheimer 2019, Kvamme 2021).", False),
]
BY_KEY = {m.key: m for m in MODELS}
FAMILIES = [("statistical", "Statistical", "Interpretable classics: hazard ratios, time ratios."),
            ("boosting", "Gradient boosting", "Trees: strong on tabular data, non-linear effects."),
            ("deep", "Deep learning", "Neural networks; benefit from larger cohorts.")]


def breslow(risk_score, time, event):
    """Breslow estimate of the cumulative baseline hazard: (event times, H0 at each)."""
    order = np.argsort(time)
    t, e, r = time[order], event[order], np.exp(risk_score[order] - risk_score.max())
    shift = risk_score.max()
    at_risk = np.cumsum(r[::-1])[::-1]
    times = np.unique(t[e == 1])
    first = np.searchsorted(t, times, side="left")
    deaths = np.array([((t == u) & (e == 1)).sum() for u in times])
    return times, np.cumsum(deaths / at_risk[first]) * np.exp(-shift)


def _cum_hazard(times, cum, at):
    idx = np.searchsorted(times, at, side="right") - 1
    return np.where(idx >= 0, cum[np.clip(idx, 0, len(cum) - 1)], 0.0)


class CoxPH:
    """Cox model, partial likelihood with Breslow ties and a ridge penalty (features standardised)."""

    def __init__(self, penalty=0.01):
        self.penalty = penalty

    def fit(self, X, time, event):
        X = np.asarray(X, float)
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mean_) / self.std_
        order = np.argsort(-time)              # descending: risk sets are prefixes
        Zs, es = Z[order], event[order].astype(float)
        n = len(Z)

        def loss(beta):
            eta = Zs @ beta
            m = eta.max()
            w = np.exp(eta - m)
            cw = np.cumsum(w)
            cwz = np.cumsum(w[:, None] * Zs, axis=0)
            ll = (es * (eta - m - np.log(cw))).sum()
            grad = (es[:, None] * (Zs - cwz / cw[:, None])).sum(0)
            pen = 0.5 * self.penalty * n * beta @ beta
            return -(ll) / n + pen / n, -grad / n + self.penalty * beta

        res = optimize.minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
        self.coef_ = res.x / self.std_          # per unit of the (preprocessed) feature
        self.baseline_ = breslow(self._eta(X), time, event)
        return self

    def _eta(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.coef_

    def predict_risk(self, X):
        return self._eta(X)

    def predict_survival(self, X, times):
        H0 = _cum_hazard(*self.baseline_, np.asarray(times, float))
        return np.exp(-np.outer(np.exp(self._eta(X)), H0))


class ParametricAFT:
    """log T = mu + x.beta + sigma * W, with W standard extreme-value (Weibull) or normal (log-normal)."""

    def __init__(self, distribution="weibull", penalty=0.01):
        self.distribution = distribution
        self.penalty = penalty

    def _logpdf_logsf(self, z):
        if self.distribution == "weibull":
            return z - np.exp(z), -np.exp(z)
        return stats.norm.logpdf(z), stats.norm.logsf(z)

    def fit(self, X, time, event):
        X = np.asarray(X, float)
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mean_) / self.std_
        y = np.log(time)
        d = Z.shape[1]

        def loss(params):
            mu, log_sigma, beta = params[0], params[1], params[2:]
            sigma = np.exp(log_sigma)
            z = (y - mu - Z @ beta) / sigma
            logpdf, logsf = self._logpdf_logsf(z)
            ll = np.where(event == 1, logpdf - log_sigma, logsf).sum()
            return -ll / len(y) + 0.5 * self.penalty * beta @ beta

        start = np.concatenate([[y.mean(), np.log(y.std() + 1e-3)], np.zeros(d)])
        res = optimize.minimize(loss, start, method="L-BFGS-B")
        self.mu_, self.sigma_ = res.x[0], np.exp(res.x[1])
        self.coef_ = res.x[2:] / self.std_
        return self

    def _location(self, X):
        return self.mu_ + (np.asarray(X, float) - self.mean_) @ self.coef_

    def predict_risk(self, X):
        return -self._location(X)               # shorter predicted times: higher risk

    def predict_survival(self, X, times):
        z = (np.log(np.asarray(times, float))[None, :] - self._location(X)[:, None]) / self.sigma_
        return np.exp(-np.exp(z)) if self.distribution == "weibull" else stats.norm.sf(z)


class XGBoostCox:
    def __init__(self, seed=0, n_estimators=300):
        self.seed, self.n_estimators = seed, n_estimators

    def fit(self, X, time, event):
        import xgboost as xgb
        self.model_ = xgb.XGBRegressor(objective="survival:cox", n_estimators=self.n_estimators, learning_rate=0.05,
                                       max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                       reg_lambda=1.0, random_state=self.seed, n_jobs=-1, tree_method="hist")
        self.model_.fit(np.asarray(X, float), np.where(event == 1, time, -time))
        self.baseline_ = breslow(self.predict_risk(X), time, event)
        return self

    def predict_risk(self, X):
        return self.model_.predict(np.asarray(X, float), output_margin=True)

    def predict_survival(self, X, times):
        H0 = _cum_hazard(*self.baseline_, np.asarray(times, float))
        return np.exp(-np.outer(np.exp(self.predict_risk(X)), H0))


class XGBoostAFT:
    def __init__(self, seed=0, n_estimators=300, sigma=1.0):
        self.seed, self.n_estimators, self.sigma = seed, n_estimators, sigma

    def fit(self, X, time, event):
        import xgboost as xgb
        data = xgb.DMatrix(np.asarray(X, float))
        data.set_float_info("label_lower_bound", time)
        data.set_float_info("label_upper_bound", np.where(event == 1, time, np.inf))
        params = {"objective": "survival:aft", "aft_loss_distribution": "normal",
                  "aft_loss_distribution_scale": self.sigma, "learning_rate": 0.05, "max_depth": 3,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "seed": self.seed,
                  "tree_method": "hist", "verbosity": 0}
        self.booster_ = xgb.train(params, data, num_boost_round=self.n_estimators)
        return self

    def _log_time(self, X):
        import xgboost as xgb
        return self.booster_.predict(xgb.DMatrix(np.asarray(X, float)), output_margin=True)

    def predict_risk(self, X):
        return -self._log_time(X)

    def predict_survival(self, X, times):
        z = (np.log(np.asarray(times, float))[None, :] - self._log_time(X)[:, None]) / self.sigma
        return stats.norm.sf(z)


# ---------------------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------------------

class _Network:
    """An MLP trained with early stopping on 15% of the training data."""

    def __init__(self, hidden=(64, 64), dropout=0.1, epochs=200, batch_size=128, learning_rate=1e-3,
                 weight_decay=1e-4, patience=20, bins=30, seed=0):
        self.hidden, self.dropout, self.epochs, self.batch_size = tuple(hidden), dropout, epochs, batch_size
        self.learning_rate, self.weight_decay, self.patience, self.bins, self.seed = \
            learning_rate, weight_decay, patience, bins, seed

    def _net(self, d, out):
        from torch import nn
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, out))
        return nn.Sequential(*layers)

    def _grid(self, time, event):
        """Interval edges: quantiles of the event times (0 first)."""
        cuts = np.unique(np.quantile(time[event == 1], np.linspace(0, 1, self.bins + 1)[1:]))
        return np.concatenate([[0.0], cuts])

    def _index(self, time):
        """Interval of each time (0 .. len(grid)-2; past the grid: the last)."""
        return np.clip(np.searchsorted(self.grid_, time, side="left") - 1, 0, len(self.grid_) - 2)

    def fit(self, X, time, event):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, np.float32)
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-6
        Z = ((X - self.mean_) / self.std_).astype(np.float32)
        self._prepare_targets(time, event)
        n = len(Z)
        valid = rng.random(n) < 0.15 if n >= 40 else np.zeros(n, bool)
        train = ~valid
        self.net_ = self._net(Z.shape[1], self._outputs())
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        Zt = torch.as_tensor(Z)
        best, best_state, waited = np.inf, None, 0
        train_idx = np.where(train)[0]
        batch = int(min(self.batch_size, max(16, len(train_idx))))
        for _ in range(self.epochs):
            self.net_.train()
            rng.shuffle(train_idx)
            for i in range(0, len(train_idx), batch):
                idx = train_idx[i:i + batch]
                if len(idx) < 2:
                    continue
                loss = self._loss(self.net_(Zt[idx]), idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if valid.any():
                self.net_.eval()
                with torch.no_grad():
                    current = float(self._loss(self.net_(Zt[valid]), np.where(valid)[0]))
                if current < best - 1e-4:
                    best, waited = current, 0
                    best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                else:
                    waited += 1
                    if waited >= self.patience:
                        break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        self._finish(X, time, event)
        return self

    def _output(self, X):
        import torch
        Z = ((np.asarray(X, np.float32) - self.mean_) / self.std_).astype(np.float32)
        with torch.no_grad():
            return self.net_(torch.as_tensor(Z)).numpy()

    def _finish(self, X, time, event):
        pass


class DeepSurv(_Network):
    def _prepare_targets(self, time, event):
        import torch
        self.t_ = torch.as_tensor(time, dtype=torch.float32)
        self.e_ = torch.as_tensor(event, dtype=torch.float32)

    def _outputs(self):
        return 1

    def _loss(self, out, idx):
        import torch
        eta = out[:, 0]
        t, e = self.t_[idx], self.e_[idx]
        order = torch.argsort(t, descending=True)
        eta, e = eta[order], e[order]
        log_risk = torch.logcumsumexp(eta, 0)
        return -((eta - log_risk) * e).sum() / e.sum().clamp(min=1)

    def _finish(self, X, time, event):
        self.baseline_ = breslow(self.predict_risk(X), time, event)

    def predict_risk(self, X):
        return self._output(X)[:, 0]

    def predict_survival(self, X, times):
        H0 = _cum_hazard(*self.baseline_, np.asarray(times, float))
        return np.exp(-np.outer(np.exp(self.predict_risk(X)), H0))


class _Discrete(_Network):
    """Networks on a grid of time intervals."""

    def _prepare_targets(self, time, event):
        import torch
        self.grid_ = self._grid(time, event)
        self.k_ = torch.as_tensor(self._index(time), dtype=torch.long)
        self.e_ = torch.as_tensor(event, dtype=torch.float32)
        self.t_ = torch.as_tensor(time, dtype=torch.float32)

    def _outputs(self):
        return len(self.grid_) - 1

    def _interval_survival(self, X):
        raise NotImplementedError

    def predict_survival(self, X, times):
        """Survival at the interval ends, interpolated linearly in time (1 at time 0)."""
        knots = np.hstack([np.ones((len(X), 1)), self._interval_survival(X)])   # survival at the grid times
        times = np.clip(np.atleast_1d(np.asarray(times, float)), self.grid_[0], self.grid_[-1])
        right = np.clip(np.searchsorted(self.grid_, times, side="left"), 1, len(self.grid_) - 1)
        left = right - 1
        frac = (times - self.grid_[left]) / np.maximum(self.grid_[right] - self.grid_[left], 1e-12)
        return knots[:, left] * (1 - frac) + knots[:, right] * frac

    def predict_risk(self, X):
        """Minus the restricted mean survival time over the grid."""
        S = np.hstack([np.ones((len(X), 1)), self._interval_survival(X)])
        widths = np.diff(self.grid_)
        return -((S[:, :-1] + S[:, 1:]) / 2 * widths).sum(1)


class DeepHit(_Discrete):
    alpha = 0.2      # weight of the ranking loss
    sigma = 0.1

    def _outputs(self):
        return len(self.grid_)              # intervals + "after the grid"

    def _loss(self, out, idx):
        import torch
        p = torch.softmax(out, 1)
        k, e = self.k_[idx], self.e_[idx]
        cdf = torch.cumsum(p, 1)
        rows = torch.arange(len(idx))
        nll_event = -torch.log(p[rows, k] + 1e-7)
        nll_cens = -torch.log(1 - cdf[rows, k] + 1e-7)
        nll = (e * nll_event + (1 - e) * nll_cens).mean()
        # Ranking: a patient with an event at k should have a higher CDF at k than those event-free longer
        t = self.t_[idx]
        f_at = cdf[:, k]                    # [j, i]: CDF of patient j at the interval of patient i
        diag = f_at.diagonal()
        comparable = (e[:, None] == 1) & (t[:, None] < t[None, :])
        rank = torch.exp(-(diag[:, None] - f_at.T) / self.sigma) * comparable
        return nll + self.alpha * rank.sum() / comparable.sum().clamp(min=1)

    def _interval_survival(self, X):
        p = np.exp(self._output(X) - self._output(X).max(1, keepdims=True))
        p = p / p.sum(1, keepdims=True)
        return 1 - np.cumsum(p, 1)[:, :-1]


class LogisticHazard(_Discrete):
    def _loss(self, out, idx):
        import torch
        k, e = self.k_[idx], self.e_[idx]
        bins = torch.arange(out.shape[1])[None, :]
        target = ((bins == k[:, None]) & (e[:, None] == 1)).float()
        mask = (bins <= k[:, None]).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target, reduction="none")
        return (loss * mask).sum() / len(idx)

    def _interval_survival(self, X):
        hazard = 1 / (1 + np.exp(-self._output(X)))
        return np.cumprod(1 - hazard, 1)


def build(key, params):
    seed = int(params.get("seed", 42))
    epochs = int(params.get("epochs", 200))
    if key == "coxph":
        return CoxPH(penalty=float(params.get("penalty", 0.01)))
    if key == "weibull_aft":
        return ParametricAFT("weibull", penalty=float(params.get("penalty", 0.01)))
    if key == "lognormal_aft":
        return ParametricAFT("lognormal", penalty=float(params.get("penalty", 0.01)))
    if key == "xgb_cox":
        return XGBoostCox(seed=seed)
    if key == "xgb_aft":
        return XGBoostAFT(seed=seed)
    networks = {"deepsurv": DeepSurv, "deephit": DeepHit, "logistic_hazard": LogisticHazard}
    return networks[key](epochs=epochs, seed=seed, learning_rate=float(params.get("learning_rate", 1e-3)))


class SimplatabSurvival:
    """A trained survival model, from the columns of Train.csv to risks and survival curves:
    preprocessing (imputation, scaling, one-hot encoding) and the model. Stored with cloudpickle:
    loading it needs numpy, pandas, scipy, scikit-learn and cloudpickle (plus xgboost or torch
    for those models), not Simplatab."""

    def __init__(self, name, key, numeric, categorical, featurizer, model, horizons):
        self.name, self.key = name, key
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.featurizer, self.model, self.horizons = featurizer, model, list(horizons)

    def transform(self, frame):
        import pandas as pd
        out = pd.DataFrame(index=frame.index)
        for column in self.numeric:
            out[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        for column in self.categorical:
            out[column] = frame[column].map(lambda v: "missing" if v is None or (isinstance(v, float) and v != v) else str(v))
        return np.asarray(self.featurizer.transform(out), dtype=float)

    def predict_risk(self, frame):
        """Risk score of every row: higher means an earlier event (only the order is meaningful)."""
        return np.asarray(self.model.predict_risk(self.transform(frame)), float)

    def predict_survival(self, frame, times=None):
        """Probability of being event-free at each time (default: the horizons of the run)."""
        import pandas as pd
        times = self.horizons if times is None else list(times)
        S = self.model.predict_survival(self.transform(frame), np.asarray(times, float))
        return pd.DataFrame(S, index=frame.index, columns=[f"S({t:g})" for t in times])


def save(model, path):
    import sys
    import cloudpickle
    module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(module)
    try:
        with open(path, "wb") as f:
            cloudpickle.dump(model, f)
    finally:
        cloudpickle.unregister_pickle_by_value(module)


def requirements(key):
    base = ["numpy==1.23.5", "pandas==2.0.3", "scipy==1.11.4", "scikit-learn==1.3.1", "cloudpickle==3.1.2"]
    if key.startswith("xgb"):
        return base + ["xgboost==1.7.6"]
    if key in ("deepsurv", "deephit", "logistic_hazard"):
        return base + ["torch==2.8.0"]
    return base
