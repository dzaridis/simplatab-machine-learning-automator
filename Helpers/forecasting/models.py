"""The ten forecasting networks (neuralforecast) and their hyperparameters.

All are global models: one network learns from all the training series and forecasts any
series, including series it has not seen. Each forecasts the H next points at once from the
``input_size`` (lookback) previous points and, when it supports them, the covariates:
future (known in advance), past (observed up to the present) and static.
"""
import inspect
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

# Windows sampled per training step (neuralforecast's default, 1024, is slow on CPU)
WINDOWS_BATCH_SIZE = 256
LEARNING_RATES = [3e-4, 1e-3, 3e-3]
LOOKBACK_MULTIPLES = [1, 2, 3, 5]


def _patchtst(args):
    """Patches of at most 16 points, at least two per lookback."""
    patch = max(2, min(16, args["input_size"] // 2))
    args.update(patch_len=patch, stride=max(1, patch // 2))


def _dlinear(args):
    """Moving average (trend) window: odd and within the lookback."""
    window = min(25, args["input_size"])
    args["moving_avg_window"] = max(3, window if window % 2 else window - 1)


@dataclass(frozen=True)
class ForecastModel:
    key: str             # neuralforecast class, also the model name
    family: str          # "mlp" | "attention" | "convolution"
    description: str
    default: bool = True
    slow: bool = False
    fixed: Dict = field(default_factory=dict)   # CPU-friendly settings
    tune: Dict = field(default_factory=dict)    # model-specific hyperparameter -> choices
    adjust: Optional[Callable] = None           # settings that depend on the lookback

    @property
    def name(self):
        return self.key

    @property
    def cls(self):
        import neuralforecast.models as nf_models
        return getattr(nf_models, self.key)

    @property
    def covariates(self):
        """Which covariates the network uses: future, past, static."""
        flags = COVARIATE_SUPPORT[self.key]
        return {"future": flags[0], "past": flags[1], "static": flags[2]}


# Future / past / static covariates (EXOGENOUS_FUTR, EXOGENOUS_HIST, EXOGENOUS_STAT of
# neuralforecast 3.1), listed here so the web pages need not import neuralforecast
COVARIATE_SUPPORT = {
    "NHITS": (True, True, True), "NBEATSx": (True, True, True), "TFT": (True, True, True),
    "TiDE": (True, True, True), "BiTCN": (True, True, True), "TCN": (True, True, True),
    "KAN": (True, True, True), "TimesNet": (True, False, False), "PatchTST": (False, False, False),
    "DLinear": (False, False, False),
}

MODELS = [
    ForecastModel("NHITS", "mlp", "Multi-rate MLP with hierarchical interpolation: accurate and fast, a strong default (Challu et al., AAAI 2023)."),
    ForecastModel("NBEATSx", "mlp", "Interpretable MLP stacks (trend, seasonality) extended with covariates (Olivares et al., 2022)."),
    ForecastModel("TiDE", "mlp", "Time-series dense encoder-decoder from Google: MLP speed, transformer-level accuracy (Das et al., 2023).",
                  tune={"hidden_size": [128, 256, 512]}),
    ForecastModel("KAN", "mlp", "Kolmogorov-Arnold network: learnable activation functions instead of fixed ones (2024)."),
    ForecastModel("DLinear", "mlp", "Linear model on trend and remainder: the simple baseline that beat many transformers (Zeng et al., AAAI 2023). Uses the target history only.",
                  adjust=_dlinear),
    ForecastModel("TFT", "attention", "Temporal Fusion Transformer: attention with variable selection for static, past and future covariates (Lim et al., 2021).",
                  fixed={"hidden_size": 64}, tune={"hidden_size": [32, 64, 128]}),
    ForecastModel("PatchTST", "attention", "Transformer over patches of the series, state of the art for long horizons (Nie et al., ICLR 2023). Uses the target history only.",
                  fixed={"hidden_size": 64}, adjust=_patchtst),
    ForecastModel("BiTCN", "convolution", "Bidirectional temporal convolutional network, parameter-efficient (Sprangers et al., 2023).",
                  tune={"hidden_size": [16, 32]}),
    ForecastModel("TCN", "convolution", "Temporal convolutional network with dilated causal convolutions (Bai et al., 2018).",
                  tune={"encoder_hidden_size": [64, 128]}),
    ForecastModel("TimesNet", "convolution", "2D convolutions over the periods of the series (Wu et al., ICLR 2023). Uses future covariates only. Slow without a GPU.",
                  default=False, slow=True,
                  fixed={"hidden_size": 32, "conv_hidden_size": 32, "top_k": 3, "windows_batch_size": 32}),
]
BY_KEY = {m.key: m for m in MODELS}


def lookback_choices(horizon, longest):
    """Lookback candidates: multiples of the horizon, within the longest useful lookback."""
    choices = [m * horizon for m in LOOKBACK_MULTIPLES if m * horizon <= max(horizon, longest)]
    return choices or [horizon]


def default_config(model, horizon, lookback):
    return {"input_size": int(lookback), "learning_rate": 1e-3}


def search_space(model, horizon, longest, lookback=None, trials=0, seed=0):
    """The configurations to compare: the default first, then ``trials`` random others.
    A fixed ``lookback`` is not tuned."""
    lookbacks = [lookback] if lookback else lookback_choices(horizon, longest)
    default_lookback = lookback or (2 * horizon if 2 * horizon in lookbacks else lookbacks[-1])
    configs = [default_config(model, horizon, default_lookback)]
    if trials <= 0:
        return configs
    grid = [{"input_size": l, "learning_rate": lr} for l in lookbacks for lr in LEARNING_RATES]
    for name, choices in model.tune.items():
        grid = [dict(c, **{name: v}) for c in grid for v in choices]
    grid = [c for c in grid if c != configs[0]]
    random.Random(seed).shuffle(grid)
    return configs + grid[:trials]


def build(model, horizon, config, futr, hist, stat, max_steps, seed=1, accelerator="cpu"):
    """The neuralforecast network for a configuration."""
    cls = model.cls
    accepted = set(inspect.signature(cls.__init__).parameters)
    args = {"h": horizon, "max_steps": max_steps, "scaler_type": "robust", "start_padding_enabled": True,
            "random_seed": seed, "windows_batch_size": WINDOWS_BATCH_SIZE, "alias": model.name}
    for (supported, name, columns) in zip(COVARIATE_SUPPORT[model.key],
                                          ("futr_exog_list", "hist_exog_list", "stat_exog_list"), (futr, hist, stat)):
        if supported and columns:
            args[name] = list(columns)
    args.update(model.fixed)
    args.update(config)
    if model.adjust:
        model.adjust(args)
    args = {k: v for k, v in args.items() if k in accepted}
    trainer = {"enable_progress_bar": False, "enable_model_summary": False, "logger": False,
               "enable_checkpointing": False, "accelerator": accelerator, "devices": 1}
    return cls(**args, **trainer)
