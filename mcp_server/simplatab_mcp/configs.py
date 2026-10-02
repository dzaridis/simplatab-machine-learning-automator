"""From the configuration an agent gives (any field left out keeps its default) to the parameters
of the Simplatab pipelines, validated against the data summary exactly as the forms of the web
application are. Python 3.9 compatible."""
import os

from . import contracts


class ConfigError(ValueError):
    """An invalid configuration, reported to the agent as is."""


def _gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _pretrained():
    # SIMPLATAB_PRETRAINED=0: networks without pretrained weights (offline use, tests)
    return os.environ.get("SIMPLATAB_PRETRAINED", "1") != "0"


class _Reader:
    """Reads the fields of a configuration, checks types and ranges, and refuses unknown fields."""

    def __init__(self, config, allowed):
        self.config = dict(config or {})
        unknown = sorted(set(self.config) - set(allowed))
        if unknown:
            raise ConfigError(f"Unknown configuration field(s) {', '.join(unknown)}; the fields are {', '.join(allowed)}.")

    def has(self, name):
        return self.config.get(name) is not None

    def number(self, name, cast, low, high, default):
        if not self.has(name):
            return default
        try:
            value = cast(self.config[name])
        except (TypeError, ValueError):
            raise ConfigError(f"{name} must be a number.")
        if cast is int and float(self.config[name]) != value:
            raise ConfigError(f"{name} must be an integer.")
        if not low <= value <= high:
            raise ConfigError(f"{name} must be between {low} and {high} (got {value}).")
        return value

    def choice(self, name, choices, default):
        if not self.has(name):
            return default
        value = self.config[name]
        if value not in choices:
            raise ConfigError(f"{name} must be one of {', '.join(map(str, choices))} (got {value!r}).")
        return value

    def flags(self, name, keys, default):
        value = dict(default)
        given = self.config.get(name) or {}
        if not isinstance(given, dict):
            raise ConfigError(f"{name} is an object of booleans: {', '.join(keys)}.")
        for key, flag in given.items():
            if key not in keys:
                raise ConfigError(f"Unknown {name} option {key!r}; the options are {', '.join(keys)}.")
            value[key] = bool(flag)
        return value

    def models(self, available, default=None):
        keys = [m["key"] for m in available]
        if not self.has("models"):
            chosen = default if default is not None else [m["key"] for m in available if m.get("default")]
            return chosen or keys[:1]
        value = self.config["models"]
        if isinstance(value, str):
            value = [value]
        if value == ["all"]:
            return keys
        unknown = [v for v in value if v not in keys]
        if unknown:
            raise ConfigError(f"Unknown model(s) {', '.join(map(str, unknown))}; the models are {', '.join(keys)}.")
        if not value:
            raise ConfigError("Select at least one model.")
        return list(dict.fromkeys(value))


def _names(available, keys):
    by_key = {m["key"]: m["name"] for m in available}
    return [by_key[k] for k in keys]


# ---------------------------------------------------------------------------------------
# Automators
# ---------------------------------------------------------------------------------------

def _tabular(config, summary):
    schema = contracts.config_schema("tabular")
    r = _Reader(config, schema)
    available = contracts.models("tabular")
    selected = r.models(available)
    max_folds = max(2, min(20, summary.get("min_class_count", 2)))
    k = r.number("k_folds", int, 2, max_folds, min(5, max_folds))
    search = r.choice("hyperparameter_search", ["none", "randomized", "exhaustive"], "randomized")
    bias = config.get("bias_feature") if config else None
    if bias is not None and bias not in summary.get("features", []):
        raise ConfigError(f"bias_feature must be a feature column of Train.csv ({', '.join(summary.get('features', []))}).")
    params = {
        "BiasAssessment": bias is not None,
        "Feature": bias if bias is not None else "None",
        "number_of_k_folds": k,
        "apply_grid_search": {"enabled": search != "none",
                              "type": {"Randomized": search != "exhaustive", "Exhaustive": search == "exhaustive"}},
        "Correlation Limit": r.number("correlation_limit", float, 0.1, 1.0, 0.7),
        "Metric For Threshold Optimization": r.choice("threshold_metric", contracts.THRESHOLD_METRICS, "Balanced Accuracy"),
        "Machine Learning Models": {name: key in selected for key, name, _, _ in contracts.TABULAR_MODELS},
    }
    return params, _names(available, selected)


def _image(config, summary):
    volume3d = summary.get("volume3d")
    requested = (config or {}).get("dim")
    if requested not in (None, 2, 3, "2", "3"):
        raise ConfigError("dim must be 2 or 3.")
    dim = 3 if volume3d and str(requested) != "2" else 2
    if str(requested) == "3" and not volume3d:
        raise ConfigError("dim=3 needs zips of 3D studies (DICOM series or NIfTI volumes, without PNG/JPEG images).")
    if dim == 2 and summary.get("errors"):
        raise ConfigError("The data can only be classified in 3D: " + " ".join(summary["errors"]))
    schema = contracts.config_schema("image-classification", dim)
    r = _Reader(config, schema)
    available = contracts.models("image-classification", dim)
    source = volume3d if dim == 3 else summary
    classes = source["classes"]
    positive = r.choice("positive_class", classes, source.get("positive_class")) if len(classes) == 2 else source.get("positive_class")
    common = {
        "models": r.models(available),
        "mode": r.choice("mode", ["features", "finetune"], "features"),
        "metric": r.choice("threshold_metric", contracts.THRESHOLD_METRICS, "Balanced Accuracy"),
        "classes": classes,
        "positive_class": positive,
        "window": r.choice("window", contracts.CT_WINDOWS, "auto"),
        "learning_rate": r.number("learning_rate", float, 1e-6, 1e-2, 1e-4),
        "pretrained": _pretrained(),
    }
    if dim == 2:
        max_folds = max(2, min(20, summary["min_class_count"]))
        params = dict(common,
                      k_folds=r.number("k_folds", int, 2, max_folds, min(5, max_folds)),
                      volume=r.choice("volume", ["middle", "mip"], "middle"),
                      augmentation=r.flags("augmentation", ["horizontal_flip", "vertical_flip", "rotation", "intensity"],
                                           {"horizontal_flip": True, "vertical_flip": False, "rotation": True, "intensity": True}),
                      epochs=r.number("epochs", int, 1, 200, 20),
                      patience=r.number("patience", int, 1, 50, 5),
                      batch_size=r.number("batch_size", int, 1, 256, 32))
    else:
        from Helpers.image3d.volumes import CROPS, SHAPES, SINGLE
        names = [s["name"] for s in volume3d["series"]]
        if volume3d["single_series"]:
            channels = [SINGLE]
        elif r.has("channels"):
            channels = list(dict.fromkeys(config["channels"]))
            unknown = [c for c in channels if c not in names]
            if unknown or not channels:
                raise ConfigError(f"channels must be series of the studies: {', '.join(names)}.")
        else:
            channels = volume3d["default_channels"]
        shape = config.get("shape") if config else None
        if shape is not None and tuple(shape) not in [tuple(s) for s in SHAPES]:
            raise ConfigError(f"shape must be one of {', '.join('x'.join(map(str, s)) for s in SHAPES)}.")
        max_folds = max(2, min(20, volume3d["min_class_patients"]))
        params = dict(common,
                      k_folds=r.number("k_folds", int, 2, max_folds, min(5, max_folds)),
                      channels=channels,
                      shape=list(shape) if shape is not None else list(SHAPES[0]),
                      crop=r.choice("crop", list(CROPS), 1.0),
                      augmentation=r.flags("augmentation", ["horizontal_flip", "vertical_flip", "rotation", "intensity"],
                                           {"horizontal_flip": False, "vertical_flip": False, "rotation": True, "intensity": True}),
                      epochs=r.number("epochs", int, 1, 200, 30),
                      patience=r.number("patience", int, 1, 50, 8),
                      batch_size=r.number("batch_size", int, 1, 64, 4))
    params["dim"] = dim
    return params, _names(available, params["models"])


def _detection(config, summary):
    r = _Reader(config, contracts.config_schema("object-detection"))
    available = contracts.models("object-detection")
    max_folds = max(2, summary["max_folds"])
    params = {
        "models": r.models(available),
        "validation": r.choice("validation", ["kfold", "holdout"], "kfold"),
        "k_folds": r.number("k_folds", int, 2, max_folds, min(5, max_folds)),
        "holdout_fraction": r.number("holdout_fraction", float, 0.1, 0.4, 0.2),
        "epochs": r.number("epochs", int, 1, 300, 30),
        "patience": r.number("patience", int, 1, 50, 5),
        "batch_size": r.number("batch_size", int, 1, 64, 4),
        "image_size": r.choice("image_size", [320, 512, 640, 800, 1024], 640),
        "lr_scale": r.number("lr_scale", float, 0.1, 10, 1.0),
        "augmentation": r.flags("augmentation", ["horizontal_flip", "vertical_flip", "intensity"],
                                {"horizontal_flip": True, "vertical_flip": False, "intensity": True}),
        "window": r.choice("window", contracts.CT_WINDOWS, "auto"),
        "negative_ratio": r.number("negative_ratio", float, 0, 5, 1.0),
        "drise_images": r.number("drise_images", int, 0, 12, 4),
        "drise_masks": r.number("drise_masks", int, 50, 2000, 300),
        "pretrained": _pretrained(),
    }
    return params, _names(available, params["models"])


def _segmentation(config, summary):
    r = _Reader(config, contracts.config_schema("image-segmentation"))
    dim = summary["dim"]
    available = contracts.models("image-segmentation", dim)
    gpu = _gpu()
    channels = None
    series = [s["name"] for s in summary.get("series", [])]
    if len(series) > 1:
        channels = list(dict.fromkeys(config["channels"])) if r.has("channels") else (summary.get("channels") or series)
        unknown = [c for c in channels if c not in series]
        if unknown or not channels:
            raise ConfigError(f"channels must be series of the cases: {', '.join(series)}.")
    elif summary.get("channels"):
        channels = summary["channels"]
    max_folds = max(2, summary["max_folds"])
    params = {
        "models": r.models(available),
        "dim": dim,
        "mapping": summary["mapping"],
        "channels": channels,
        "validation": r.choice("validation", ["kfold", "holdout"], "kfold" if gpu else "holdout"),
        "k_folds": r.number("k_folds", int, 2, max_folds, min(5, max_folds)),
        "holdout_fraction": r.number("holdout_fraction", float, 0.1, 0.4, 0.2),
        "epochs": r.number("epochs", int, 1, 1000, 100 if gpu else 20),
        "iterations": r.number("iterations", int, 1, 1000, 250 if gpu else 50),
        "batch_size": r.number("batch_size", int, 1, 64, 2 if dim == 3 else 8),
        "learning_rate": r.number("learning_rate", float, 1e-5, 1e-2, 1e-3),
        "augmentation": r.flags("augmentation", ["rotation", "intensity", "horizontal_flip", "vertical_flip", "depth_flip"],
                                {"rotation": True, "intensity": True, "horizontal_flip": bool(summary.get("rgb")),
                                 "vertical_flip": False, "depth_flip": False}),
        "normalisation": r.choice("normalisation", ["auto", "ct", "zscore"], "auto"),
        "tta": str((config or {}).get("tta", True)).lower() not in ("false", "0", "no"),
        "nnunet_epochs": r.number("nnunet_epochs", int, 1, 1000, 100 if gpu else 10),
        "nnunet_iterations": r.number("nnunet_iterations", int, 1, 250, 250 if gpu else 50),
        "pretrained": _pretrained(),
    }
    return params, _names(available, params["models"])


def _forecasting(config, summary):
    from Helpers.forecasting import data as forecast_data
    r = _Reader(config, contracts.config_schema("time-series-forecasting"))
    available = contracts.models("time-series-forecasting")
    horizon = r.number("horizon", int, 1, max(1, summary["max_horizon"]), summary["suggested_horizon"])
    folds_limit = min(10, forecast_data.max_folds(summary["length_min"], horizon))
    if folds_limit < 1:
        raise ConfigError(f"The shortest training series ({summary['length_min']} points) is too short for a horizon of "
                          f"{horizon}: it needs at least two horizons of points. Choose a shorter horizon.")
    future = list((config or {}).get("future_columns") or [])
    unknown = [c for c in future if c not in summary["dynamic"]]
    if unknown:
        raise ConfigError(f"future_columns must be varying covariates of the data: {', '.join(summary['dynamic']) or 'none'}.")
    params = {
        "models": r.models(available),
        "horizon": horizon,
        "k_folds": r.number("k_folds", int, 1, folds_limit, min(3, folds_limit)),
        "lookback": r.number("lookback", int, 0, 10 * summary["length_max"], 0),
        "trials": r.number("trials", int, 0, 50, 0),
        "max_steps": r.number("max_steps", int, 50, 10000, 500),
        "season": r.number("season", int, 1, 1000, summary["season"]),
        "future_columns": future,
    }
    return params, list(params["models"])


BUILDERS = {"tabular": _tabular, "image-classification": _image, "object-detection": _detection,
            "image-segmentation": _segmentation, "time-series-forecasting": _forecasting}


def build(automator, config, summary):
    """(pipeline parameters, display names of the models) of a configuration."""
    if not isinstance(config or {}, dict):
        raise ConfigError("The configuration is an object of fields (see the data contract).")
    return BUILDERS[automator](config or {}, summary)


def defaults(automator, summary):
    """The configuration used when an agent gives none, in the agent's vocabulary."""
    params, names = build(automator, {}, summary)
    if automator == "tabular":
        search = params["apply_grid_search"]
        return {"models": [k for k, n, _, _ in contracts.TABULAR_MODELS if params["Machine Learning Models"][n]],
                "k_folds": params["number_of_k_folds"], "threshold_metric": params["Metric For Threshold Optimization"],
                "hyperparameter_search": "none" if not search["enabled"] else
                ("randomized" if search["type"]["Randomized"] else "exhaustive"),
                "correlation_limit": params["Correlation Limit"], "bias_feature": None}
    hidden = {"classes", "mapping", "pretrained"}
    out = {k: v for k, v in params.items() if k not in hidden}
    if automator == "image-classification":
        out["threshold_metric"] = out.pop("metric")
    return out
