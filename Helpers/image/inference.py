"""Exported image classifiers: saving, loading and predicting on new images.

Each network is saved as a TorchScript file that runs without the Simplatab code (only PyTorch):
it takes RGB images of 224 x 224 pixels with values in [0, 1], normalises them and returns the
class probabilities. Its metadata (classes, decision threshold, image preprocessing) is stored in
the file as ``simplatab.json``:
    meta = {"simplatab.json": ""}
    model = torch.jit.load("DINOv2-Small.pt", _extra_files=meta)
    info = json.loads(meta["simplatab.json"])

Within Simplatab:
    from Helpers.image.inference import load_model, predict
    model, info = load_model("Materials/Models/DINOv2-Small.pt")
    predict(model, info, ["scan_001.dcm", "scan_002.png"])
"""
import copy
import json
import warnings

import numpy as np
import torch
from torch import nn

from . import io as mio
from .dataset import CACHE_SIZE
from .models import INPUT_SIZE, normalization

FORMAT = "simplatab-image-classifier"
METADATA = "simplatab.json"


class _Standalone(nn.Module):
    """Normalisation + network + softmax, for images in [0, 1]."""

    def __init__(self, model):
        super().__init__()
        mean, std = normalization(model.backbone)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model((x - self.mean) / self.std), dim=1)


def export_model(model, path, classes, threshold, mode, preprocessing):
    window = preprocessing.get("window", "auto")
    info = {
        "format": FORMAT,
        "version": 2,
        "network": model.spec.name,
        "timm_name": model.spec.timm_name,
        "classes": list(classes),
        # Binary problems: class 1 is predicted when its probability exceeds the threshold
        "threshold": threshold,
        "mode": mode,
        "window": window,
        # CT window (center, width) chosen for the run, for CT DICOM and NIfTI files
        "ct_window": list(mio.CT_WINDOWS[window]) if window in mio.CT_WINDOWS else None,
        "volume": preprocessing.get("volume", "middle"),
        "formats": preprocessing.get("formats", {}),
        "cache_size": CACHE_SIZE,
        "input_size": INPUT_SIZE,
    }
    standalone = _Standalone(copy.deepcopy(model)).cpu().eval()  # the pipeline keeps using the model
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # shape checks are traced as constants: the input size is fixed
        traced = torch.jit.trace(standalone, torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE))
    torch.jit.save(traced, path, _extra_files={METADATA: json.dumps(info)})


def load_model(path, map_location="cpu"):
    """The TorchScript model and its metadata."""
    meta = {METADATA: ""}
    model = torch.jit.load(path, map_location=map_location, _extra_files=meta)
    info = json.loads(meta[METADATA])
    if info.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Simplatab image classifier")
    return model.eval(), info


def prepare(image, info):
    """8-bit image (as read by Helpers.image.io) -> (1, 3, 224, 224) tensor in [0, 1], exactly as
    during training: square padding, resizing to the cache size, then to the input size."""
    from PIL import Image
    square = mio.letterbox(image, info.get("cache_size", CACHE_SIZE)).convert("RGB")
    size = info.get("input_size", INPUT_SIZE)
    square = square.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(square, dtype=np.float32) / 255).permute(2, 0, 1)[None]


@torch.inference_mode()
def predict(model, info, paths):
    """[{"file", "predicted_class", "probabilities": {class: p}}] for each image file."""
    rows = []
    for path in paths:
        image = mio.load_image(path, info.get("window", "auto"), info.get("volume", "middle"))
        probabilities = model(prepare(image, info))[0].numpy()
        if len(info["classes"]) == 2 and info.get("threshold") is not None:
            predicted = int(probabilities[1] > info["threshold"])
        else:
            predicted = int(np.argmax(probabilities))
        rows.append({"file": path, "predicted_class": info["classes"][predicted],
                     "probabilities": {c: float(p) for c, p in zip(info["classes"], probabilities)}})
    return rows
