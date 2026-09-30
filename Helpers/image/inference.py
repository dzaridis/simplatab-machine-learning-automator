"""Exported image classifiers: saving, loading and predicting on new images.

Example:
    from Helpers.image.inference import load_model, predict
    model, info = load_model("Materials/Models/DINOv2-Small.pt")
    predict(model, info, ["scan_001.dcm", "scan_002.png"])
The images are preprocessed exactly as during training (DICOM/NIfTI windowing, volume
reduction, square padding, resizing and normalisation).
"""
import numpy as np
import torch

from . import io as mio
from .dataset import CACHE_SIZE
from .models import BY_KEY, INPUT_SIZE, ImageClassifier, normalization
from .training import build_transform

FORMAT = "simplatab-image-classifier"


def export_model(model, path, classes, threshold, mode, preprocessing):
    torch.save({
        "format": FORMAT,
        "version": 1,
        "backbone": model.spec.key,
        "backbone_name": model.spec.name,
        "timm_name": model.spec.timm_name,
        "num_classes": len(classes),
        "classes": list(classes),
        # Binary problems: class 1 is predicted when its probability exceeds the threshold
        "threshold": threshold,
        "mode": mode,
        "preprocessing": dict(preprocessing, cache_size=CACHE_SIZE, input_size=INPUT_SIZE),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }, path)


def load_model(path, map_location="cpu"):
    """The model (in evaluation mode) and its metadata. No pretrained weights are downloaded."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Simplatab image classifier")
    model = ImageClassifier(BY_KEY[checkpoint["backbone"]], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    info = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return model.eval(), info


@torch.inference_mode()
def predict(model, info, paths):
    """[{"file", "predicted_class", "probabilities": {class: p}}] for each image file."""
    prep = info["preprocessing"]
    mean, std = normalization(model.backbone)
    transform = build_transform(mean, std)
    rows = []
    for path in paths:
        image = mio.letterbox(mio.load_image(path, prep.get("window", "auto"), prep.get("volume", "middle")),
                              prep.get("cache_size", CACHE_SIZE))
        x = transform(image.convert("RGB")).unsqueeze(0)
        probabilities = torch.softmax(model(x), dim=1)[0].numpy()
        if len(info["classes"]) == 2 and info.get("threshold") is not None:
            predicted = int(probabilities[1] > info["threshold"])
        else:
            predicted = int(np.argmax(probabilities))
        rows.append({"file": path, "predicted_class": info["classes"][predicted],
                     "probabilities": {c: float(p) for c, p in zip(info["classes"], probabilities)}})
    return rows
