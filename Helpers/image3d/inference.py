"""Exported 3D classifiers: saving, loading and predicting on new studies.

Each network is saved as a TorchScript file that runs with PyTorch only: it takes a
(1, C, D, H, W) volume in [0, 1] (the series of a study, aligned and resized as in training)
and returns the class probabilities. Its metadata (classes, threshold, series, volume shape,
crop and window) is stored in the file as ``simplatab.json``.

Within Simplatab:
    from Helpers.image3d.inference import load_model, predict
    model, info = load_model("Materials/Models/MedicalNet_ResNet-10.pt")
    predict(model, info, [{"t2": "case_01/t2", "adc": "case_01/adc.nii.gz"}])
"""
import copy
import json
import warnings

import numpy as np
import torch
from torch import nn

from Helpers.image import io as mio
from . import volumes

FORMAT = "simplatab-volume-classifier"
METADATA = "simplatab.json"


class _Standalone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def export_model(model, path, classes, threshold, mode, preprocessing):
    window = preprocessing["window"]
    info = {
        "format": FORMAT,
        "version": 1,
        "network": model.spec.name,
        "classes": list(classes),
        "threshold": threshold,
        "mode": mode,
        "channels": list(preprocessing["channels"]),
        "shape": list(preprocessing["shape"]),
        "crop": preprocessing["crop"],
        "window": window,
        "ct_window": list(mio.CT_WINDOWS[window]) if window in mio.CT_WINDOWS else None,
        "kinds": preprocessing.get("kinds", {}),
    }
    standalone = _Standalone(copy.deepcopy(model)).cpu().eval()
    example = torch.rand(1, len(info["channels"]), *info["shape"])
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # shape checks are traced as constants: the input shape is fixed
        traced = torch.jit.trace(standalone, example, check_trace=False)
    torch.jit.save(traced, path, _extra_files={METADATA: json.dumps(info)})


def load_model(path, map_location="cpu"):
    meta = {METADATA: ""}
    model = torch.jit.load(path, map_location=map_location, _extra_files=meta)
    info = json.loads(meta[METADATA])
    if info.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Simplatab 3D classifier")
    return model.eval(), info


def _series(path):
    """A series description from a path: a DICOM folder (its first series) or a file."""
    import os
    if os.path.isdir(path):
        import SimpleITK as sitk
        uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
        if not uids:
            raise ValueError(f"no DICOM series in {path}")
        return {"kind": "dicom", "path": path, "uid": uids[0], "name": os.path.basename(path)}
    return {"kind": "file", "path": path, "name": os.path.basename(path),
            "modality": "NIFTI" if path.lower().endswith((".nii", ".nii.gz")) else ""}


@torch.inference_mode()
def predict(model, info, studies):
    """For each study ({series name: DICOM folder or file}): the predicted class and probabilities."""
    rows = []
    for study in studies:
        series = [_series(study[name]) for name in info["channels"]] if info["channels"] != [volumes.SINGLE] \
            else [_series(next(iter(study.values())))]
        array, _ = volumes.load_study(series, info["shape"], info["crop"], info["window"])
        probabilities = model(torch.from_numpy(array)[None])[0].numpy()
        if len(info["classes"]) == 2 and info.get("threshold") is not None:
            predicted = int(probabilities[1] > info["threshold"])
        else:
            predicted = int(np.argmax(probabilities))
        rows.append({"study": study, "predicted_class": info["classes"][predicted],
                     "probabilities": {c: float(p) for c, p in zip(info["classes"], probabilities)}})
    return rows
