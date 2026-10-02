"""Exported segmentation networks (nnU-Net aside, exported as its own model folder).

Each network is saved as a TorchScript file mapping a normalised patch (1, C, [D,] H, W) to class
logits, with its preprocessing in ``simplatab.json``: channels, normalisation (and the CT
statistics), target spacing (3D), patch size, classes and the mask value of each class. The code
of the results page reads an image, normalises and resamples it, predicts by sliding windows with
test-time flips and writes the mask.
"""
import copy
import json
import warnings

import torch
from torch import nn

from .models import logits

FORMAT = "simplatab-segmenter"
METADATA = "simplatab.json"


class _Logits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return logits(self.model(x))


def export_model(model, path, info):
    info = dict(info, format=FORMAT, version=1)
    standalone = _Logits(copy.deepcopy(model)).cpu().eval()
    example = torch.rand(1, info["channels"], *info["patch"])
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # shape checks are traced as constants: the patch size is fixed
        traced = torch.jit.trace(standalone, example, check_trace=False)
    torch.jit.save(traced, path, _extra_files={METADATA: json.dumps(info)})


def load_model(path, map_location="cpu"):
    meta = {METADATA: ""}
    model = torch.jit.load(path, map_location=map_location, _extra_files=meta)
    info = json.loads(meta[METADATA])
    if info.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Simplatab segmentation network")
    return model.eval(), info
