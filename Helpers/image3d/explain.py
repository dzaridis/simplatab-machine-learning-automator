"""3D Grad-CAM: the regions of a volume that drive a prediction.

The class score is differentiated with respect to the feature map of the penultimate stage of
the encoder (B, K, d, h, w): the last stage of a 3D network is too coarse to localise anything
in a volume of a few dozen slices (e.g. 2 x 4 x 4). The channels are weighted by their mean
gradient, summed, and the map is resized to the volume. The figures show, for test studies of each class, the slices where the
map is strongest, over the first series (the reference of the alignment).
"""
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from Helpers.image.explain import select_examples
from .training import device

SLICES_SHOWN = 5


def grad_cam(model, x, target):
    """(D, H, W) heatmap in [0, 1] for the class ``target`` of the (1, C, D, H, W) volume x."""
    model.eval()
    with torch.enable_grad():
        x = x.to(device())
        feature_map = model.early(x)
        feature_map.retain_grad()
        logits = model.from_early(feature_map)
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()
    activations, gradients = feature_map.detach(), feature_map.grad
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam.float(), size=tuple(x.shape[2:]), mode="trilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max()).cpu().numpy() if cam.max() > 0 else cam.cpu().numpy()


def strongest_slices(cam, count=SLICES_SHOWN):
    """The slices where the heatmap is strongest, in anatomical order (evenly spaced slices when
    the heatmap is flat)."""
    count = min(count, cam.shape[0])
    energy = cam.reshape(cam.shape[0], -1).sum(axis=1)
    if np.ptp(energy) <= 1e-6 * max(energy.max(), 1e-12):
        return sorted({int(round(z)) for z in np.linspace(0, cam.shape[0] - 1, count + 2)[1:-1]})
    return sorted(np.argsort(-energy, kind="stable")[:count].tolist())


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "class"


def save_gradcam_figures(model, paths, labels, probabilities, classes, folder, model_name, predictions,
                         channel_name="", per_class=3):
    """One figure per class: for each example study, the slices where Grad-CAM is strongest, over
    the first series."""
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    saved = []
    for c, indices in select_examples(np.asarray(labels), probabilities, predictions, per_class).items():
        if not indices:
            continue
        fig, axes = plt.subplots(len(indices), SLICES_SHOWN, figsize=(2.1 * SLICES_SHOWN, 2.5 * len(indices)),
                                 squeeze=False)
        for row, i in enumerate(indices):
            volume = np.load(paths[i]).astype(np.float32)
            cam = grad_cam(model, torch.from_numpy(volume)[None], int(predictions[i]))
            correct = predictions[i] == labels[i]
            slices = strongest_slices(cam)
            for col in range(SLICES_SHOWN):
                ax = axes[row, col]
                ax.axis("off")
                if col >= len(slices):
                    continue
                z = slices[col]
                ax.imshow(volume[0, z], cmap="gray", vmin=0, vmax=1)
                # weak values transparent: the anatomy stays visible outside the highlighted regions
                ax.imshow(np.ma.masked_less(cam[z], 0.2), cmap="jet", alpha=0.45, vmin=0, vmax=1)
                if col:
                    ax.set_title(f"slice {z + 1}/{cam.shape[0]}", fontsize=8)
            axes[row, 0].set_title(f"True: {classes[labels[i]]} · predicted: {classes[predictions[i]]} "
                                   f"({probabilities[i, predictions[i]]:.2f}){'' if correct else ' ✗'}\n"
                                   f"slice {slices[0] + 1}/{cam.shape[0]}", fontsize=8, loc="left",
                                   color="black" if correct else "#b3261e")
        series = f" (over {channel_name})" if channel_name else ""
        fig.suptitle(f"{model_name}: class {classes[c]}{series}", fontsize=11)
        fig.tight_layout()
        path = os.path.join(folder, f"{_safe(model_name)}_gradcam_{_safe(classes[c])}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        saved.append(path)
    return saved
