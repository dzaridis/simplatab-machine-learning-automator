"""Grad-CAM: the image regions that drive a prediction.

The class score is differentiated with respect to the last spatial feature map of the
network; the map channels are weighted by their mean gradient and summed. For vision
transformers the map is the patch tokens entering the last block (the class token alone
carries the prediction afterwards).
"""
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from .models import INPUT_SIZE, normalization
from .training import build_transform, device


def _to_nchw(tensor, layout, backbone):
    if layout == "nhwc":
        return tensor.permute(0, 3, 1, 2)
    if layout == "tokens":
        tokens = tensor[:, getattr(backbone, "num_prefix_tokens", 1):, :]
        height, width = backbone.patch_embed.grid_size
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)
    return tensor


def grad_cam(model, x, target):
    """Heatmap in [0, 1] of size INPUT_SIZE x INPUT_SIZE for the class ``target``."""
    model.eval()
    backbone, layout = model.backbone, model.spec.layout
    captured = {}
    with torch.enable_grad():
        x = x.to(device())
        if layout == "tokens":
            def keep(module, inputs, output):
                output.retain_grad()
                captured["map"] = output
            handle = backbone.blocks[-1].norm1.register_forward_hook(keep)
            try:
                logits = model(x)
            finally:
                handle.remove()
        else:
            feature_map = backbone.forward_features(x)
            feature_map.retain_grad()
            captured["map"] = feature_map
            logits = model.head(backbone.forward_head(feature_map, pre_logits=True))
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()
    activations = _to_nchw(captured["map"].detach(), layout, backbone)
    gradients = _to_nchw(captured["map"].grad, layout, backbone)
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max()).cpu().numpy() if cam.max() > 0 else cam.cpu().numpy()


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "class"


def select_examples(labels, probabilities, predictions, per_class=4):
    """For each class: the most confident errors first (the most useful to review), then the
    most confident correct predictions."""
    chosen = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        wrong = [i for i in members if predictions[i] != c]
        right = [i for i in members if predictions[i] == c]
        wrong.sort(key=lambda i: -probabilities[i, predictions[i]])
        right.sort(key=lambda i: -probabilities[i, c])
        chosen[int(c)] = (wrong[: per_class // 2] + right)[:per_class] if wrong else right[:per_class]
    return chosen


def save_gradcam_figures(model, paths, labels, probabilities, classes, folder, model_name, per_class=4,
                         predictions=None):
    """One figure per class: each example image next to its Grad-CAM overlay for the
    predicted class."""
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(folder, exist_ok=True)
    mean, std = normalization(model.backbone)
    transform = build_transform(mean, std)
    predictions = probabilities.argmax(axis=1) if predictions is None else predictions
    saved = []
    for c, indices in select_examples(np.asarray(labels), probabilities, predictions, per_class).items():
        if not indices:
            continue
        fig, axes = plt.subplots(len(indices), 2, figsize=(6, 3 * len(indices)), squeeze=False)
        for row, i in enumerate(indices):
            with Image.open(paths[i]) as image:
                rgb = image.convert("RGB")
                shown = np.asarray(rgb.resize((INPUT_SIZE, INPUT_SIZE)))
                x = transform(rgb).unsqueeze(0)
            cam = grad_cam(model, x, int(predictions[i]))
            correct = predictions[i] == labels[i]
            title = (f"True: {classes[labels[i]]}\nPredicted: {classes[predictions[i]]} "
                     f"({probabilities[i, predictions[i]]:.2f}){'' if correct else ' ✗'}")
            axes[row, 0].imshow(shown)
            axes[row, 0].set_title(title, fontsize=9, color="black" if correct else "#b3261e")
            axes[row, 1].imshow(shown)
            axes[row, 1].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            axes[row, 1].set_title("Grad-CAM", fontsize=9)
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle(f"{model_name}: class {classes[c]}", fontsize=11)
        fig.tight_layout()
        path = os.path.join(folder, f"{_safe(model_name)}_gradcam_{_safe(classes[c])}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        saved.append(path)
    return saved
