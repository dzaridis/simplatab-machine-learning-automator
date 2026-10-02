"""Training and inference of the segmentation networks (nnU-Net aside), with an nnU-Net-like recipe:
- patches of a fixed size, a third of them centred on a foreground voxel (rare structures are seen);
- augmentation: rotations and scaling, optional flips, brightness, contrast, gamma and noise;
- Dice + cross-entropy loss, AdamW with warm-up and cosine decay, gradient clipping, mixed
  precision on GPU, a fixed number of epochs of a fixed number of iterations;
- inference by sliding windows (half overlap, Gaussian weighting) with test-time flips; the
  normalised entropy of the averaged probabilities is the uncertainty map.
"""
import math
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from Helpers.image.training import describe_device, device  # noqa: F401
from .models import build, logits

FOREGROUND_FRACTION = 0.33


class PreparedCase:
    """A normalised case on the grid of the networks: image (C, z, y, x), labels (z, y, x)."""

    def __init__(self, image, labels):
        self.image, self.labels = image, labels
        foreground = np.argwhere(labels > 0)
        if len(foreground) > 20000:
            foreground = foreground[np.random.default_rng(0).choice(len(foreground), 20000, replace=False)]
        self.foreground = foreground


def _crop(array, start, size, value=0):
    """Crop (..., spatial) starting at start, padded where it leaves the array."""
    spatial = array.shape[-len(size):]
    pads, slices = [], []
    for s, n, length in zip(start, size, spatial):
        lo, hi = max(0, s), min(length, s + n)
        slices.append(slice(lo, hi))
        pads.append((lo - s, s + n - hi))
    out = array[(Ellipsis,) + tuple(slices)]
    if any(p for pair in pads for p in pair):
        out = np.pad(out, [(0, 0)] * (array.ndim - len(size)) + pads, constant_values=value)
    return out


def sample_patch(case, patch, rng, dim):
    """(image patch (C, *patch), label patch (*patch)), centred on foreground a third of the time."""
    image, labels = case.image, case.labels
    if dim == 2:
        image, labels = image[:, 0], labels[0]
        foreground = case.foreground[:, 1:] if len(case.foreground) else case.foreground
    else:
        foreground = case.foreground
    spatial = labels.shape
    if len(foreground) and rng.random() < FOREGROUND_FRACTION:
        center = foreground[rng.integers(len(foreground))]
        start = [int(c) - p // 2 + int(rng.integers(-p // 4, p // 4 + 1)) for c, p in zip(center, patch)]
    else:
        start = [int(rng.integers(min(0, n - p), max(0, n - p) + 1)) for n, p in zip(spatial, patch)]
    return _crop(image, start, patch), _crop(labels, start, patch)


def augment(image, labels, settings, generator):
    """Joint augmentation of a batch: image (B, C, *S) float, labels (B, *S) long."""
    settings = settings or {}
    dims = labels.dim() - 1
    b = image.shape[0]
    if settings.get("rotation", True):
        angles = (torch.rand(b, generator=generator) * 2 - 1) * math.pi / 12          # ±15°, in-plane
        scales = 1 + (torch.rand(b, generator=generator) * 2 - 1) * 0.15              # ±15%
        cos, sin = torch.cos(angles) / scales, torch.sin(angles) / scales
        if dims == 2:
            theta = torch.stack([torch.stack([cos, -sin, torch.zeros(b)], 1), torch.stack([sin, cos, torch.zeros(b)], 1)], 1)
        else:
            zeros, ones = torch.zeros(b), torch.ones(b) / scales
            theta = torch.stack([torch.stack([cos, -sin, zeros, zeros], 1), torch.stack([sin, cos, zeros, zeros], 1),
                                 torch.stack([zeros, zeros, ones, zeros], 1)], 1)
        grid = F.affine_grid(theta, list(image.shape), align_corners=False)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        labels = F.grid_sample(labels[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                               align_corners=False)[:, 0].long()
    flips = [settings.get("horizontal_flip"), settings.get("vertical_flip")] + ([settings.get("depth_flip")] if dims == 3 else [])
    for axis, enabled in zip(range(-1, -dims - 1, -1), flips):
        if enabled:
            chosen = torch.rand(b, generator=generator) < 0.5
            image[chosen] = image[chosen].flip(axis)
            labels[chosen] = labels[chosen].flip(axis)
    if settings.get("intensity", True):
        shape = (b, 1) + (1,) * dims
        image = image * (1 + (torch.rand(shape, generator=generator) * 2 - 1) * 0.25)          # brightness
        mean = image.mean(dim=tuple(range(2, image.dim())), keepdim=True)
        image = (image - mean) * (1 + (torch.rand(shape, generator=generator) * 2 - 1) * 0.25) + mean  # contrast
        image = image + torch.randn(image.shape, generator=generator) * torch.rand(shape, generator=generator) * 0.1
    return image, labels


def _batch(cases, patch, batch_size, rng, dim):
    pairs = [sample_patch(cases[rng.integers(len(cases))], patch, rng, dim) for _ in range(batch_size)]
    image = torch.from_numpy(np.stack([p[0] for p in pairs]).astype(np.float32))
    labels = torch.from_numpy(np.stack([p[1] for p in pairs]).astype(np.int64))
    return image, labels


def _autocast():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def train(spec, cases, channels, num_classes, patch, settings, log=print, label="", pretrained=True, rgb=False, seed=0):
    """Trains a network on prepared cases for settings["epochs"] x settings["iterations"] steps."""
    from monai.losses import DiceCELoss
    torch.manual_seed(seed)
    rng, generator = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    model = build(spec, channels, num_classes, patch, pretrained=pretrained, rgb=rgb).to(device())
    dim = len(patch)
    epochs, iterations = int(settings["epochs"]), int(settings["iterations"])
    rate = float(settings["learning_rate"]) * (0.3 if spec.family in ("pretrained", "ssl") else 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-4)
    steps = max(1, epochs * iterations)
    warmup = min(iterations, steps // 10 + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)))
    criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    for epoch in range(1, epochs + 1):
        model.train()
        total, started = 0.0, time.time()
        for _ in range(iterations):
            image, labels = _batch(cases, patch, int(settings["batch_size"]), rng, dim)
            image, labels = augment(image, labels, settings.get("augmentation"), generator)
            image, labels = image.to(device(), non_blocking=True), labels.to(device(), non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                loss = criterion(logits(model(image)).float(), labels[:, None])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        log(f"{label} epoch {epoch}/{epochs}: loss {total / iterations:.4f} ({time.time() - started:.0f} s)")
    return model.eval()


# ---------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------

def flips(dim, tta=True):
    if not tta:
        return [()]
    return [(), (-1,), (-2,)] + ([(-3,)] if dim == 3 else [(-1, -2)])


@torch.inference_mode()
def predict(model, image, patch, tta=True, batch_size=4):
    """Probabilities (K + 1, z, y, x) and uncertainty (z, y, x, in [0, 1]) of a normalised image
    (C, z, y, x) on the grid of the network."""
    from monai.inferers import sliding_window_inference
    dim = len(patch)
    x = torch.from_numpy(image[:, 0] if dim == 2 else image)[None].to(device())
    model.eval().to(device())

    def network(window):
        with _autocast():
            return torch.softmax(logits(model(window)).float(), dim=1)
    total = None
    passes = flips(dim, tta)
    for axes in passes:
        inputs = x.flip(axes) if axes else x
        with warnings.catch_warnings():  # MONAI indexes with lists, deprecated by PyTorch
            warnings.simplefilter("ignore", UserWarning)
            probabilities = sliding_window_inference(inputs, tuple(patch), batch_size, network, overlap=0.5,
                                                     mode="gaussian")
        probabilities = probabilities.flip(axes) if axes else probabilities
        total = probabilities if total is None else total + probabilities
    mean = (total / len(passes))[0].float()
    entropy = -(mean * torch.log(mean.clamp_min(1e-8))).sum(0) / math.log(mean.shape[0])
    mean, entropy = mean.cpu().numpy(), entropy.cpu().numpy()
    if dim == 2:
        mean, entropy = mean[:, None], entropy[None]
    return mean, entropy


def to_original(probabilities, uncertainty, shape):
    """Probabilities and uncertainty on the cached grid of a case: (labels, uncertainty)."""
    if list(probabilities.shape[1:]) != list(shape):
        p = torch.from_numpy(probabilities)[None]
        probabilities = F.interpolate(p, size=tuple(shape), mode="trilinear", align_corners=False)[0].numpy()
        u = torch.from_numpy(uncertainty)[None, None]
        uncertainty = F.interpolate(u, size=tuple(shape), mode="trilinear", align_corners=False)[0, 0].numpy()
    return probabilities.argmax(0).astype(np.uint8), uncertainty.astype(np.float32)
