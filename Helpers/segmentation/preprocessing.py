"""Preprocessing plan of the networks trained by the automator (nnU-Net plans its own), in the
spirit of nnU-Net:
- target spacing (3D): the median spacing of the training cases; for anisotropic data (the
  coarsest axis more than 3 times the finest) the coarse axis uses its 10th percentile;
- normalisation of each channel: CT (clipped to the 0.5-99.5 percentiles of the foreground
  intensities of the training cases, then standardised with their mean and standard deviation),
  RGB (scaled to [0, 1]; the pretrained encoders then apply the ImageNet statistics) or z-score
  (per image);
- patch size: the median shape of the resampled cases, rounded to the multiples the networks
  need and reduced to a voxel budget.
"""
import numpy as np
import torch
import torch.nn.functional as F

PATCH_BUDGET = {2: 512 * 512, 3: 128 * 128 * 96}


def target_spacing(metas):
    spacings = np.array([m["spacing"] for m in metas], dtype=np.float64)
    target = np.median(spacings, axis=0)
    coarse = int(np.argmax(target))
    if target[coarse] > 3 * np.min(target):
        target[coarse] = np.percentile(spacings[:, coarse], 10)
    return [float(s) for s in target]


def normalisation_modes(metas, channels, rule="auto"):
    """'ct', 'rgb' or 'zscore' for each channel."""
    first = metas[0]
    modes = []
    for c in range(channels):
        modality = (first["modality"][c] if c < len(first["modality"]) else "") or ""
        if first.get("rgb"):
            modes.append("rgb")
        elif rule == "ct" or (rule == "auto" and modality == "CT"):
            modes.append("ct")
        else:
            modes.append("zscore")
    return modes


def foreground_statistics(samples):
    """samples: per channel, arrays of foreground intensities -> CT clipping and standardisation."""
    stats = []
    for values in samples:
        values = np.concatenate(values) if values else np.zeros(1)
        low, high = np.percentile(values, [0.5, 99.5])
        clipped = np.clip(values, low, high)
        stats.append({"low": float(low), "high": float(high), "mean": float(clipped.mean()),
                      "std": float(max(clipped.std(), 1e-6))})
    return stats


def normalise(image, modes, stats):
    """image (C, z, y, x) float32 -> normalised copy."""
    out = np.empty_like(image, dtype=np.float32)
    for c, mode in enumerate(modes):
        x = image[c]
        if mode == "rgb":
            out[c] = x / 255.0
        elif mode == "ct":
            s = stats[c]
            out[c] = (np.clip(x, s["low"], s["high"]) - s["mean"]) / s["std"]
        else:
            out[c] = (x - x.mean()) / max(float(x.std()), 1e-6)
    return out


def resampled_shape(shape, spacing, target):
    return [max(1, int(round(n * s / t))) for n, s, t in zip(shape, spacing, target)]


def resample(array, shape, labels=False):
    """(C, z, y, x) image (trilinear) or (z, y, x) labels (nearest) to a new shape."""
    if list(array.shape[-3:]) == list(shape):
        return array
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if labels:
        out = F.interpolate(tensor[None, None].float(), size=tuple(shape), mode="nearest")[0, 0]
        return out.numpy().astype(np.uint8)
    return F.interpolate(tensor[None].float(), size=tuple(shape), mode="trilinear", align_corners=False)[0].numpy()


def patch_size(shapes, dim, multiple=32, budget=None):
    """Patch from the median (resampled) shape: each side rounded up to ``multiple`` (at least
    one multiple), then the largest side reduced until the patch fits the voxel budget."""
    budget = budget or PATCH_BUDGET[dim]
    median = np.median(np.array(shapes, dtype=np.float64), axis=0)[-dim:]
    patch = [max(multiple, int(np.ceil(s / multiple)) * multiple) for s in median]
    while np.prod(patch) > budget and max(patch) > multiple:
        i = int(np.argmax(patch))
        patch[i] -= multiple
    return [int(p) for p in patch]


def plan(metas, dim, channels, rule="auto", intensities=None):
    """The preprocessing plan of a run (stored with the exported models)."""
    spacing = target_spacing(metas) if dim == 3 else [1.0, 1.0, 1.0]
    shapes = [resampled_shape(m["shape"], m["spacing"], spacing) if dim == 3 else m["shape"] for m in metas]
    modes = normalisation_modes(metas, channels, rule)
    return {"dim": dim, "spacing": spacing, "modes": modes,
            "stats": foreground_statistics(intensities) if intensities is not None and "ct" in modes else [{}] * channels,
            "median_shape": [int(v) for v in np.median(np.array(shapes), axis=0)]}


def prepare(image, labels, meta, plan_):
    """Cached case -> (normalised image, labels) on the grid of the networks."""
    image = normalise(image, plan_["modes"], plan_["stats"])
    if plan_["dim"] == 3:
        shape = resampled_shape(meta["shape"], meta["spacing"], plan_["spacing"])
        image = resample(image, shape)
        labels = resample(labels, shape, labels=True) if labels is not None else None
    return image.astype(np.float32), labels
