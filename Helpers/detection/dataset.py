"""Datasets of the detection automator: upload summary, cache of 8-bit images and volumes,
validation splits and the PyTorch dataset of training images.

- 2D: every image is a training unit.
- 3D: the networks are 2D (2.5D input: a slice and its neighbours); the training units are the
  slices that contain boxes, plus slices without boxes (negatives) so the networks also learn
  what normal anatomy looks like. Volumes are forecast slice by slice and the boxes of adjacent
  slices merged into 3D boxes (see merge.py).
"""
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from Helpers.image import io as mio
from . import volumes
from .annotations import AnnotationError, load_split

MAX_SIZE_SAMPLES = 100  # headers read for the image size summary


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------
# Upload summary
# ---------------------------------------------------------------------------------------

def summarize(train_folder, test_folder):
    """What the configuration page shows: format, dimension, classes and their boxes, image
    sizes, groups. ``errors`` lists what blocks the run."""
    summary = {"errors": [], "warnings": []}
    try:
        train, test = load_split(train_folder), load_split(test_folder)
    except AnnotationError as e:
        summary["errors"].append(str(e))
        return summary
    summary["warnings"] += [f"Train.zip: {w}" for w in train.warnings] + [f"Test.zip: {w}" for w in test.warnings]
    if train.dim != test.dim:
        summary["errors"].append("Train.zip and Test.zip must both hold 2D images or both 3D volumes.")
        return summary
    unknown = sorted(set(test.classes) - set(train.classes))
    if unknown:
        summary["errors"].append(f"Test.zip has classes that Train.zip has no box of: {', '.join(unknown)}.")
    if not train.classes:
        summary["errors"].append("Train.zip has no boxes.")
    counts = []
    for cls in train.classes:
        counts.append({"class": cls,
                       "train_boxes": sum(l == cls for s in train.samples for l in s.labels),
                       "train_images": sum(cls in s.labels for s in train.samples),
                       "test_boxes": sum(l == cls for s in test.samples for l in s.labels)})
    paths = [s.path for s in train.samples + test.samples]
    kinds = Counter("dicom series" if os.path.isdir(p) else mio.file_kind(p) for p in paths)
    sizes = []
    for sample in (train.samples + test.samples)[:MAX_SIZE_SAMPLES]:
        try:
            sizes.append(volumes.volume_shape(sample.path) if train.dim == 3 else volumes.image_size(sample.path))
        except Exception:
            continue
    groups = len({s.group for s in train.samples})
    summary.update({
        "dim": train.dim,
        "format": train.format,
        "test_format": test.format,
        "classes": train.classes,
        "class_counts": counts,
        "train_images": len(train.samples),
        "test_images": len(test.samples),
        "train_negatives": sum(not s.labels for s in train.samples),
        "test_negatives": sum(not s.labels for s in test.samples),
        "train_boxes": sum(len(s.labels) for s in train.samples),
        "test_boxes": sum(len(s.labels) for s in test.samples),
        "groups": groups,
        "grouped": groups < len(train.samples),
        "kinds": dict(kinds),
        "medical": bool(kinds.get("dicom") or kinds.get("nifti") or kinds.get("dicom series")),
        "has_ct": _has_ct(paths),
        "size_range": _size_range(sizes, train.dim),
        "max_folds": min(10, groups),
    })
    if groups < 2:
        summary["errors"].append("Train.zip needs at least two images (or patients) to validate the networks.")
    return summary


def _has_ct(paths):
    import pydicom
    for path in paths[:MAX_SIZE_SAMPLES]:
        if volumes.is_nifti(path):
            return True  # NIfTI volumes are often CT: the CT windows are offered
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))][:1] if os.path.isdir(path) else [path]
        for f in files:
            if mio.file_kind(f) == "dicom":
                try:
                    if str(pydicom.dcmread(f, stop_before_pixels=True, force=True).get("Modality", "")).upper() == "CT":
                        return True
                except Exception:
                    continue
    return False


def _size_range(sizes, dim):
    if not sizes:
        return None
    if dim == 2:
        sides = [max(s) for s in sizes]
        return {"min": int(min(sides)), "max": int(max(sides))}
    return {"slices_min": int(min(s[0] for s in sizes)), "slices_max": int(max(s[0] for s in sizes)),
            "side_min": int(min(max(s[1:]) for s in sizes)), "side_max": int(max(max(s[1:]) for s in sizes))}


# ---------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------

@dataclass
class Item:
    """A cached image or volume with its boxes (class indices)."""
    name: str
    path: str
    cache: str
    dim: int
    shape: tuple                 # (height, width) or (slices, height, width)
    boxes: np.ndarray
    labels: np.ndarray
    group: str


def _cache_one(args):
    index, path, cache_folder, dim, window = args
    try:
        if dim == 2:
            array = volumes.read_image(path, window)
            target = os.path.join(cache_folder, f"{index:06d}.png")
            Image.fromarray(array).save(target)
            return target, array.shape[:2], None
        volume = volumes.read_volume(path, window)
        target = os.path.join(cache_folder, f"{index:06d}.npy")
        np.save(target, volume)
        return target, volume.shape, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def cache_split(annotated, cache_folder, classes, window="auto", workers=None, log=print):
    """Reads every image (volume) once into 8-bit arrays. Returns the items and the failures."""
    os.makedirs(cache_folder, exist_ok=True)
    index = {c: i for i, c in enumerate(classes)}
    jobs = [(i, s.path, cache_folder, annotated.dim, window) for i, s in enumerate(annotated.samples)]
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    if len(jobs) > 8 and workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_cache_one, jobs, chunksize=4))
    else:
        results = [_cache_one(job) for job in jobs]
    items, failed = [], []
    for sample, (target, shape, error) in zip(annotated.samples, results):
        if error:
            failed.append({"image": sample.name, "error": error})
            continue
        boxes = sample.boxes.copy()
        dim = annotated.dim
        limits = [shape[1], shape[0]] if dim == 2 else [shape[2], shape[1], shape[0]]
        boxes[:, :dim] = np.clip(boxes[:, :dim], 0, limits)
        boxes[:, dim:] = np.clip(boxes[:, dim:], 0, limits)
        keep = np.all(boxes[:, dim:] > boxes[:, :dim] + 1e-3, axis=1)
        labels = np.array([index[label] for label in sample.labels], dtype=np.int64)
        items.append(Item(sample.name, sample.path, target, dim, tuple(shape), boxes[keep], labels[keep], sample.group))
    if failed:
        log(f"{len(failed)} image(s) could not be read and were skipped (e.g. {failed[0]['image']}: {failed[0]['error']})")
    return items, failed


# ---------------------------------------------------------------------------------------
# Training units (images, or slices of volumes)
# ---------------------------------------------------------------------------------------

@dataclass
class Unit:
    item: int                     # index in the items
    z: Optional[int]              # slice of a volume (None for a 2D image)
    boxes: np.ndarray             # (n, 4) on this image or slice
    labels: np.ndarray


def slice_boxes(item, z):
    """2D boxes of the 3D boxes crossing slice z."""
    if not len(item.boxes):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    inside = (item.boxes[:, 2] <= z) & (item.boxes[:, 5] > z)
    return item.boxes[inside][:, [0, 1, 3, 4]].astype(np.float32), item.labels[inside]


def units(items, indices, negative_ratio=1.0, seed=0):
    """Training units of the items: every 2D image; for volumes, the slices with boxes and
    ``negative_ratio`` times as many slices without (at least two per volume)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in indices:
        item = items[i]
        if item.dim == 2:
            out.append(Unit(i, None, item.boxes.astype(np.float32), item.labels))
            continue
        depth = item.shape[0]
        positive = sorted({z for b in item.boxes for z in range(int(np.floor(b[2])), int(np.ceil(b[5]))) if 0 <= z < depth})
        negative = [z for z in range(depth) if z not in set(positive)]
        count = min(len(negative), max(2, int(round(negative_ratio * len(positive)))))
        chosen = sorted(rng.choice(negative, count, replace=False).tolist()) if count else []
        for z in positive + chosen:
            boxes, labels = slice_boxes(item, z)
            out.append(Unit(i, z, boxes, labels))
    return out


def load_unit_image(item, z, volume_cache=None):
    """(H, W, 3) uint8 image of a unit."""
    if item.dim == 2:
        with Image.open(item.cache) as image:
            return np.array(image.convert("RGB"))
    volume = (volume_cache or {}).get(item.cache)
    if volume is None:
        volume = np.load(item.cache, mmap_mode="r")
        if volume_cache is not None:
            volume_cache[item.cache] = volume
    return volumes.slice_25d(volume, z)


# ---------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------

def _strata(items, indices):
    labels = []
    for i in indices:
        counts = Counter(items[i].labels.tolist())
        labels.append(counts.most_common(1)[0][0] if counts else -1)
    return np.array(labels)


def kfold_splits(items, indices, k, seed=0):
    """Grouped (patient) and stratified (main class, or none) K-fold splits of the items."""
    from sklearn.model_selection import StratifiedGroupKFold
    indices = np.asarray(indices)
    groups = np.array([items[i].group for i in indices])
    if len(set(groups)) < k:
        raise ValueError(f"{k} folds need at least {k} images (or patients); there are {len(set(groups))}.")
    strata = _strata(items, indices)
    rare = {s for s, c in Counter(strata).items() if c < k}
    strata = np.array([-2 if s in rare else s for s in strata])  # classes too rare to stratify on
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(indices[a].tolist(), indices[b].tolist()) for a, b in splitter.split(indices, strata, groups)]


def holdout_split(items, indices, fraction=0.2, seed=0):
    """One grouped, stratified train / validation split with about ``fraction`` validation."""
    k = max(2, int(round(1 / max(0.05, min(0.5, fraction)))))
    k = min(k, len({items[i].group for i in indices}))
    return kfold_splits(items, indices, k, seed)[0]


# ---------------------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------------------

@dataclass
class Augmentation:
    horizontal_flip: bool = False
    vertical_flip: bool = False
    intensity: bool = True


class DetectionDataset(torch.utils.data.Dataset):
    """Units resized to size x size (boxes scaled), as float tensors in [0, 1]."""

    def __init__(self, items, units_, size, augmentation=None, seed=0):
        self.items, self.units, self.size = items, units_, size
        self.augmentation = augmentation
        self.rng = np.random.default_rng(seed)
        self.volume_cache = {}

    def __len__(self):
        return len(self.units)

    def __getitem__(self, index):
        unit = self.units[index]
        image = load_unit_image(self.items[unit.item], unit.z, self.volume_cache)
        height, width = image.shape[:2]
        resized = np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize((self.size, self.size), Image.BILINEAR),
                             dtype=np.float32) / 255
        boxes = unit.boxes.copy() * np.array([self.size / width, self.size / height] * 2, dtype=np.float32)
        a = self.augmentation
        if a is not None:
            if a.horizontal_flip and self.rng.random() < 0.5:
                resized = resized[:, ::-1]
                boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
            if a.vertical_flip and self.rng.random() < 0.5:
                resized = resized[::-1]
                boxes[:, [1, 3]] = self.size - boxes[:, [3, 1]]
            if a.intensity:
                resized = np.clip((resized - 0.5) * self.rng.uniform(0.8, 1.2) + 0.5 + self.rng.uniform(-0.1, 0.1), 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float()
        target = {"boxes": torch.from_numpy(boxes).float().reshape(-1, 4), "labels": torch.from_numpy(unit.labels).long()}
        return tensor, target, index


def collate(batch):
    images, targets, indices = zip(*batch)
    return torch.stack(images), list(targets), list(indices)
