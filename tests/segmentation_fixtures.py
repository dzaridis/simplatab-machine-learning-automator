"""Synthetic data for the tests of the segmentation automator, in every supported layout."""
import json
import os

import numpy as np
from PIL import Image

from image3d_fixtures import write_dicom_series, write_nifti

PALETTE = [0, 0, 0, 220, 40, 40, 40, 160, 60]  # background, class 1 (red), class 2 (green)


def _shapes_2d(rng, size):
    """An RGB image with a disc (class 1) and a square (class 2), and its label map."""
    image = np.full((size, size, 3), 40, np.float32) + 15 * rng.standard_normal((size, size, 3))
    labels = np.zeros((size, size), np.uint8)
    y, x = np.mgrid[:size, :size]
    cx, cy, r = rng.uniform(0.25, 0.45) * size, rng.uniform(0.3, 0.7) * size, rng.uniform(0.1, 0.18) * size
    disc = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    image[disc] = [200, 60, 60]
    labels[disc] = 1
    s = int(rng.uniform(0.15, 0.25) * size)
    x0, y0 = int(rng.uniform(0.55, 0.95) * size - s), int(rng.uniform(0.1, 0.9) * size - s / 2)
    x0, y0 = max(0, min(size - s, x0)), max(0, min(size - s, y0))
    image[y0:y0 + s, x0:x0 + s] = [60, 180, 80]
    labels[y0:y0 + s, x0:x0 + s] = 2
    return np.clip(image, 0, 255).astype(np.uint8), labels


def make_2d(root, n, seed=0, size=64, mask="palette", nested=False, suffix=""):
    """RGB images with palette ("palette"), colour ("rgb") or integer ("gray") masks."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        image, labels = _shapes_2d(rng, size)
        name = f"patient_{i // 2:02d}/img_{i:03d}" if nested else f"img_{i:03d}"
        for folder in ("images", "masks"):
            os.makedirs(os.path.dirname(os.path.join(root, folder, name)), exist_ok=True)
        Image.fromarray(image).save(os.path.join(root, "images", name + ".jpg" if i % 3 == 0 else name + ".png"))
        target = os.path.join(root, "masks", name + suffix + ".png")
        if mask == "palette":
            m = Image.fromarray(labels, mode="P")
            m.putpalette(PALETTE)
            m.save(target)
        elif mask == "rgb":
            colours = np.array(PALETTE, np.uint8).reshape(-1, 3)
            Image.fromarray(colours[labels]).save(target)
        else:
            Image.fromarray(labels * 100).save(target)  # values 0, 100, 200
    with open(os.path.join(root, "classes.txt"), "w") as f:
        f.write("0,background\n1,disc\n2,square\n" if mask == "gray" else "")
    if mask != "gray":
        os.remove(os.path.join(root, "classes.txt"))


def _volume(rng, shape=(12, 40, 40)):
    """A 3D volume with a bright sphere (class 1) and its label map."""
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]]
    c = [rng.uniform(4, shape[0] - 4), rng.uniform(12, shape[1] - 12), rng.uniform(12, shape[2] - 12)]
    r = rng.uniform(5, 8)
    inside = ((z - c[0]) * 2.5) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r ** 2
    volume = 300 + 40 * rng.standard_normal(shape)
    volume[inside] += 400
    return volume, inside.astype(np.uint8)


def make_3d(root, n, seed=0, layout="nifti"):
    """Layouts: "nifti" (images/case.nii.gz), "folder" (images/case/t2/ DICOM + adc.nii.gz,
    mask on the T2 grid) or "nnunet" (imagesTr/case_0000.nii.gz, labelsTr/, dataset.json)."""
    rng = np.random.default_rng(seed)
    spacing, origin = (0.8, 0.8, 2.5), (-16.0, -16.0, -15.0)
    for i in range(n):
        volume, labels = _volume(rng)
        name = f"case_{i:03d}"
        if layout == "nifti":
            write_nifti(os.path.join(root, "images", name + ".nii.gz"), volume, spacing, origin)
            write_nifti(os.path.join(root, "masks", name + ".nii.gz"), labels, spacing, origin)
        elif layout == "folder":
            write_dicom_series(os.path.join(root, "images", name, "t2"), volume, spacing, origin)
            write_nifti(os.path.join(root, "images", name, "adc.nii.gz"), 2000 - volume, spacing, origin)
            write_nifti(os.path.join(root, "masks", name + ".nii.gz"), labels, spacing, origin)
        else:
            write_nifti(os.path.join(root, "imagesTr", name + "_0000.nii.gz"), volume, spacing, origin)
            write_nifti(os.path.join(root, "imagesTr", name + "_0001.nii.gz"), 2000 - volume, spacing, origin)
            write_nifti(os.path.join(root, "labelsTr", name + ".nii.gz"), labels, spacing, origin)
    if layout == "nnunet":
        with open(os.path.join(root, "dataset.json"), "w") as f:
            json.dump({"channel_names": {"0": "T2", "1": "ADC"}, "labels": {"background": 0, "lesion": 1},
                       "numTraining": n, "file_ending": ".nii.gz"}, f)
