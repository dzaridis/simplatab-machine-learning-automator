"""Synthetic examples for the Object Detection automator. Writes, next to this script:

- Train.zip / Test.zip (2D, COCO JSON): 256 x 256 radiograph-like images of 40 patients
  (3 images each, one folder per patient) with round "nodule" and elongated "mass" lesions,
  some images without lesion;
- Train3D.zip / Test3D.zip (3D, CSV boxes): 64 x 64 x 32 CT-like NIfTI volumes (Hounsfield
  units) with spherical "nodule" lesions and their 3D boxes (z_min, z_max = first and last slice).

Run it to regenerate the zips: python Examples/object-detection/make_example.py
"""
import csv
import json
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["nodule", "mass"]


def _background(rng, size):
    """Smooth anatomy-like background (low-frequency noise) with fine texture."""
    coarse = rng.random((8, 8))
    image = np.asarray(Image.fromarray((coarse * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), np.float32) / 255
    y, x = np.mgrid[:size, :size] / size
    image = 0.25 + 0.35 * image + 0.15 * np.exp(-((x - 0.5) ** 2 + (y - 0.55) ** 2) / 0.08)
    return image + 0.04 * rng.standard_normal((size, size))


def _lesions(rng, size, count):
    objects = []
    for _ in range(count * 10):
        if len(objects) == count:
            break
        cls = CLASSES[int(rng.random() < 0.4)]
        w = int(rng.integers(14, 30)) if cls == "nodule" else int(rng.integers(30, 52))
        h = w if cls == "nodule" else int(w * rng.uniform(0.45, 0.7))
        x, y = int(rng.integers(16, size - w - 16)), int(rng.integers(16, size - h - 16))
        box = [x, y, x + w, y + h]
        if all(box[2] < b[0] - 4 or b[2] < box[0] - 4 or box[3] < b[1] - 4 or b[3] < box[1] - 4 for _, b in objects):
            objects.append((cls, box))
    return objects


def _draw(image, objects, rng):
    size = image.shape[0]
    y, x = np.mgrid[:size, :size]
    for cls, (x1, y1, x2, y2) in objects:
        cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
        d = ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2
        image += np.clip(1 - d, 0, 1) ** 0.5 * rng.uniform(0.25, 0.4)
    return image


def make_2d(folder, patients, seed):
    rng = np.random.default_rng(seed)
    images, annotations = [], []
    for p in range(patients):
        for k in range(3):
            objects = _lesions(rng, 256, int(rng.choice([0, 1, 1, 2]))) if (p + k) % 5 else []
            image = _draw(_background(rng, 256), objects, rng)
            name = f"patient_{seed:02d}{p:03d}/image_{k}.png"
            os.makedirs(os.path.join(folder, os.path.dirname(name)), exist_ok=True)
            Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(os.path.join(folder, name))
            images.append({"id": len(images) + 1, "file_name": name, "width": 256, "height": 256})
            for cls, (x1, y1, x2, y2) in objects:
                annotations.append({"id": len(annotations) + 1, "image_id": images[-1]["id"], "category_id": CLASSES.index(cls) + 1,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    with open(os.path.join(folder, "annotations.json"), "w") as f:
        json.dump({"images": images, "annotations": annotations,
                   "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}, f)


def make_3d(folder, volumes, seed):
    import nibabel as nib
    rng = np.random.default_rng(seed)
    rows = []
    z, y, x = np.mgrid[:32, :64, :64]
    for v in range(volumes):
        hu = -800 + 150 * rng.standard_normal((32, 64, 64))  # lung parenchyma
        hu[:, :, :6] = hu[:, :, -6:] = 40                     # chest wall
        name = f"case_{seed:02d}{v:03d}.nii.gz"
        for _ in range(int(rng.choice([0, 1, 1, 2]))):
            r = rng.uniform(3, 7)
            c = [rng.uniform(14, 50), rng.uniform(14, 50), rng.uniform(6, 26)]
            inside = ((x + 0.5 - c[0]) ** 2 + (y + 0.5 - c[1]) ** 2 + ((z + 0.5 - c[2]) * 1.5) ** 2) <= r ** 2
            if not inside.any():
                continue
            hu[inside] = 30 + 20 * rng.standard_normal(int(inside.sum()))
            zs, ys, xs = np.where(inside)
            rows.append([name, "nodule", int(xs.min()), int(ys.min()), int(zs.min()), int(xs.max()) + 1, int(ys.max()) + 1, int(zs.max())])
        nib.save(nib.Nifti1Image(hu.astype(np.int16).transpose(2, 1, 0), np.diag([0.8, 0.8, 1.2, 1])), os.path.join(folder, name))
        if not any(r[0] == name for r in rows):
            rows.append([name, "", "", "", "", "", "", ""])  # listed without box: a negative
    with open(os.path.join(folder, "boxes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "x_min", "y_min", "z_min", "x_max", "y_max", "z_max"])
        writer.writerows(rows)


def _zip(folder, target):
    shutil.make_archive(target[:-4], "zip", folder)


def main():
    work = tempfile.mkdtemp()
    try:
        for name, maker, count, seed in (("Train", make_2d, 30, 1), ("Test", make_2d, 10, 2),
                                         ("Train3D", make_3d, 24, 3), ("Test3D", make_3d, 8, 4)):
            folder = os.path.join(work, name)
            os.makedirs(folder)
            maker(folder, count, seed)
            _zip(folder, os.path.join(HERE, f"{name}.zip"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
