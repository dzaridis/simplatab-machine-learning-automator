"""Small synthetic detection datasets in every annotation format, for the tests.

2D: noisy images with bright discs ("lesion") and bright squares ("nodule"), as PNG (and
DICOM) files. 3D: NIfTI volumes (or DICOM series) with bright balls and cubes.
"""
import csv
import json
import os

import numpy as np
from PIL import Image

CLASSES = ("lesion", "nodule")


def _objects(rng, size, count, dim=2, depth=0):
    """Random non-overlapping objects: [(class, box)] in continuous coordinates."""
    objects, tries = [], 0
    while len(objects) < count and tries < 200:
        tries += 1
        side = int(rng.integers(size // 6, size // 3))
        x, y = rng.integers(2, size - side - 2, 2)
        box = [int(x), int(y), int(x + side), int(y + side)]
        if dim == 3:
            thick = int(rng.integers(3, max(4, depth // 3)))
            z = int(rng.integers(1, depth - thick - 1))
            box = box[:2] + [z] + box[2:] + [z + thick]
        overlap = any(not (box[dim] <= b[0] or b[dim] <= box[0] or box[dim + 1] <= b[1] or b[dim + 1] <= box[1])
                      for _, b in objects)
        if not overlap:
            objects.append((CLASSES[int(rng.integers(0, len(CLASSES)))], box))
    return objects


def _draw2d(size, objects, rng):
    image = 0.2 + 0.05 * rng.standard_normal((size, size))
    y, x = np.mgrid[:size, :size]
    for label, (x1, y1, x2, y2) in objects:
        if label == "lesion":
            cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2
            image[(x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r ** 2] = 0.9
        else:
            image[y1:y2, x1:x2] = 0.7
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def make_2d(root, n_images, fmt="coco", seed=0, size=64, negatives=1, nested=False):
    """Writes ``n_images`` PNG images with 1-2 objects (and ``negatives`` without) and their
    annotations in ``fmt``: coco, yolo, voc, csv or mask. Returns {relative path: objects}."""
    rng = np.random.default_rng(seed)
    truth = {}
    image_dir = os.path.join(root, "images") if fmt == "yolo" else root
    for i in range(n_images + negatives):
        objects = _objects(rng, size, int(rng.integers(1, 3))) if i < n_images else []
        name = f"patient_{i // 2:02d}/img_{i:03d}.png" if nested else f"img_{i:03d}.png"
        path = os.path.join(image_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(_draw2d(size, objects, rng)).save(path)
        truth[os.path.relpath(path, root).replace(os.sep, "/")] = objects
    _write_2d(root, truth, fmt, size)
    return truth


def _write_2d(root, truth, fmt, size):
    if fmt == "coco":
        images, annotations = [], []
        for i, (name, objects) in enumerate(truth.items()):
            images.append({"id": i, "file_name": name, "width": size, "height": size})
            for label, (x1, y1, x2, y2) in objects:
                annotations.append({"id": len(annotations), "image_id": i, "category_id": CLASSES.index(label) + 1,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        categories = [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]
        with open(os.path.join(root, "annotations.json"), "w") as f:
            json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    elif fmt == "yolo":
        for name, objects in truth.items():
            label = os.path.join(root, "labels", os.path.splitext(os.path.relpath(name, "images"))[0] + ".txt")
            os.makedirs(os.path.dirname(label), exist_ok=True)
            with open(label, "w") as f:
                for cls, (x1, y1, x2, y2) in objects:
                    f.write(f"{CLASSES.index(cls)} {(x1 + x2) / 2 / size} {(y1 + y2) / 2 / size} "
                            f"{(x2 - x1) / size} {(y2 - y1) / size}\n")
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("\n".join(CLASSES) + "\n")
    elif fmt == "voc":
        for name, objects in truth.items():
            objs = "".join(f"<object><name>{c}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
                           f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>" for c, b in objects)
            with open(os.path.join(root, os.path.splitext(name)[0] + ".xml"), "w") as f:
                f.write(f"<annotation><filename>{os.path.basename(name)}</filename>{objs}</annotation>")
    elif fmt == "csv":
        with open(os.path.join(root, "boxes.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "class", "x_min", "y_min", "x_max", "y_max"])
            for name, objects in truth.items():
                if not objects:
                    writer.writerow([name, "", "", "", "", ""])
                for c, (x1, y1, x2, y2) in objects:
                    writer.writerow([name, c, x1, y1, x2, y2])
    elif fmt == "mask":
        for name, objects in truth.items():
            mask = np.zeros((size, size), np.uint8)
            for c, (x1, y1, x2, y2) in objects:
                mask[y1:y2, x1:x2] = CLASSES.index(c) + 1
            path = os.path.join(root, "masks", name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.fromarray(mask).save(path)
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("".join(f"{i + 1},{c}\n" for i, c in enumerate(CLASSES)))
    else:
        raise ValueError(fmt)


def _volume(shape, objects, rng):
    depth, size = shape[0], shape[1]
    volume = 100 + 20 * rng.standard_normal(shape)
    z, y, x = np.mgrid[:depth, :size, :size]
    for label, (x1, y1, z1, x2, y2, z2) in objects:
        if label == "lesion":
            c = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]
            r = [(x2 - x1) / 2, (y2 - y1) / 2, (z2 - z1) / 2]
            inside = (((x + 0.5 - c[0]) / r[0]) ** 2 + ((y + 0.5 - c[1]) / r[1]) ** 2 + ((z + 0.5 - c[2]) / r[2]) ** 2) <= 1
            volume[inside] = 900
        else:
            volume[z1:z2, y1:y2, x1:x2] = 600
    return volume.astype(np.int16)


def make_3d(root, n_volumes, fmt="csv", seed=0, size=48, depth=16, negatives=1, series=0):
    """NIfTI volumes (and ``series`` DICOM series folders) with 1-2 objects, annotated with a
    CSV of 3D boxes or with NIfTI masks. Returns {relative name: objects (x1 y1 z1 x2 y2 z2)}."""
    import nibabel as nib
    rng = np.random.default_rng(seed)
    truth = {}
    os.makedirs(root, exist_ok=True)
    for i in range(n_volumes + negatives):
        objects = _objects(rng, size, int(rng.integers(1, 3)), dim=3, depth=depth) if i < n_volumes else []
        volume = _volume((depth, size, size), objects, rng)  # (z, y, x)
        if i < series:
            name = f"case_{i:03d}"
            _write_series(os.path.join(root, name), volume)
        else:
            name = f"case_{i:03d}.nii.gz"
            nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), np.eye(4)), os.path.join(root, name))
        truth[name] = objects
    if fmt == "csv":
        with open(os.path.join(root, "boxes.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "class", "x_min", "y_min", "z_min", "x_max", "y_max", "z_max"])
            for name, objects in truth.items():
                for c, (x1, y1, z1, x2, y2, z2) in objects:
                    writer.writerow([name, c, x1, y1, z1, x2, y2, z2 - 1])  # z_max: last slice
    elif fmt == "mask":
        os.makedirs(os.path.join(root, "masks"))
        for name, objects in truth.items():
            mask = np.zeros((depth, size, size), np.uint8)
            for c, (x1, y1, z1, x2, y2, z2) in objects:
                mask[z1:z2, y1:y2, x1:x2] = CLASSES.index(c) + 1
            nib.save(nib.Nifti1Image(mask.transpose(2, 1, 0), np.eye(4)), os.path.join(root, "masks", name))
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("".join(f"{i + 1},{c}\n" for i, c in enumerate(CLASSES)))
    return truth


def _write_series(folder, volume):
    """A volume as a folder of single-frame CT DICOM slices (in shuffled file order)."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
    os.makedirs(folder)
    series = generate_uid()
    order = np.random.default_rng(1).permutation(volume.shape[0])
    for k in order:
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series
        ds.Modality, ds.PatientID = "CT", "SYNTHETIC"
        ds.ImagePositionPatient = [0, 0, float(k) * 2.5]
        ds.InstanceNumber = int(k) + 1
        ds.Rows, ds.Columns = volume.shape[1:]
        ds.PhotometricInterpretation, ds.SamplesPerPixel = "MONOCHROME2", 1
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = (volume[k] + 1024).astype(np.int16).tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(os.path.join(folder, f"slice_{k:03d}.dcm"), write_like_original=False)
