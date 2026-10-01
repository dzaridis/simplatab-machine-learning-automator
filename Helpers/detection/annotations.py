"""Annotations of the detection automator: the images (or volumes) of a split and their boxes.

Each zip holds images and their boxes in one of these formats, detected automatically:
- COCO: a JSON file with "images", "annotations" and "categories" (bbox = [x, y, width, height]);
- YOLO: one .txt file per image ("class cx cy w h", normalised), class names in data.yaml,
  classes.txt or a .names file;
- Pascal VOC: one .xml file per image (<object><name>, <bndbox>);
- CSV: one row per box: image, class, x_min, y_min, x_max, y_max; with z_min and z_max (first
  and last slice of the object) the images are 3D volumes;
- masks: label images in a "masks" (or "labels") folder with the names of the images; each
  connected region of a label value is a box (2D PNG/TIFF masks or 3D NIfTI masks).
Images without boxes are negatives (e.g. normal scans). Box coordinates are in pixels (voxels)
of the original image: x and y edges as in COCO (x_max = x_min + width).
"""
import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from typing import List

import numpy as np

from Helpers.image import io as mio
from . import volumes

MASK_FOLDERS = {"masks", "mask", "labels", "label", "segmentations", "annotations"}
CLASS_FILES = ("classes.txt", "classes.names", "obj.names", "labels.txt")
COLUMNS = {
    "image": ("image", "image_id", "file", "filename", "file_name", "path", "volume", "image_path"),
    "class": ("class", "label", "category", "class_name", "category_name"),
    "x_min": ("x_min", "xmin", "x1", "left"), "y_min": ("y_min", "ymin", "y1", "top"),
    "x_max": ("x_max", "xmax", "x2", "right"), "y_max": ("y_max", "ymax", "y2", "bottom"),
    "z_min": ("z_min", "zmin", "z1", "slice_min", "first_slice"), "z_max": ("z_max", "zmax", "z2", "slice_max", "last_slice"),
}
FORMAT_NAMES = {"coco": "COCO JSON", "yolo": "YOLO", "voc": "Pascal VOC XML", "csv": "CSV", "mask": "masks"}


class AnnotationError(ValueError):
    """A problem with the annotations, explained to the user."""


@dataclass
class Sample:
    path: str                 # image file, NIfTI / multi-frame DICOM volume, or folder of DICOM slices
    name: str                 # path relative to the split folder
    boxes: np.ndarray         # (n, 4) x1 y1 x2 y2, or (n, 6) x1 y1 z1 x2 y2 z2
    labels: List[str]
    group: str = ""           # patient / study: the first sub-folder


@dataclass
class Annotated:
    samples: List[Sample]
    classes: List[str]
    dim: int
    format: str
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------
# Files of a split
# ---------------------------------------------------------------------------------------

def split_root(folder):
    """Skips wrapping folders (a single sub-folder and no file)."""
    while True:
        entries = [e for e in os.listdir(folder) if not e.startswith(".") and e != "__MACOSX"]
        if len(entries) == 1 and os.path.isdir(os.path.join(folder, entries[0])) and entries[0].lower() not in MASK_FOLDERS | {"images"}:
            folder = os.path.join(folder, entries[0])
        else:
            return folder


def _relative(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


def _files(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__MACOSX")
        for name in sorted(filenames):
            if not name.startswith("."):
                out.append(os.path.join(dirpath, name))
    return out


def _in_mask_folder(path, root):
    return any(part.lower() in MASK_FOLDERS for part in _relative(path, root).split("/")[:-1])


def image_files(root):
    """2D images (and NIfTI / DICOM volumes), outside the mask folders."""
    return [p for p in _files(root) if mio.file_kind(p) and not _in_mask_folder(p, root)]


def dicom_series_folders(root):
    """Folders holding several DICOM files: one series (volume) each."""
    folders = {}
    for path in image_files(root):
        if mio.file_kind(path) == "dicom":
            folders.setdefault(os.path.dirname(path), []).append(path)
    return [folder for folder, files in folders.items() if len(files) > 1]


def _group(name):
    """The patient or study of an image: its first sub-folder (after an "images" folder)."""
    parts = name.split("/")
    while len(parts) > 1 and parts[0].lower() in ("images", "imgs", "img"):
        parts = parts[1:]
    return parts[0] if len(parts) > 1 else name


def _stem(path):
    name = os.path.basename(path)
    for extension in (".nii.gz", ".nii"):
        if name.lower().endswith(extension):
            return name[:-len(extension)]
    return os.path.splitext(name)[0]


class _Lookup:
    """Finds an image from a name given by the annotations: relative path, else file name,
    else name without extension."""

    def __init__(self, paths, root):
        self.root = root
        self.by_relative = {_relative(p, root): p for p in paths}
        self.by_name, self.by_stem = {}, {}
        for p in paths:
            self.by_name.setdefault(os.path.basename(p), []).append(p)
            self.by_stem.setdefault(_stem(p), []).append(p)

    def find(self, name, base=None):
        name = str(name).replace("\\", "/").strip()
        candidates = [name]
        if base:
            candidates.insert(0, _relative(os.path.normpath(os.path.join(base, name)), self.root))
        for candidate in candidates:
            if candidate in self.by_relative:
                return self.by_relative[candidate]
        for table, key in ((self.by_name, os.path.basename(name)), (self.by_stem, _stem(name))):
            if len(table.get(key, [])) == 1:
                return table[key][0]
        return None


def _class_names(root):
    """Class names of YOLO / mask annotations, if given (data.yaml, classes.txt, *.names)."""
    for path in _files(root):
        name = os.path.basename(path).lower()
        if name in ("data.yaml", "data.yml", "dataset.yaml"):
            import yaml
            with open(path) as f:
                names = (yaml.safe_load(f) or {}).get("names")
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return {i: str(v) for i, v in enumerate(names)}
    for path in _files(root):
        name = os.path.basename(path).lower()
        if name in CLASS_FILES or name.endswith(".names"):
            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]
            pairs = [re.split(r"[,;\t]", line, maxsplit=1) for line in lines]
            if all(len(p) == 2 and p[0].strip().isdigit() for p in pairs):
                return {int(p[0]): p[1].strip() for p in pairs}
            return {i: line for i, line in enumerate(lines)}
    return {}


# ---------------------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------------------

def _coco_files(root):
    found = []
    for path in _files(root):
        if path.lower().endswith(".json"):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and {"images", "annotations", "categories"} <= set(data):
                found.append((path, data))
    return found


def _csv_files(root):
    found = []
    for path in _files(root):
        if path.lower().endswith(".csv") and not _in_mask_folder(path, root):
            with open(path, newline="", encoding="utf-8-sig") as f:
                header = [h.strip().lower() for h in next(csv.reader(f), [])]
            columns = {key: next((h for h in names if h in header), None) for key, names in COLUMNS.items()}
            if all(columns[k] for k in ("image", "x_min", "y_min", "x_max", "y_max")):
                found.append((path, columns))
    return found


def _voc_files(root):
    found = []
    for path in _files(root):
        if path.lower().endswith(".xml"):
            try:
                tree = ET.parse(path)
            except ET.ParseError:
                continue
            if tree.getroot().tag == "annotation":
                found.append((path, tree.getroot()))
    return found


_YOLO_LINE = re.compile(r"^\s*\d+(\s+[-+0-9.eE]+){4}\s*$")


def _yolo_files(root):
    found = []
    for path in _files(root):
        name = os.path.basename(path).lower()
        if name.endswith(".txt") and name not in CLASS_FILES:
            with open(path, errors="replace") as f:
                lines = [line for line in f if line.strip()]
            if all(_YOLO_LINE.match(line) for line in lines):
                found.append(path)
    return found


def _mask_files(root):
    return [p for p in _files(root) if _in_mask_folder(p, root) and mio.file_kind(p) in ("raster", "nifti")]


def detect_format(root):
    detected = {"coco": _coco_files(root), "csv": _csv_files(root), "voc": _voc_files(root), "mask": _mask_files(root)}
    labels = _yolo_files(root)
    # Empty .txt files alone do not make a YOLO dataset
    detected["yolo"] = labels if any(os.path.getsize(p) for p in labels) else []
    present = [k for k, v in detected.items() if v]
    if not present:
        raise AnnotationError("No annotations found. Add a COCO JSON file, YOLO .txt labels, Pascal VOC .xml files, "
                              "a CSV file (image, class, x_min, y_min, x_max, y_max[, z_min, z_max]) or a masks folder.")
    if len(present) > 1:
        raise AnnotationError("Several annotation formats were found (" + ", ".join(FORMAT_NAMES[k] for k in present)
                              + "): keep one per zip.")
    return present[0], detected[present[0]]


def _new(path, root):
    name = _relative(path, root)
    return Sample(path=path, name=name, boxes=[], labels=[], group=_group(name))


def _parse_coco(root, files, warnings):
    lookup = _Lookup(image_files(root), root)
    samples, names, crowd, missing = {}, {}, 0, set()
    for path, data in files:
        categories = {c["id"]: str(c.get("name", c["id"])) for c in data["categories"]}
        for cid in sorted(categories):
            names.setdefault(categories[cid], cid)
        images = {}
        for image in data["images"]:
            found = lookup.find(image["file_name"], os.path.dirname(path))
            if found is None:
                missing.add(image["file_name"])
                continue
            images[image["id"]] = samples.setdefault(found, _new(found, root))
        for a in data["annotations"]:
            if a.get("iscrowd"):
                crowd += 1
                continue
            sample = images.get(a["image_id"])
            if sample is None or a.get("category_id") not in categories:
                continue
            x, y, w, h = [float(v) for v in a["bbox"]]
            sample.boxes.append((x, y, x + w, y + h))
            sample.labels.append(categories[a["category_id"]])
    if missing:
        warnings.append(f"{len(missing)} image(s) listed in the COCO file were not found in the zip (e.g. {sorted(missing)[0]}).")
    if crowd:
        warnings.append(f"{crowd} crowd annotation(s) were ignored.")
    order = sorted(names, key=lambda n: names[n])
    return list(samples.values()), order, 2


def _parse_csv(root, files, warnings):
    paths = image_files(root)
    series = dicom_series_folders(root)
    lookup = _Lookup(paths + series, root)
    samples, unmatched, dims = {}, set(), set()
    for path, columns in files:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                name = row.get(columns["image"], "")
                found = lookup.find(name, os.path.dirname(path))
                if found is None:
                    unmatched.add(name)
                    continue
                sample = samples.setdefault(found, _new(found, root))
                coords = [row.get(columns[k], "") for k in ("x_min", "y_min", "x_max", "y_max")]
                if not all(coords):  # an image listed without box: a negative
                    continue
                label = row.get(columns["class"], "") if columns["class"] else "object"
                x1, y1, x2, y2 = [float(v) for v in coords]
                if columns["z_min"] and columns["z_max"] and row.get(columns["z_min"]) and row.get(columns["z_max"]):
                    z1, z2 = float(row[columns["z_min"]]), float(row[columns["z_max"]]) + 1  # last slice included
                    sample.boxes.append((x1, y1, z1, x2, y2, z2))
                    dims.add(3)
                else:
                    sample.boxes.append((x1, y1, x2, y2))
                    dims.add(2)
                sample.labels.append(label or "object")
    if unmatched:
        warnings.append(f"{len(unmatched)} image(s) of the CSV file were not found in the zip (e.g. {sorted(unmatched)[0]}).")
    if len(dims) > 1:
        raise AnnotationError("The CSV file mixes 2D boxes and 3D boxes (with z_min and z_max): use one kind.")
    dim = dims.pop() if dims else (3 if any(c["z_min"] for _, c in files) else 2)
    if dim == 3:
        volumes_found = [p for p in paths if volumes.is_nifti(p)] + series + [
            p for p in paths if mio.file_kind(p) == "dicom" and os.path.dirname(p) not in series]
        for path in volumes_found:
            samples.setdefault(path, _new(path, root))
    else:
        for path in paths:
            samples.setdefault(path, _new(path, root))
    return list(samples.values()), None, dim


def _parse_voc(root, files, warnings):
    lookup = _Lookup(image_files(root), root)
    samples, missing = {}, 0
    for path, node in files:
        filename = node.findtext("filename") or ""
        found = lookup.find(filename, os.path.dirname(path)) if filename else None
        found = found or lookup.find(_stem(path))
        if found is None:
            missing += 1
            continue
        sample = samples.setdefault(found, _new(found, root))
        for obj in node.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue
            x1, y1, x2, y2 = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            sample.boxes.append((x1, y1, x2, y2))
            sample.labels.append((obj.findtext("name") or "object").strip())
    if missing:
        warnings.append(f"{missing} VOC file(s) do not match an image of the zip.")
    for path in image_files(root):
        samples.setdefault(path, _new(path, root))
    return list(samples.values()), None, 2


def _parse_yolo(root, files, warnings):
    paths = image_files(root)
    names = _class_names(root)
    labels = {}
    for path in files:
        rel = _relative(path, root)
        labels[rel] = path
        labels.setdefault(_stem(path), path)
    samples, used = [], set()
    for image in paths:
        rel = _relative(image, root)
        mirrored = re.sub(r"(^|/)images/", r"\1labels/", rel)
        key = os.path.splitext(mirrored)[0] + ".txt"
        label = labels.get(key) or labels.get(os.path.splitext(rel)[0] + ".txt") or labels.get(_stem(image))
        sample = _new(image, root)
        if label:
            used.add(label)
            width, height = volumes.image_size(image)
            with open(label) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w, h = int(parts[0]), *[float(v) for v in parts[1:]]
                    sample.boxes.append(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
                    sample.labels.append(names.get(cls, f"class_{cls}"))
        samples.append(sample)
    unused = [p for p in files if p not in used]
    if unused:
        warnings.append(f"{len(unused)} YOLO label file(s) do not match an image (e.g. {_relative(unused[0], root)}).")
    order = [names[k] for k in sorted(names)] if names else None
    return samples, order, 2


def _parse_masks(root, files, warnings):
    paths = [p for p in image_files(root)]
    series = dicom_series_folders(root)
    masks = {_stem(p): p for p in files}
    names = _class_names(root)
    samples, dims, used = [], set(), set()
    candidates = [p for p in paths if not (mio.file_kind(p) == "dicom" and os.path.dirname(p) in series)] + series
    for image in candidates:
        sample = _new(image, root)
        mask = masks.get(_stem(image)) or masks.get(_stem(image).replace("_image", "").replace("_img", ""))
        if mask:
            used.add(mask)
            values = volumes.read_mask(mask)
            dims.add(values.ndim)
            for value, box in volumes.mask_boxes(values):
                sample.boxes.append(box)
                sample.labels.append(names.get(value, f"label_{value}"))
        samples.append(sample)
    unused = [p for p in files if p not in used]
    if unused:
        warnings.append(f"{len(unused)} mask(s) do not match an image (e.g. {_relative(unused[0], root)}).")
    if len(dims) > 1:
        raise AnnotationError("The masks mix 2D images and 3D volumes: use one kind.")
    dim = dims.pop() if dims else 2
    if dim == 3:  # volumes only
        samples = [s for s in samples if volumes.is_nifti(s.path) or os.path.isdir(s.path)
                   or (mio.file_kind(s.path) == "dicom" and volumes.volume_shape(s.path)[0] > 1)]
    order = [names[k] for k in sorted(names)] if names else None
    return samples, order, dim


PARSERS = {"coco": _parse_coco, "csv": _parse_csv, "voc": _parse_voc, "yolo": _parse_yolo, "mask": _parse_masks}


def load_split(folder):
    """The samples of a split with their boxes. Raises AnnotationError for unusable data."""
    root = split_root(folder)
    fmt, files = detect_format(root)
    warnings = []
    samples, order, dim = PARSERS[fmt](root, files, warnings)
    invalid = 0
    for sample in samples:
        boxes = np.asarray(sample.boxes, dtype=np.float32).reshape(-1, 2 * dim)
        keep = np.all(boxes[:, dim:] > boxes[:, :dim], axis=1) & np.all(boxes[:, :dim] >= -1, axis=1)
        invalid += int((~keep).sum())
        sample.boxes = boxes[keep]
        sample.labels = [label for label, k in zip(sample.labels, keep) if k]
    if invalid:
        warnings.append(f"{invalid} box(es) with a zero or negative size were ignored.")
    samples = sorted(samples, key=lambda s: s.name)
    if not samples:
        raise AnnotationError("No images were found next to the annotations.")
    found = Counter(label for s in samples for label in s.labels)
    classes = [c for c in (order or []) if c in found] + sorted(c for c in found if c not in (order or []))
    return Annotated(samples=samples, classes=classes, dim=dim, format=fmt, warnings=warnings)
