"""Cases of the segmentation automator: images and their masks, read, aligned and cached.

Two layouts are accepted in Train.zip and Test.zip:
- images/ and masks/ folders whose paths mirror each other, e.g.
      images/case_01.png           masks/case_01.png        (2D: PNG, JPEG, TIFF, DICOM, NIfTI)
      images/patient_07/scan.dcm   masks/patient_07/scan.png
      images/case_02.nii.gz        masks/case_02.nii.gz     (3D: NIfTI, multi-frame DICOM)
      images/case_03/              masks/case_03.nii.gz     (3D: a folder of DICOM slices, or of
                                                             several series, e.g. t2/ and adc.nii.gz)
  Mask names may carry a suffix (case_01_mask.png). A sub-folder groups the cases of a patient
  (they stay in the same validation fold).
- the nnU-Net raw format: imagesTr/ (case_0000.nii.gz, case_0001.nii.gz: one file per channel),
  labelsTr/ (case.nii.gz) and dataset.json (labels and channel names); imagesTs/ and labelsTs/ in
  Test.zip.

Masks are label images: integer values (0 = background; binary masks 0/255 are read as 0/1),
palette PNGs or colour (RGB) masks. Their values are mapped to classes 0..K over the whole dataset;
names come from classes.txt ("1,liver"), labels.json, dataset.json, or default to class_<value>.

Each case is cached in the nnU-Net naming (<id>_0000.nii.gz per channel, <id>.nii.gz for the mask):
3D series are reoriented (LPS) and aligned on the first series, the mask is put on the same grid
(copied when it has the shape of the image, else resampled in patient space). 2D images are stored
as volumes of one slice.
"""
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List

import numpy as np

from Helpers.image import io as mio

IMAGE_DIRS = {"images", "image", "imgs", "img", "imagestr", "imagests"}
MASK_DIRS = {"masks", "mask", "labels", "label", "labelstr", "labelsts", "segmentations", "segmentation",
             "annotations", "ground_truth", "groundtruth", "gt"}
MASK_SUFFIXES = ("_segmentation", "_labels", "_label", "_mask", "_seg", "_gt", "-mask", "-seg")
CLASS_FILES = ("classes.txt", "labels.txt", "classes.csv", "labels.csv")
MAX_CLASSES = 32


class SegmentationDataError(ValueError):
    """A problem with the uploaded data, shown to the user as is."""


@dataclass
class Case:
    id: str                       # path of the mask without extension (and suffix), relative to masks/
    group: str                    # patient: the first sub-folder, else the case itself
    mask: str                     # mask file
    sources: List[dict]           # channels: {"name", "kind": "file" | "dicom", "path", "uid"}
    folder: bool = False          # the image is a folder of series (3D, one or more series)
    notes: List[str] = field(default_factory=list)


def _visible(entries):
    return sorted(e for e in entries if not e.startswith(".") and e != "__MACOSX")


def _files(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = _visible(dirnames)
        out += [os.path.join(dirpath, name) for name in _visible(filenames)]
    return out


def _strip_extension(name):
    lower = name.lower()
    for extension in (".nii.gz", ".nii"):
        if lower.endswith(extension):
            return name[:-len(extension)]
    return os.path.splitext(name)[0]


def _key(path, root):
    """Path relative to root, without extension, '/' separated."""
    return _strip_extension(os.path.relpath(path, root).replace(os.sep, "/"))


def _without_suffix(key):
    lower = key.lower()
    for suffix in MASK_SUFFIXES:
        if lower.endswith(suffix):
            return key[:-len(suffix)]
    return key


def split_root(folder):
    """Skips wrapping folders down to the folder holding the image and mask folders."""
    while True:
        entries = _visible(os.listdir(folder))
        names = {e.lower() for e in entries if os.path.isdir(os.path.join(folder, e))}
        if names & IMAGE_DIRS or names & MASK_DIRS:
            return folder
        dirs = [e for e in entries if os.path.isdir(os.path.join(folder, e))]
        if len(dirs) == 1 and len(entries) == 1:
            folder = os.path.join(folder, dirs[0])
        else:
            return folder


def _find_dirs(root):
    entries = [e for e in _visible(os.listdir(root)) if os.path.isdir(os.path.join(root, e))]
    images = [e for e in entries if e.lower() in IMAGE_DIRS]
    masks = [e for e in entries if e.lower() in MASK_DIRS]
    if not images or not masks:
        raise SegmentationDataError(
            "Put the images in an images/ folder and the masks in a masks/ folder (or use the nnU-Net "
            "layout imagesTr/ and labelsTr/).")
    # nnU-Net: imagesTr with labelsTr (or imagesTs with labelsTs)
    for suffix in ("tr", "ts"):
        image = next((e for e in images if e.lower() == f"images{suffix}"), None)
        mask = next((e for e in masks if e.lower() == f"labels{suffix}"), None)
        if image and mask:
            return os.path.join(root, image), os.path.join(root, mask)
    return os.path.join(root, images[0]), os.path.join(root, masks[0])


_CHANNEL = re.compile(r"^(.*)_(\d{4})$")


def scan_split(folder):
    """The cases of a split, the unmatched files (images without mask, masks without image) and
    the format ("folders" or "nnunet")."""
    from Helpers.image3d.volumes import _folder_series, series_name
    root = split_root(folder)
    image_dir, mask_dir = _find_dirs(root)
    masks = [p for p in _files(mask_dir) if mio.file_kind(p)]
    images = [p for p in _files(image_dir) if mio.file_kind(p)]
    # nnU-Net channel files: <case>_0000, <case>_0001, ... next to masks named <case>
    mask_keys = {_key(p, mask_dir) for p in masks}
    channel_files = {}
    for path in images:
        match = _CHANNEL.match(_key(path, image_dir))
        if match and match.group(1) in mask_keys:
            channel_files.setdefault(match.group(1), []).append((int(match.group(2)), path))
    nnunet = bool(channel_files) and len(channel_files) >= len(mask_keys) / 2
    by_key = {}
    for path in images:
        by_key.setdefault(_key(path, image_dir), []).append(path)
    folders = {}
    for dirpath, dirnames, _ in os.walk(image_dir):
        dirnames[:] = _visible(dirnames)
        for d in dirnames:
            folders[_key(os.path.join(dirpath, d), image_dir)] = os.path.join(dirpath, d)

    cases, used, unmatched = [], set(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for mask in masks:
            key = _key(mask, mask_dir)
            sources, folder = None, False
            for candidate in dict.fromkeys([key, _without_suffix(key)]):
                if nnunet and candidate in channel_files:
                    files = [p for _, p in sorted(channel_files[candidate])]
                    sources = [{"name": f"channel_{i}", "kind": "file", "path": p} for i, p in enumerate(files)]
                    used.update(files)
                elif candidate in by_key and len(by_key[candidate]) == 1:
                    path = by_key[candidate][0]
                    sources = [{"name": "image", "kind": "file", "path": path}]
                    used.add(path)
                elif candidate in folders:
                    series = []
                    for dirpath, dirnames, _ in os.walk(folders[candidate]):
                        dirnames[:] = _visible(dirnames)
                        found, _ = _folder_series(dirpath, folders[candidate], pool)
                        series += found
                    if len(series) > 1 and len({s["name"] for s in series}) < len(series):
                        counts = Counter()
                        for s in series:
                            counts[s["name"]] += 1
                            if counts[s["name"]] > 1:
                                s["name"] = f"{s['name']}_{counts[s['name']]}"
                    if len(series) == 1:
                        series[0]["name"] = "image"
                    sources, folder = sorted(series, key=lambda s: s["name"]) or None, True
                    used.update(_files(folders[candidate]))
                if sources:
                    key = candidate
                    break
            if not sources:
                unmatched.append(os.path.relpath(mask, root))
                continue
            parts = key.split("/")
            cases.append(Case(id=key, group=parts[0] if len(parts) > 1 else key, mask=mask, sources=sources, folder=folder))
    lonely = [os.path.relpath(p, root) for p in images if p not in used]
    return cases, {"masks_without_image": unmatched, "images_without_mask": lonely}, "nnunet" if nnunet else "folders"


# ---------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------

def _read_mask_array(path):
    """A mask as an integer array (2D raster: (H, W) or (H, W, 3) for colour masks; others:
    the SimpleITK array (z, y, x))."""
    kind = mio.file_kind(path)
    if kind == "raster":
        from PIL import Image
        with Image.open(path) as image:
            image.seek(0)
            if image.mode == "P":
                return np.array(image)
            if image.mode in ("RGB", "RGBA"):
                rgb = np.array(image.convert("RGB"))
                if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
                    return rgb[..., 0]
                return rgb
            if image.mode == "LA":
                return np.array(image.convert("L"))
            return np.array(image)
    import SimpleITK as sitk
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() > 1:
        array = array[..., 0]
    return array


def mask_values(path):
    """The values (or colours) of a mask and their counts."""
    array = _read_mask_array(path)
    if array.ndim == 3 and array.shape[-1] == 3 and mio.file_kind(path) == "raster":
        colours, counts = np.unique(array.reshape(-1, 3), axis=0, return_counts=True)
        return {tuple(int(v) for v in c): int(n) for c, n in zip(colours, counts)}
    if not np.issubdtype(array.dtype, np.integer):
        if not np.allclose(array, np.round(array)):
            raise SegmentationDataError(f"{os.path.basename(path)} is not a label image (non-integer values).")
        array = np.round(array).astype(np.int64)
    values, counts = np.unique(array, return_counts=True)
    return {int(v): int(n) for v, n in zip(values, counts)}


def _class_names(root):
    """Names of the label values given with the data: dataset.json (nnU-Net), labels.json or
    classes.txt / classes.csv ("value,name" or one name per line from 0)."""
    names = {}
    for path in _files(root):
        base = os.path.basename(path).lower()
        if base in ("dataset.json", "labels.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            labels = data.get("labels", data) if isinstance(data, dict) else {}
            for k, v in labels.items():
                if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
                    names[int(v)] = str(k)               # nnU-Net: {"background": 0, "liver": 1}
                elif str(k).isdigit():
                    names[int(k)] = str(v)               # {"1": "liver"}
            if names:
                return names
    for path in _files(root):
        if os.path.basename(path).lower() in CLASS_FILES:
            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]
            pairs = [re.split(r"[,;\t]", line, maxsplit=1) for line in lines]
            if all(len(p) == 2 and p[0].strip().isdigit() for p in pairs):
                return {int(p[0]): p[1].strip() for p in pairs}
            return {i: line for i, line in enumerate(lines)}
    return {}


def label_mapping(value_counts, names):
    """Classes from the values found in the masks: {"values": [raw values], "classes": [names],
    "counts": [pixels]}. Colours (tuples) and integers are sorted with the background first;
    binary masks 0/255 become 0/1."""
    values = sorted(value_counts, key=lambda v: (v != 0 and v != (0, 0, 0), v))
    if values and isinstance(values[0], tuple):
        background = (0, 0, 0)
    else:
        background = 0
    if background not in value_counts:
        values = [background] + values
    if len(values) > MAX_CLASSES + 1:
        raise SegmentationDataError(f"The masks hold {len(values)} different values: at most {MAX_CLASSES} classes "
                                    "(plus the background) are supported. Are they label images?")
    classes = []
    for i, value in enumerate(values):
        if i == 0:
            classes.append(names.get(0, "background") if not isinstance(value, tuple) else "background")
        elif isinstance(value, tuple):
            classes.append(names.get(i, "#%02x%02x%02x" % value))
        elif len(values) == 2:
            classes.append(names.get(value, names.get(1, "foreground")))
        else:  # names given by mask value, else by class number
            classes.append(names.get(value, names.get(i, f"class_{value}")))
    return {"values": [list(v) if isinstance(v, tuple) else v for v in values], "classes": classes,
            "counts": [value_counts.get(v, 0) for v in values]}


def map_labels(array, mapping):
    """Raw mask values (or colours) to class indices 0..K (unknown values: background)."""
    values = mapping["values"]
    if isinstance(values[0], list):  # colour masks: (..., 3)
        if array.shape[-1] != 3:
            raise ValueError("a grey-level mask in a dataset of colour masks")
        out = np.zeros(array.shape[:-1], np.uint8)
        for i, colour in enumerate(values):
            out[np.all(array == np.array(colour), axis=-1)] = i
        return out
    out = np.zeros(array.shape, np.uint8)
    for i, value in enumerate(values):
        out[array == value] = i
    return out


# ---------------------------------------------------------------------------------------
# Reading and caching
# ---------------------------------------------------------------------------------------

def _raster_channels(path):
    """A 2D raster image as channels (each (1, H, W) float32), and whether it is RGB."""
    from PIL import Image
    with Image.open(path) as image:
        image.seek(0)
        if image.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
            return [np.array(image, dtype=np.float32)[None]], False
        if image.mode in ("1", "L", "LA"):
            return [np.array(image.convert("L"), dtype=np.float32)[None]], False
        rgb = np.array(image.convert("RGB"), dtype=np.float32)
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
        return [rgb[..., 0][None]], False
    return [rgb[..., c][None] for c in range(3)], True


def _image_from_array(array, spacing=(1.0, 1.0, 1.0)):
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(np.ascontiguousarray(array))
    image.SetSpacing(tuple(float(s) for s in spacing))
    return image


def read_case(case, channels=None):
    """(channel images, mask array (z, y, x) of raw values, info) of a case, aligned on the same grid.
    ``channels``: the series names to use for folder cases (None: all, in order)."""
    import SimpleITK as sitk
    from Helpers.image3d.volumes import read_series
    info = {"rgb": False, "modality": [], "dim": 2}
    sources = case.sources
    if case.folder and channels:
        by_name = {s["name"]: s for s in sources}
        missing = [c for c in channels if c not in by_name]
        if missing:
            raise ValueError(f"no series {', '.join(missing)}")
        sources = [by_name[c] for c in channels]
    first = sources[0]
    raster = first["kind"] == "file" and mio.file_kind(first["path"]) == "raster"
    if raster:
        arrays, info["rgb"] = [], False
        for source in sources:
            found, rgb = _raster_channels(source["path"])
            arrays += found
            info["rgb"] |= rgb
        images = [_image_from_array(a) for a in arrays]
        info["modality"] = ["RGB" if info["rgb"] else "image"] * len(images)
        original_shape = arrays[0].shape
    else:
        reference, header = read_series(first)
        original = reference
        original_shape = sitk.GetArrayViewFromImage(reference).shape
        info["dim"] = 3 if reference.GetSize()[2] > 1 else 2
        if info["dim"] == 3:
            reference = sitk.DICOMOrient(reference, "LPS")
        images, info["modality"] = [reference], [header.get("modality") or ""]
        for source in sources[1:]:
            image, header = read_series(source)
            if image.GetSize() == original.GetSize() and info["dim"] == 2:
                image.CopyInformation(reference)
                aligned = image
            else:
                background = float(sitk.GetArrayViewFromImage(image).min())
                aligned = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear, background, sitk.sitkFloat32)
            images.append(aligned)
            info["modality"].append(header.get("modality") or "")

    # The mask, on the grid of the first channel
    raw = _read_mask_array(case.mask)
    colour = raw.ndim == 3 and raw.shape[-1] == 3 and mio.file_kind(case.mask) == "raster"
    if raster or (info["dim"] == 2 and mio.file_kind(case.mask) == "raster"):
        target = sitk.GetArrayViewFromImage(images[0]).shape[1:]
        mask2d = raw if colour else raw.reshape(raw.shape[-2:]) if raw.ndim > 2 and raw.shape[0] == 1 else raw
        if mask2d.shape[:2] != tuple(target):
            from PIL import Image
            case.notes.append(f"mask resized from {mask2d.shape[1]}x{mask2d.shape[0]} to {target[1]}x{target[0]}")
            if colour:
                mask2d = np.array(Image.fromarray(mask2d.astype(np.uint8)).resize(target[::-1], Image.NEAREST))
            else:
                mask2d = np.array(Image.fromarray(mask2d.astype(np.int32)).resize(target[::-1], Image.NEAREST))
        mask = mask2d[None]
    else:
        mask_image = sitk.ReadImage(case.mask)
        if mask_image.GetNumberOfComponentsPerPixel() > 1:
            mask_image = sitk.VectorIndexSelectionCast(mask_image, 0)
        if mask_image.GetDimension() == 2:
            mask_image = sitk.JoinSeries(mask_image)
        if sitk.GetArrayViewFromImage(mask_image).shape == tuple(original_shape):
            mask_image.CopyInformation(original)   # same voxels as the image: its geometry
        if info["dim"] == 3:
            mask_image = sitk.Resample(mask_image, images[0], sitk.Transform(), sitk.sitkNearestNeighbor, 0,
                                       mask_image.GetPixelID())
        mask = sitk.GetArrayFromImage(mask_image)
        if mask.shape != sitk.GetArrayViewFromImage(images[0]).shape:
            raise ValueError(f"mask shape {mask.shape} differs from the image {sitk.GetArrayViewFromImage(images[0]).shape}")
    return images, mask, info


def cache_case(case, index, folder, mapping, channels=None):
    """Writes <index>_000c.nii.gz (channels) and <index>.nii.gz (classes 0..K). Returns the case
    metadata (spacing (z, y, x), shape, modality, dim, classes present)."""
    import SimpleITK as sitk
    images, raw, info = read_case(case, channels)
    labels = map_labels(raw, mapping)
    name = f"case_{index:05d}"
    for c, image in enumerate(images):
        image = sitk.Cast(image, sitk.sitkUInt8 if info["rgb"] else sitk.sitkFloat32)
        sitk.WriteImage(image, os.path.join(folder, f"{name}_{c:04d}.nii.gz"))
    label_image = _image_from_array(labels.astype(np.uint8))
    label_image.CopyInformation(images[0])
    sitk.WriteImage(label_image, os.path.join(folder, f"{name}.nii.gz"))
    spacing = list(images[0].GetSpacing())[::-1]
    return {"id": case.id, "group": case.group, "name": name, "dim": info["dim"], "rgb": info["rgb"],
            "channels": len(images),
            "modality": info["modality"], "spacing": spacing, "shape": list(labels.shape),
            "present": sorted(int(v) for v in np.unique(labels)), "notes": case.notes}


def cache_split(cases, folder, mapping, channels=None, log=print, workers=None):
    """Caches every case; returns the metadata of the readable ones and the failures."""
    os.makedirs(folder, exist_ok=True)
    workers = workers or min(8, os.cpu_count() or 1)
    ready, failed = [], []

    def one(item):
        i, case = item
        try:
            return cache_case(case, i, folder, mapping, channels), None
        except Exception as e:
            return None, f"{case.id}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as pool:  # SimpleITK and PIL release the GIL
        for n, (meta, error) in enumerate(pool.map(one, enumerate(cases)), start=1):
            if meta:
                ready.append(meta)
            else:
                failed.append(error)
            if n % 50 == 0:
                log(f"Prepared {n}/{len(cases)} cases")
    return ready, failed


def load_cached(folder, meta, channels):
    """(image (C, z, y, x) float32, labels (z, y, x) uint8) of a cached case."""
    import SimpleITK as sitk
    arrays = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, f"{meta['name']}_{c:04d}.nii.gz")))
              for c in range(channels)]
    labels = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, f"{meta['name']}.nii.gz")))
    return np.stack(arrays).astype(np.float32), labels.astype(np.uint8)


# ---------------------------------------------------------------------------------------
# Summary of an upload
# ---------------------------------------------------------------------------------------

def _dicom_size(series):
    """In-plane size (columns, rows) of a DICOM series, from the header of one of its slices."""
    import pydicom
    for name in sorted(os.listdir(series["path"])):
        try:
            ds = pydicom.dcmread(os.path.join(series["path"], name), stop_before_pixels=True,
                                 specific_tags=["SeriesInstanceUID", "Rows", "Columns"])
        except Exception:
            continue
        if str(ds.get("SeriesInstanceUID", "")) == series.get("uid") and "Rows" in ds:
            return [int(ds.Columns), int(ds.Rows)]
    return None


def _case_geometry(case):
    """(dim, size (x, y[, z]), modality, rgb) of a case from the header of its reference channel
    (the first in the default order: T2-weighted series first)."""
    from Helpers.image3d.volumes import channel_order, series_info
    first = min(case.sources, key=lambda s: channel_order(s["name"])) if case.folder else case.sources[0]
    if first["kind"] == "file" and mio.file_kind(first["path"]) == "raster":
        from PIL import Image
        with Image.open(first["path"]) as image:
            rgb = image.mode in ("RGB", "RGBA", "P") and len(case.sources) == 1
            return 2, list(image.size), "", rgb
    if first["kind"] == "dicom":
        return 3, _dicom_size(first), first.get("modality") or "", False
    info = series_info(first)
    dim = 3 if (info["slices"] or 1) > 1 else 2
    return dim, info["size"], info["modality"] or "", False


def summarize(train_folder, test_folder):
    """Everything the configuration page needs, and the blocking errors of the upload."""
    errors, warnings = [], []
    splits = {}
    for split, folder in (("train", train_folder), ("test", test_folder)):
        try:
            splits[split] = scan_split(folder)
        except SegmentationDataError as e:
            errors.append(f"{split.capitalize()}.zip: {e}")
            splits[split] = ([], {"masks_without_image": [], "images_without_mask": []}, "folders")
    (train, train_lonely, train_format), (test, test_lonely, _) = splits["train"], splits["test"]
    if errors:
        return {"errors": errors, "warnings": warnings}
    if not train:
        errors.append("No image with a mask was found in Train.zip.")
    if not test:
        errors.append("No image with a mask was found in Test.zip.")
    for split, lonely in (("Train", train_lonely), ("Test", test_lonely)):
        if lonely["images_without_mask"]:
            warnings.append(f"{split}.zip: {len(lonely['images_without_mask'])} image(s) without a mask are ignored "
                            f"(e.g. {lonely['images_without_mask'][0]}).")
        if lonely["masks_without_image"]:
            warnings.append(f"{split}.zip: {len(lonely['masks_without_image'])} mask(s) without an image are ignored "
                            f"(e.g. {lonely['masks_without_image'][0]}).")
    if errors:
        return {"errors": errors, "warnings": warnings}

    # Classes: the values of every mask
    def values(case):
        try:
            return case, mask_values(case.mask), None
        except Exception as e:
            return case, None, f"{os.path.basename(case.mask)} ({e})"
    totals, presence, test_presence, unreadable = Counter(), Counter(), Counter(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for split, cases in (("train", train), ("test", test)):
            for case, counts, error in pool.map(values, cases):
                if error:
                    unreadable.append(error)
                    continue
                if split == "train":
                    totals.update(counts)
                    presence.update(counts.keys())
                else:
                    test_presence.update(counts.keys())
                    for value in counts:
                        totals.setdefault(value, 0)
    if unreadable:
        warnings.append(f"{len(unreadable)} mask(s) could not be read and their cases will be skipped (e.g. {unreadable[0]}).")
    try:
        mapping = label_mapping(totals, _class_names(split_root(train_folder)))
    except SegmentationDataError as e:
        return {"errors": [str(e)], "warnings": warnings}
    raw = [tuple(v) if isinstance(v, list) else v for v in mapping["values"]]
    classes = [{"index": i, "name": name, "value": mapping["values"][i], "train_cases": presence.get(v, 0),
                "test_cases": test_presence.get(v, 0), "pixels": mapping["counts"][i]}
               for i, (name, v) in enumerate(zip(mapping["classes"], raw))]
    if len(classes) < 2:
        errors.append("The masks hold no structure (only background): nothing to segment.")
    absent = [c["name"] for c in classes[1:] if c["train_cases"] == 0]
    if absent:
        warnings.append("Classes found only in Test.zip (never learned): " + ", ".join(absent) + ".")
    missing = [c["name"] for c in classes[1:] if c["test_cases"] == 0]
    if missing:
        warnings.append("Classes without test cases (their test metrics cannot be computed): " + ", ".join(missing) + ".")

    # Dimension, channels, sizes
    geometry = [_case_geometry(case) for case in train[:40] + test[:10]]
    dims = Counter(g[0] for g in geometry)
    dim = dims.most_common(1)[0][0]
    if len(dims) > 1:
        warnings.append("The cases mix 2D images and 3D volumes: they are all handled as "
                        f"{dim}D ({'volumes' if dim == 3 else 'images'}).")
    folder_cases = [c for c in train if c.folder]
    series = Counter(s["name"] for c in folder_cases for s in c.sources)
    test_series = Counter(s["name"] for c in test if c.folder for s in c.sources)
    from Helpers.image3d.volumes import channel_order
    channels = sorted([n for n, k in series.items() if k == len(folder_cases)], key=channel_order) if folder_cases else []
    if folder_cases and not channels:
        warnings.append("No series is present in every training case: give the series the same names in every case "
                        "folder (e.g. t2, adc), or choose the series on the next page.")
    groups = len({c.group for c in train})
    sizes = [g[1] for g in geometry if g[1]]
    modalities = Counter(g[2] for g in geometry if g[2])
    rgb = any(g[3] for g in geometry)
    kinds = Counter(mio.file_kind(s["path"]) if s["kind"] == "file" else "dicom" for c in train + test for s in c.sources)
    return {
        "dim": dim,
        "format": train_format,
        "train_cases": len(train),
        "test_cases": len(test),
        "groups": groups,
        "grouped": groups < len(train),
        "max_folds": max(2, min(10, groups)),
        "classes": classes,
        "mapping": mapping,
        "series": [{"name": n, "train": series.get(n, 0), "test": test_series.get(n, 0)}
                   for n in sorted(set(series) | set(test_series), key=channel_order)],
        "channels": channels,
        "folder_cases": bool(folder_cases),
        "rgb": rgb,
        "kinds": dict(kinds),
        "modalities": dict(modalities),
        "has_ct": "CT" in modalities,
        "size_range": [min(min(s[:2]) for s in sizes), max(max(s[:2]) for s in sizes)] if sizes else None,
        "errors": errors,
        "warnings": warnings,
    }


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
