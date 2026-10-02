"""Which automator fits a dataset: a quick look at the training data (the CSV columns, or the names
and kinds of the files of a zip or folder), without reading the images. Python 3.9 compatible."""
import json
import os
import zipfile
from pathlib import Path

from .paths import resolve_data_path

RASTER = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_ENTRIES = 50000


def _kind(name):
    lower = name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        return "nifti"
    suffix = os.path.splitext(lower)[1]
    if suffix in RASTER:
        return "raster"
    if suffix in (".dcm", ".dicom") or (not suffix and "/" in name):
        return "dicom"
    return {".json": "json", ".xml": "xml", ".csv": "csv", ".txt": "txt", ".yaml": "yaml", ".yml": "yaml"}.get(suffix, "other")


def _entries(path):
    """Relative file names of a zip or a folder (hidden files and __MACOSX left out)."""
    names = []
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/")]
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                names.append(os.path.relpath(os.path.join(dirpath, name), path).replace(os.sep, "/"))
                if len(names) >= MAX_ENTRIES:
                    break
    names = [n for n in names if "__MACOSX" not in n and not os.path.basename(n).startswith(".")]
    # A single folder wrapping everything (Train/...) is not part of the layout
    while names and len({n.split("/", 1)[0] for n in names}) == 1 and all("/" in n for n in names):
        names = [n.split("/", 1)[1] for n in names]
    return names


def _read_json(path, name):
    try:
        if path.is_file():
            with zipfile.ZipFile(path) as archive:
                candidates = [n for n in archive.namelist() if n.endswith(name)]
                return json.loads(archive.read(candidates[0])) if candidates else None
        found = next(path.rglob(name), None)
        return json.loads(found.read_text()) if found else None
    except Exception:
        return None


def _csv(path):
    import pandas as pd
    frame = pd.read_csv(path, nrows=2000)
    columns = list(frame.columns)
    ids = [c for c in ("ID", "patient_id") if c in columns]
    evidence = {"columns": columns[:50], "rows_read": len(frame)}
    features = len(columns) - len(ids) - (1 if "Target" in columns else 0)
    if "Target" not in columns:
        if features < 1:
            return [], evidence, ["No feature column: clustering needs at least one column besides ID."]
        return [{"automator": "clustering", "confidence": "high",
                 "reason": f"no Target column: unsupervised clustering of the rows ({features} feature columns)"}], evidence, \
            ["Tabular classification and forecasting need a Target column (the label, or the value to forecast)."]
    candidates = []
    if "Time" in columns and ids:
        repeated = frame[ids[0]].duplicated().any()
        candidates.append({"automator": "time-series-forecasting", "confidence": "high" if repeated else "medium",
                           "reason": f"long format: {ids[0]}, Time and Target columns"
                                     + ("" if repeated else " (but each ID appears once: a time series has several rows)")})
    target = frame["Target"]
    integer_classes = pd.api.types.is_numeric_dtype(target) and (target.dropna() % 1 == 0).all() and target.nunique() <= 50
    if integer_classes and not candidates:
        candidates.append({"automator": "tabular", "confidence": "high",
                           "reason": f"a Target column with {target.nunique()} classes and {features} feature columns"})
    elif integer_classes:
        candidates.append({"automator": "tabular", "confidence": "low",
                           "reason": "also possible: each row classified on its own (Time would then be a feature)"})
    text_classes = not pd.api.types.is_numeric_dtype(target) and target.nunique() <= 50
    if not candidates and text_classes:
        candidates.append({"automator": "clustering", "confidence": "medium",
                           "reason": f"a Target of {target.nunique()} text classes: clustering evaluated against them "
                                     "(tabular classification needs classes numbered 0..K-1)"})
    elif integer_classes or text_classes:
        candidates.append({"automator": "clustering", "confidence": "low",
                           "reason": "also possible: clustering the rows, the clusters evaluated against Target"})
    elif not candidates:
        candidates.append({"automator": "clustering", "confidence": "low",
                           "reason": "Target holds continuous values (left out by clustering)"})
    notes = [] if candidates[0]["confidence"] in ("high", "medium") else \
        ["Target is not made of class numbers 0..K-1 and there is no ID/Time series layout."]
    return candidates, evidence, notes


def _archive(path):
    names = _entries(path)
    lower = [n.lower() for n in names]
    kinds = {}
    for n in names:
        kinds[_kind(n)] = kinds.get(_kind(n), 0) + 1
    tops = sorted({n.split("/", 1)[0] for n in names if "/" in n})
    evidence = {"files": len(names), "kinds": kinds, "top_folders": tops[:30], "sample": names[:10]}
    candidates, notes = [], []
    images = kinds.get("raster", 0) + kinds.get("nifti", 0) + kinds.get("dicom", 0)
    top = {t.lower() for t in tops}

    # Segmentation: images/ + masks/, or the nnU-Net raw layout
    nnunet = {"imagestr", "labelstr"} <= top or (_read_json(path, "dataset.json") or {}).get("channel_names") is not None
    mask_dirs = top & {"masks", "mask", "segmentations", "segmentation", "ground_truth", "groundtruth", "gt"}
    label_dirs = top & {"labels", "label"}
    image_dirs = top & {"images", "image", "imgs", "img"}
    label_images = sum(1 for n in lower if n.split("/", 1)[0] in label_dirs and _kind(n) in ("raster", "nifti"))
    label_texts = sum(1 for n in lower if n.split("/", 1)[0] in label_dirs and n.endswith(".txt") and "classes" not in n)
    if nnunet:
        candidates.append({"automator": "image-segmentation", "confidence": "high", "reason": "nnU-Net raw layout (imagesTr/, labelsTr/, dataset.json)"})
    elif image_dirs and (mask_dirs or label_images > label_texts):
        candidates.append({"automator": "image-segmentation", "confidence": "high",
                           "reason": f"{sorted(image_dirs)[0]}/ and {sorted(mask_dirs or label_dirs)[0]}/ folders with images (masks)"})
        candidates.append({"automator": "object-detection", "confidence": "low",
                           "reason": "masks can also be read as boxes (each connected region of a label)"})

    # Detection: COCO JSON, YOLO labels, Pascal VOC XML, CSV boxes
    coco = next((n for n in names if n.lower().endswith(".json") and "dataset.json" not in n.lower()), None)
    if coco:
        data = _read_json(path, os.path.basename(coco)) or {}
        if isinstance(data, dict) and {"images", "annotations"} <= set(data):
            candidates.append({"automator": "object-detection", "confidence": "high", "reason": f"COCO annotations ({coco})"})
    if image_dirs and label_texts and label_texts >= label_images:
        candidates.append({"automator": "object-detection", "confidence": "high", "reason": "YOLO layout (images/ and labels/*.txt)"})
    if kinds.get("xml", 0) and kinds.get("xml", 0) >= 0.5 * max(1, kinds.get("raster", 0)):
        candidates.append({"automator": "object-detection", "confidence": "high", "reason": "Pascal VOC (.xml next to the images)"})
    for name in names:
        if name.lower().endswith(".csv"):
            try:
                if path.is_file():
                    with zipfile.ZipFile(path) as archive:
                        header = archive.read([n for n in archive.namelist() if n.endswith(name)][0])[:400].decode(errors="ignore")
                else:
                    header = next(path.rglob(os.path.basename(name))).read_text(errors="ignore")[:400]
            except Exception:
                continue
            first = header.splitlines()[0].lower() if header else ""
            if {"x_min", "y_min", "x_max", "y_max"} <= {c.strip() for c in first.split(",")}:
                candidates.append({"automator": "object-detection", "confidence": "high",
                                   "reason": f"CSV boxes ({name}: x_min, y_min, x_max, y_max" + (", z_min, z_max: 3D" if "z_min" in first else "") + ")"})
                break

    # Classification: one folder per class holding images
    if not candidates and len(tops) >= 2 and images:
        per_class = {t: sum(1 for n in names if n.startswith(t + "/") and _kind(n) in ("raster", "nifti", "dicom")) for t in tops}
        if sum(1 for v in per_class.values() if v) >= 2:
            volumes = not kinds.get("raster") and (kinds.get("nifti") or kinds.get("dicom"))
            candidates.append({"automator": "image-classification", "confidence": "high",
                               "reason": f"one folder per class ({', '.join(tops[:6])}) holding "
                                         + ("DICOM series / NIfTI volumes: 3D studies by default" if volumes else "images"),
                               "dim": 3 if volumes else 2})
    if not candidates:
        notes.append("No known layout: see get_data_contract for the layouts of the image automators (class folders; "
                     "images/ + masks/; images with COCO, YOLO, VOC or CSV boxes).")
    # One entry per automator, the most confident first
    order = {"high": 0, "medium": 1, "low": 2}
    best = {}
    for c in candidates:
        if c["automator"] not in best or order[c["confidence"]] < order[best[c["automator"]]["confidence"]]:
            best[c["automator"]] = c
    return sorted(best.values(), key=lambda c: order[c["confidence"]]), evidence, notes


def inspect(train):
    """{suggested_automator, candidates: [{automator, confidence, reason}], evidence, notes}."""
    path = resolve_data_path(train)
    if path.is_file() and path.suffix.lower() == ".csv":
        candidates, evidence, notes = _csv(path)
        evidence["format"] = "csv"
    elif (path.is_file() and zipfile.is_zipfile(path)) or path.is_dir():
        candidates, evidence, notes = _archive(path)
        evidence["format"] = "zip" if path.is_file() else "folder"
    else:
        candidates, evidence, notes = [], {"format": path.suffix or "unknown"}, [
            "Give a CSV file (tabular, forecasting, clustering), or a zip file or folder (image automators)."]
    suggested = candidates[0]["automator"] if candidates and candidates[0]["confidence"] in ("high", "medium") else None
    return {"path": str(path), "suggested_automator": suggested, "candidates": candidates, "evidence": evidence, "notes": notes}
