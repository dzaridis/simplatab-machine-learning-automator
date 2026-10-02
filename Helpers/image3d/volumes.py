"""Studies and series of the 3D automator: discovery in the uploaded folders, reading,
alignment of the series and conversion into fixed-size arrays.

Layout of Train.zip and Test.zip (the study and series levels are optional):
    <class>/<patient>/<study>/<series>/  slices of a DICOM series
    <class>/<patient>/<study>/<series>.nii.gz  (or a multi-frame DICOM file)
A series is a folder of DICOM slices (one series instance UID; a folder that holds several
series, as in some PACS exports, is a study whose series are told apart by their UID and
named after their description), or a single NIfTI or multi-frame DICOM file. A study is the
folder that holds the series; a patient is the first folder under the class folder (a file
directly in the class folder is a patient with a single study and series).

Each study is one sample. Its series are matched by name across studies (e.g. "t2", "adc"):
the chosen series become the channels of the network input. They are read with SimpleITK,
the first one is reoriented (LPS: axial slices, rows from anterior to posterior) and the
others are resampled onto its grid in patient coordinates, so that the channels are aligned
even when the series have different fields of view or resolutions. Each channel is scaled
to [0, 1] (a CT window, the DICOM window or the 0.5-99.5 percentiles), the volume is cropped
in-plane around its centre (optional) and resized to depth x height x width.
"""
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from Helpers.image import io as mio

SINGLE = "*"  # channel of studies made of a single series, whatever its name
SHAPES = [(32, 128, 128), (64, 128, 128), (64, 192, 192), (96, 160, 160), (128, 128, 128)]
CROPS = [1.0, 0.75, 0.5]  # in-plane field of view kept around the centre
MIN_SLICES = 3  # fewer slices: a 2D image rather than a volume


def series_name(text):
    """Normalised series name: lower case, separators as underscores (e.g. "T2 TSE" -> "t2_tse")."""
    text = re.sub(r"\.(nii\.gz|nii|dcm|dicom)$", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "series"


def _visible(entries):
    return sorted(e for e in entries if not e.startswith(".") and e != "__MACOSX")


def _dicom_header(path):
    import pydicom
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True,
                             specific_tags=["SeriesInstanceUID", "SeriesDescription", "SeriesNumber",
                                            "NumberOfFrames", "Modality"])
    except Exception:
        return None
    return {"uid": str(ds.get("SeriesInstanceUID", "") or ""), "description": str(ds.get("SeriesDescription", "") or ""),
            "number": str(ds.get("SeriesNumber", "") or ""), "frames": int(ds.get("NumberOfFrames", 1) or 1),
            "modality": str(ds.get("Modality", "") or "").upper()}


def _folder_series(folder, patient_dir, pool):
    """The series held directly by a folder, and the study folder they belong to."""
    files = [os.path.join(folder, e) for e in _visible(os.listdir(folder)) if os.path.isfile(os.path.join(folder, e))]
    series, study_of = [], {}
    dicoms = []
    for path in files:
        kind = mio.file_kind(path)
        if kind == "nifti":
            series.append({"name": series_name(os.path.basename(path)), "kind": "file", "path": path,
                           "modality": "NIFTI", "slices": None})
            study_of[path] = folder
        elif kind == "dicom":
            dicoms.append(path)
    headers = list(pool.map(_dicom_header, dicoms)) if dicoms else []
    groups = {}
    for path, header in zip(dicoms, headers):
        if header is None:
            continue
        if header["frames"] > 1:  # multi-frame file: a series on its own
            series.append({"name": series_name(os.path.basename(path)), "kind": "file", "path": path,
                           "modality": header["modality"], "slices": header["frames"]})
            study_of[path] = folder
        else:
            groups.setdefault(header["uid"], []).append(header)
    if groups:
        several = len(groups) > 1
        # A folder with one series is a series folder (its study is the parent folder, unless it is
        # the patient folder); a folder with several series is a study folder
        study = folder if several or os.path.samefile(folder, patient_dir) else os.path.dirname(folder)
        names = Counter()
        for uid, members in groups.items():
            if several:
                name = series_name(members[0]["description"] or f"series_{members[0]['number'] or len(names) + 1}")
            else:
                name = series_name(os.path.basename(folder))
            names[name] += 1
            if names[name] > 1:
                name = f"{name}_{names[name]}"
            key = f"{folder}#{uid}"
            series.append({"name": name, "kind": "dicom", "path": folder, "uid": uid,
                           "modality": members[0]["modality"], "slices": len(members)})
            study_of[key] = study
    return series, study_of


def scan_split(folder):
    """The studies of a split: [{"id", "patient", "class", "series": [...]}], and the ignored files."""
    from Helpers.image.dataset import dataset_root
    root = dataset_root(folder)
    studies, ignored = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for class_name in _visible(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                ignored.append(class_name)
                continue
            for entry in _visible(os.listdir(class_dir)):
                path = os.path.join(class_dir, entry)
                if os.path.isfile(path):
                    kind = mio.file_kind(path)
                    header = _dicom_header(path) if kind == "dicom" else None
                    if kind == "nifti" or (header and header["frames"] > 1):
                        studies.append({"id": f"{class_name}/{entry}", "patient": series_name(entry), "class": class_name,
                                        "series": [{"name": "volume", "kind": "file", "path": path,
                                                    "modality": header["modality"] if header else "NIFTI",
                                                    "slices": header["frames"] if header else None}]})
                    else:
                        ignored.append(f"{class_name}/{entry}")
                    continue
                by_study = {}
                for dirpath, dirnames, _ in os.walk(path):
                    dirnames[:] = _visible(dirnames)
                    series, study_of = _folder_series(dirpath, path, pool)
                    for item in series:
                        key = f"{item['path']}#{item['uid']}" if item["kind"] == "dicom" else item["path"]
                        by_study.setdefault(study_of[key], []).append(item)
                for study_dir in sorted(by_study):
                    items = by_study[study_dir]
                    names = Counter()
                    for item in items:  # same name twice in a study (e.g. two t2 files in sub-folders)
                        names[item["name"]] += 1
                        if names[item["name"]] > 1:
                            item["name"] = f"{item['name']}_{names[item['name']]}"
                    studies.append({"id": os.path.relpath(study_dir, root).replace(os.sep, "/"), "patient": entry,
                                    "class": class_name, "series": sorted(items, key=lambda s: s["name"])})
                if not by_study:
                    ignored.append(f"{class_name}/{entry}")
    return studies, ignored


def default_channels(studies):
    """The series present in every study (most frequent first), or the single-series channel."""
    if studies and all(len(s["series"]) == 1 for s in studies):
        names = {s["series"][0]["name"] for s in studies}
        return [next(iter(names))] if len(names) == 1 else [SINGLE]
    counts = Counter(name for s in studies for name in {x["name"] for x in s["series"]})
    common = sorted((name for name, n in counts.items() if n == len(studies)), key=channel_order)
    return common[:3]


def channel_order(name):
    """Default order of the channels: T2-weighted series first (usually the finest grid, used as
    the reference), then by name."""
    return (not name.startswith("t2"), name)


def series_of(study, channels):
    """The series of a study for each channel, or None when one is missing."""
    if channels == [SINGLE]:
        return study["series"][:1] if len(study["series"]) == 1 else None
    by_name = {s["name"]: s for s in study["series"]}
    chosen = [by_name.get(name) for name in channels]
    return None if any(s is None for s in chosen) else chosen


# ---------------------------------------------------------------------------------------
# Reading and preprocessing
# ---------------------------------------------------------------------------------------

def _metadata(source, key, index=None):
    try:
        if index is None:
            return source.GetMetaData(key).strip() if source.HasMetaDataKey(key) else None
        return source.GetMetaData(index, key).strip() if source.HasMetaDataKey(index, key) else None
    except Exception:
        return None


def read_series(series):
    """(SimpleITK image as float32 3D, header info {"modality", "window"})."""
    import SimpleITK as sitk
    header = {"modality": series.get("modality") or "", "window": None}
    if series["kind"] == "dicom":
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(series["path"], series["uid"])
        if not files:
            raise ValueError(f"no DICOM slice of series {series['name']}")
        reader.SetFileNames(files)
        reader.MetaDataDictionaryArrayUpdateOn()
        image = reader.Execute()
        get = lambda key: _metadata(reader, key, 0)  # noqa: E731
    else:
        image = sitk.ReadImage(series["path"])
        get = lambda key: _metadata(image, key)  # noqa: E731
    modality = get("0008|0060")
    if modality:
        header["modality"] = modality.upper()
    center, width = get("0028|1050"), get("0028|1051")
    if center and width:
        header["window"] = (mio._first_value(center.split("\\")), mio._first_value(width.split("\\")))
    if image.GetNumberOfComponentsPerPixel() > 1:  # colour: mean of the components
        components = [sitk.VectorIndexSelectionCast(image, i, sitk.sitkFloat32)
                      for i in range(image.GetNumberOfComponentsPerPixel())]
        image = sum(components[1:], components[0]) / float(len(components))
    if image.GetDimension() == 4:  # time series or several volumes: the first one
        size = list(image.GetSize())
        size[3] = 0
        image = sitk.Extract(image, size, [0, 0, 0, 0])
    if image.GetDimension() == 2:
        image = sitk.JoinSeries(image)
    return sitk.Cast(image, sitk.sitkFloat32), header


def to_unit(values, window, header):
    """Scales a channel to [0, 1]: a CT window when chosen (CT DICOM or NIfTI), the window of the
    DICOM header in automatic mode for CT, else the 0.5-99.5 percentiles of the volume."""
    modality = header.get("modality", "")
    if window in mio.CT_WINDOWS and modality in ("CT", "NIFTI"):
        return mio._window(values, *mio.CT_WINDOWS[window]).astype(np.float32)
    if window == "auto" and modality == "CT" and header.get("window") and header["window"][0] is not None \
            and header["window"][1]:
        return mio._window(values, *header["window"]).astype(np.float32)
    return mio._percentiles(values).astype(np.float32)


def resize(array, shape, crop=1.0):
    """(C, z, y, x) -> (C, D, H, W): central in-plane crop, then trilinear resizing."""
    import torch
    import torch.nn.functional as F
    if crop < 1.0:
        _, _, rows, columns = array.shape
        h, w = max(1, int(round(rows * crop))), max(1, int(round(columns * crop)))
        top, left = (rows - h) // 2, (columns - w) // 2
        array = array[:, :, top:top + h, left:left + w]
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None]
    return F.interpolate(tensor, size=tuple(shape), mode="trilinear", align_corners=False)[0].numpy()


def load_study(series_list, shape, crop=1.0, window="auto"):
    """The aligned channels of a study as a float32 (C, D, H, W) array in [0, 1], and the shape
    and spacing (z, y, x) of the reference series."""
    import SimpleITK as sitk
    reference, header = read_series(series_list[0])
    reference = sitk.DICOMOrient(reference, "LPS")
    channels = [to_unit(sitk.GetArrayFromImage(reference), window, header)]
    for series in series_list[1:]:
        image, header = read_series(series)
        background = float(sitk.GetArrayViewFromImage(image).min())
        aligned = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear, background, sitk.sitkFloat32)
        channels.append(to_unit(sitk.GetArrayFromImage(aligned), window, header))
    array = resize(np.stack(channels), shape, crop)
    return array, {"size": list(channels[0].shape), "spacing": list(reference.GetSpacing())[::-1]}


def series_info(series):
    """Cheap information for the summary: number of slices, in-plane size and modality."""
    import SimpleITK as sitk
    if series["kind"] == "dicom":
        return {"slices": series["slices"], "modality": series.get("modality") or None, "size": None}
    reader = sitk.ImageFileReader()
    reader.SetFileName(series["path"])
    reader.ReadImageInformation()
    size = list(reader.GetSize())
    slices = size[2] if len(size) > 2 else 1
    return {"slices": slices, "modality": series.get("modality") or None, "size": size[:2]}
