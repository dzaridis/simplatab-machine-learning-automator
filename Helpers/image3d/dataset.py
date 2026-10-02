"""Dataset summary (shown on the configuration page) and cache of the 3D automator: every study
is converted once into a float16 (C, D, H, W) array of its aligned series."""
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from Helpers.image.dataset import default_positive_class
from . import volumes

SAMPLE_SERIES = 80  # series whose header is read for the summary


def summarize(train_folder, test_folder):
    """The 3D view of an upload: classes, studies, patients, series names and their counts."""
    train, train_ignored = volumes.scan_split(train_folder)
    test, test_ignored = volumes.scan_split(test_folder)
    classes = sorted({s["class"] for s in train})
    errors, warnings = [], []
    if not train:
        errors.append("No volume (DICOM series, NIfTI or multi-frame DICOM) was found in Train.zip.")
    elif len(classes) < 2:
        errors.append(f"Train.zip has a single class folder ({classes[0]}): at least two are needed.")
    if not test:
        errors.append("No volume (DICOM series, NIfTI or multi-frame DICOM) was found in Test.zip.")
    unknown = sorted({s["class"] for s in test} - set(classes))
    if unknown:
        errors.append("Test.zip has class folders that are not in Train.zip: " + ", ".join(unknown) + ".")

    studies = Counter(s["class"] for s in train)
    test_studies = Counter(s["class"] for s in test)
    patients = {c: len({s["patient"] for s in train if s["class"] == c}) for c in classes}
    series_train = Counter(name for s in train for name in {x["name"] for x in s["series"]})
    series_test = Counter(name for s in test for name in {x["name"] for x in s["series"]})
    channels = volumes.default_channels(train) if train else []
    if train and not channels:
        warnings.append("No series is present in every training study: choose the series to use (the studies "
                        "missing one are skipped), or give the series folders the same names in every study "
                        "(e.g. t2, adc, dwi).")
    if channels and test:
        lacking = sum(volumes.series_of(s, channels) is None for s in test)
        if lacking:
            warnings.append(f"{lacking} of the {len(test)} test studies do not have the series "
                            f"{', '.join(channels)} and would be skipped: give the series the same names in "
                            "Train.zip and Test.zip.")
    shared = {s["patient"] for s in train} & {s["patient"] for s in test}
    if shared:
        warnings.append(f"{len(shared)} patient folder name(s) are in both Train.zip and Test.zip "
                        f"(e.g. {sorted(shared)[0]}): if they are the same patients, the test metrics will be optimistic.")
    if patients:
        smallest = min(patients.values())
        if smallest < 10:
            warnings.append(f"The smallest class has {smallest} training patient(s): the results will be unreliable.")
        if max(studies.values()) >= 5 * min(studies.values()):
            warnings.append(f"The classes are imbalanced ({max(studies.values())} vs {min(studies.values())} studies): "
                            "balanced accuracy and AUC are more informative than accuracy.")
    ignored = train_ignored + test_ignored
    if ignored:
        warnings.append(f"{len(ignored)} file(s) or folder(s) hold no volume and will be ignored (e.g. {ignored[0]}).")

    # Header information from a sample of the series
    every = [x for s in train + test for x in s["series"]]
    step = max(1, len(every) // SAMPLE_SERIES)
    slices, modalities, unreadable = [], Counter(), []
    for series in every[::step][:SAMPLE_SERIES]:
        try:
            info = volumes.series_info(series)
        except Exception as e:
            unreadable.append(f"{series['name']} ({e})")
            continue
        slices.append(info["slices"])
        if info["modality"]:
            modalities[info["modality"]] += 1
    if unreadable:
        warnings.append(f"Some series could not be read and their studies will be skipped (e.g. {unreadable[0]}).")
    volumetric = sum(n >= volumes.MIN_SLICES for n in slices)
    kinds = Counter("nifti" if x["kind"] == "file" and x["path"].lower().endswith((".nii", ".nii.gz")) else "dicom"
                    for x in every)
    return {
        "dim": 3,
        "classes": classes,
        "positive_class": default_positive_class(classes),
        "class_counts": [{"class": c, "train": studies.get(c, 0), "test": test_studies.get(c, 0),
                          "patients": patients.get(c, 0)} for c in classes],
        "train_studies": len(train),
        "test_studies": len(test),
        "train_patients": len({s["patient"] for s in train}),
        "min_class_patients": min(patients.values()) if patients else 0,
        "series": [{"name": name, "train": series_train.get(name, 0), "test": series_test.get(name, 0)}
                   for name in sorted(set(series_train) | set(series_test), key=volumes.channel_order)],
        "single_series": channels == [volumes.SINGLE],
        "default_channels": channels,
        "slices_range": [min(slices), max(slices)] if slices else None,
        "volumetric": bool(slices) and volumetric >= 0.5 * len(slices),
        "modalities": dict(modalities),
        "has_ct": "CT" in modalities or kinds.get("nifti", 0) > 0,
        "kinds": dict(kinds),
        "errors": errors,
        "warnings": warnings,
    }


def _cache_one(args):
    series, target, shape, crop, window = args
    try:
        array, _ = volumes.load_study(series, shape, crop, window)
        np.save(target, array.astype(np.float16))
        return None
    except Exception as e:
        return f"{e}"


def cache_studies(studies, channels, shape, crop, window, folder, log=print, workers=None):
    """Converts the studies into cached arrays. Returns the studies that could be read (with their
    "cached" path), the studies without one of the channels, and the failures."""
    os.makedirs(folder, exist_ok=True)
    jobs, ready, missing, failed = [], [], [], []
    for i, study in enumerate(studies):
        series = volumes.series_of(study, channels)
        if series is None:
            missing.append(study)
            continue
        jobs.append((study, (series, os.path.join(folder, f"{i:06d}.npy"), tuple(shape), crop, window)))
    workers = workers or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:  # SimpleITK and PyTorch release the GIL
        for n, ((study, job), error) in enumerate(zip(jobs, pool.map(_cache_one, [j for _, j in jobs])), start=1):
            if error is None:
                ready.append(dict(study, cached=job[1]))
            else:
                failed.append({"study": study["id"], "error": error})
            if n % 50 == 0:
                log(f"Prepared {n}/{len(jobs)} studies")
    return ready, missing, failed
