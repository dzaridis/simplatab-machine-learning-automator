"""Image datasets: safe extraction of the uploaded zip files, scan of the class folders,
checks shown before the run, and conversion of every image into a small cached PNG.

Expected layout of Train.zip and Test.zip: one folder per class, e.g.
    benign/case_001.dcm, benign/patient_07/slice.png, malignant/...
A single wrapping folder (e.g. "Train/") is ignored. Sub-folders inside a class folder are
allowed (e.g. one folder per patient): every supported file is one sample.
"""
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import io as mio

MAX_ZIP_BYTES = 5 * 1024 ** 3  # upload limit per zip file
MAX_UNCOMPRESSED_BYTES = 40 * 1024 ** 3  # protection against zip bombs
MAX_FILES = 500_000
CACHE_SIZE = 256  # side of the cached images; the networks see 224 x 224 crops/resizes
SAMPLE_PER_KIND = 100  # headers read per file kind for the summary

# Class names that usually denote the negative class of a binary problem
NEGATIVE_NAMES = {"normal", "negative", "healthy", "benign", "control", "controls", "absent", "no",
                  "none", "0", "neg", "nofinding", "no_finding", "no finding", "non-cancer", "noncancer"}


class DatasetError(ValueError):
    """A problem with the uploaded data, shown to the user as is."""


def extract_zip(zip_path, destination):
    """Extracts a zip file, refusing unsafe paths and archives that expand beyond the limits.
    macOS metadata and hidden files are skipped."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise DatasetError(f"{os.path.basename(zip_path)} is not a valid zip file.")
    with archive:
        members, total = [], 0
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if name.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
                raise DatasetError(f"Unsafe path in {os.path.basename(zip_path)}: {info.filename}")
            if not parts or parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts):
                continue
            members.append((info, parts))
            total += info.file_size
        if len(members) > MAX_FILES:
            raise DatasetError(f"{os.path.basename(zip_path)} contains more than {MAX_FILES:,} files.")
        if total > MAX_UNCOMPRESSED_BYTES:
            raise DatasetError(f"{os.path.basename(zip_path)} expands to more than "
                               f"{MAX_UNCOMPRESSED_BYTES // 1024 ** 3} GB.")
        root = os.path.realpath(destination)
        for info, parts in members:
            target = os.path.realpath(os.path.join(root, *parts))
            if not target.startswith(root + os.sep):
                raise DatasetError(f"Unsafe path in {os.path.basename(zip_path)}: {info.filename}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)


def _visible(entries):
    return [e for e in entries if not e.startswith(".") and e != "__MACOSX"]


def _holds_files(folder):
    return any(os.path.isfile(os.path.join(folder, e)) for e in _visible(os.listdir(folder)))


def dataset_root(folder):
    """Skips wrapping folders (e.g. "Train/") down to the folder that holds the class folders.
    A single folder that directly contains files is a class folder, not a wrapper."""
    current = folder
    while True:
        entries = _visible(os.listdir(current))
        dirs = [e for e in entries if os.path.isdir(os.path.join(current, e))]
        files = [e for e in entries if os.path.isfile(os.path.join(current, e))]
        if len(dirs) == 1 and not files and not _holds_files(os.path.join(current, dirs[0])):
            current = os.path.join(current, dirs[0])
        else:
            return current


def scan_split(folder):
    """The samples of a split: [{"path", "class", "kind", "size"}], and the ignored files."""
    root = dataset_root(folder)
    samples, ignored = [], []
    for entry in sorted(_visible(os.listdir(root))):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            ignored.append(entry)
            continue
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames[:] = sorted(_visible(dirnames))
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                kind = mio.file_kind(path)
                if kind is None:
                    if not name.startswith("."):
                        ignored.append(os.path.relpath(path, root))
                    continue
                samples.append({"path": path, "class": entry, "kind": kind, "size": os.path.getsize(path)})
    return samples, ignored


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_duplicates(train, test):
    """Files present in both splits (same content): a leak of test data into training.
    Only files of equal size are hashed."""
    train_sizes = {s["size"] for s in train}
    candidates = [s for s in test if s["size"] in train_sizes]
    if not candidates:
        return []
    sizes = {s["size"] for s in candidates}
    train_hashes = {_md5(s["path"]): s for s in train if s["size"] in sizes}
    return [{"test": s["path"], "train": train_hashes[h]["path"]}
            for s in candidates for h in [_md5(s["path"])] if h in train_hashes]


def default_positive_class(classes):
    """For two classes, the class that is not a usual negative name (else the second one)."""
    if len(classes) != 2:
        return None
    normalised = [c.strip().lower().replace("-", "_") for c in classes]
    negatives = [i for i, name in enumerate(normalised) if name in NEGATIVE_NAMES or name.startswith("no_")]
    if len(negatives) == 1:
        return classes[1 - negatives[0]]
    return classes[1]


def summarize(train_folder, test_folder):
    """Everything the configuration page needs, and the blocking errors of the upload."""
    train, train_ignored = scan_split(train_folder)
    test, test_ignored = scan_split(test_folder)
    classes = sorted({s["class"] for s in train})
    train_counts = Counter(s["class"] for s in train)
    test_counts = Counter(s["class"] for s in test)

    errors, warnings = [], []
    if not train:
        errors.append("No supported image was found in Train.zip. Put the images in one folder per class.")
    elif len(classes) < 2:
        errors.append(f"Train.zip has a single class folder ({classes[0]}): at least two are needed.")
    if not test:
        errors.append("No supported image was found in Test.zip. Put the images in one folder per class.")
    unknown = sorted(set(test_counts) - set(classes))
    if unknown:
        errors.append("Test.zip has class folders that are not in Train.zip: " + ", ".join(unknown) + ".")
    missing = [c for c in classes if c not in test_counts]
    if missing and test:
        warnings.append("Classes without test images (their test metrics cannot be computed): " + ", ".join(missing) + ".")
    if train_counts:
        smallest, largest = min(train_counts.values()), max(train_counts.values())
        if smallest < 10:
            warnings.append(f"The smallest class has {smallest} training image(s): the results will be unreliable.")
        if largest >= 5 * smallest:
            warnings.append(f"The classes are imbalanced ({largest} vs {smallest} images): "
                            "balanced accuracy and AUC are more informative than accuracy.")
    ignored = len(train_ignored) + len(test_ignored)
    if ignored:
        warnings.append(f"{ignored} file(s) are not supported images and will be ignored "
                        f"(e.g. {(train_ignored + test_ignored)[0]}).")

    duplicates = find_duplicates(train, test) if train and test else []
    if duplicates:
        warnings.append(f"{len(duplicates)} test image(s) are identical to training images "
                        f"(e.g. {os.path.basename(duplicates[0]['test'])}): the test metrics will be optimistic.")

    # Header information from a sample of each kind of file
    kinds = Counter(s["kind"] for s in train + test)
    modalities, volumes, high_bit, sizes, unreadable = Counter(), 0, 0, [], []
    by_kind = {}
    for s in train + test:
        by_kind.setdefault(s["kind"], []).append(s["path"])
    for kind, paths in by_kind.items():
        step = max(1, len(paths) // SAMPLE_PER_KIND)
        for path in paths[::step][:SAMPLE_PER_KIND]:
            try:
                info = mio.image_info(path)
            except Exception as e:
                unreadable.append(f"{os.path.basename(path)} ({e})")
                continue
            if info["modality"]:
                modalities[info["modality"]] += 1
            volumes += info["volume"]
            high_bit += info["high_bit_depth"]
            if info["size"]:
                sizes.append(info["size"])
    if unreadable:
        warnings.append(f"Some files could not be read and will be skipped (e.g. {unreadable[0]}).")

    return {
        "classes": classes,
        "positive_class": default_positive_class(classes),
        "class_counts": [{"class": c, "train": train_counts.get(c, 0), "test": test_counts.get(c, 0)} for c in classes],
        "train_images": len(train),
        "test_images": len(test),
        "min_class_count": min(train_counts.values()) if train_counts else 0,
        "kinds": dict(kinds),
        "modalities": dict(modalities),
        "has_ct": "CT" in modalities or kinds.get("nifti", 0) > 0,
        "has_volumes": volumes > 0,
        "has_high_bit_depth": high_bit > 0,
        "size_range": [min(min(s) for s in sizes), max(max(s) for s in sizes)] if sizes else None,
        "duplicates": len(duplicates),
        "ignored": ignored,
        "errors": errors,
        "warnings": warnings,
    }


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _preprocess_one(args):
    source, target, window, volume = args
    try:
        image = mio.letterbox(mio.load_image(source, window, volume), CACHE_SIZE)
        image.save(target)
        return None
    except Exception as e:
        return f"{e}"


def preprocess(samples, cache_folder, window="auto", volume="middle", workers=None, log=print):
    """Converts every sample into a CACHE_SIZE x CACHE_SIZE 8-bit PNG (grayscale or RGB).
    Returns the samples that could be read, with their cached path, and the failures."""
    os.makedirs(cache_folder, exist_ok=True)
    jobs = [(s["path"], os.path.join(cache_folder, f"{i:07d}.png"), window, volume) for i, s in enumerate(samples)]
    workers = workers or min(8, os.cpu_count() or 1)
    ready, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (sample, error) in enumerate(zip(samples, pool.map(_preprocess_one, jobs))):
            if error is None:
                ready.append(dict(sample, cached=jobs[i][1]))
            else:
                failed.append({"path": sample["path"], "error": error})
            if (i + 1) % 500 == 0:
                log(f"Prepared {i + 1}/{len(samples)} images")
    return ready, failed
