"""Small synthetic medical images in every supported format, for the tests.

Images of the lesion class contain a bright disc, the others do not.
"""
import io
import os
import zipfile

import numpy as np
from PIL import Image


def _disc(size, rng, lesion):
    """A noisy 2D pattern in [0, 1], with a bright disc for lesions."""
    y, x = np.mgrid[:size, :size]
    image = 0.3 + 0.1 * rng.standard_normal((size, size))
    if lesion:
        cy, cx = rng.integers(size // 4, 3 * size // 4, 2)
        image[(y - cy) ** 2 + (x - cx) ** 2 < (size // 6) ** 2] += 0.6
    return np.clip(image, 0, 1)


def _dicom_dataset(pixels, modality="CT", photometric="MONOCHROME2", frames=1, **extra):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.PatientID = "SYNTHETIC"
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 3 if photometric in ("RGB", "YBR_FULL_422") else 1
    ds.Rows, ds.Columns = pixels.shape[-3:-1] if ds.SamplesPerPixel == 3 else pixels.shape[-2:]
    if frames > 1:
        ds.NumberOfFrames = frames
    if pixels.dtype == np.uint8:
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    else:
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    if ds.SamplesPerPixel == 3:
        ds.PlanarConfiguration = 0
    for key, value in extra.items():
        setattr(ds, key, value)
    ds.PixelData = pixels.tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    return ds


def write_image(path, kind, lesion, rng, size=48):
    """Writes one synthetic image of the given kind and returns its path."""
    unit = _disc(size, rng, lesion)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if kind == "png8":
        Image.fromarray((unit * 255).astype(np.uint8)).save(path + ".png")
        return path + ".png"
    if kind == "png16":
        Image.fromarray((unit * 4000).astype(np.uint16)).save(path + ".png")
        return path + ".png"
    if kind == "jpeg_rgb":
        rgb = np.stack([unit, unit * 0.8, unit * 0.6], axis=-1)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(path + ".jpg", quality=95)
        return path + ".jpg"
    if kind in ("dicom_ct", "dicom_noext"):
        hu = (unit * 2000 - 1000).astype(np.int16)  # Hounsfield units
        stored = (hu + 1024).astype(np.int16)
        ds = _dicom_dataset(stored, "CT", RescaleSlope=1, RescaleIntercept=-1024, WindowCenter=40, WindowWidth=400)
        target = path + ".dcm" if kind == "dicom_ct" else path
        ds.save_as(target, write_like_original=False)
        return target
    if kind == "dicom_mono1":
        stored = ((1 - unit) * 4000).astype(np.int16)  # MONOCHROME1: high values are dark
        ds = _dicom_dataset(stored, "CR", photometric="MONOCHROME1")
        ds.save_as(path + ".dcm", write_like_original=False)
        return path + ".dcm"
    if kind == "dicom_multiframe":
        frames = np.stack([_disc(size, rng, lesion and i == 1) for i in range(3)])
        ds = _dicom_dataset((frames * 3000).astype(np.int16), "MR", frames=3)
        ds.save_as(path + ".dcm", write_like_original=False)
        return path + ".dcm"
    if kind == "dicom_jpeg_rgb":
        from pydicom.encaps import encapsulate
        from pydicom.uid import JPEGBaseline8Bit
        rgb = (np.stack([unit, unit * 0.5, unit * 0.2], axis=-1) * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
        ds = _dicom_dataset(rgb, "XC", photometric="YBR_FULL_422")
        ds.file_meta.TransferSyntaxUID = JPEGBaseline8Bit
        ds.PixelData = encapsulate([buffer.getvalue()])
        ds["PixelData"].is_undefined_length = True
        ds.save_as(path + ".dcm", write_like_original=False)
        return path + ".dcm"
    if kind == "nifti":
        import nibabel as nib
        volume = np.stack([_disc(size, rng, lesion and i == 2) for i in range(5)], axis=-1)
        nib.save(nib.Nifti1Image((volume * 1000).astype(np.int16), np.eye(4)), path + ".nii.gz")
        return path + ".nii.gz"
    raise ValueError(kind)


KINDS = ["png8", "png16", "jpeg_rgb", "dicom_ct", "dicom_noext", "dicom_mono1", "dicom_multiframe", "nifti"]


def make_split(root, per_class, kinds=KINDS, seed=0, classes=("lesion", "normal")):
    """A folder per class, cycling through the given file kinds. Images of the first class
    contain a lesion."""
    rng = np.random.default_rng(seed)
    for label in classes:
        for i in range(per_class):
            kind = kinds[i % len(kinds)]
            write_image(os.path.join(root, label, f"{label}_{i:03d}"), kind, label == classes[0], rng)
    return root


def zip_folder(folder, zip_path, prefix=None):
    """Zips a folder; ``prefix`` adds a top-level folder inside the archive (e.g. "Train/")."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for dirpath, _, files in os.walk(folder):
            for name in files:
                full = os.path.join(dirpath, name)
                arcname = os.path.relpath(full, folder)
                archive.write(full, os.path.join(prefix, arcname) if prefix else arcname)
    return zip_path
