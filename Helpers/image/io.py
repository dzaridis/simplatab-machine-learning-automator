"""Reads medical and natural images as 8-bit 2D images ready for the networks.

Supported files: PNG, JPEG, BMP and TIFF (including 16-bit), DICOM (including compressed
and multi-frame) and NIfTI (.nii, .nii.gz). Pixel values are mapped to 8 bits with:
- the DICOM modality LUT (e.g. Hounsfield units for CT), then a CT window or the window of
  the DICOM header, or the 0.5-99.5 percentiles of the image;
- MONOCHROME1 images inverted, so that high values are always bright;
- volumes (3D NIfTI, multi-frame DICOM) reduced to one 2D image: middle slice or maximum
  intensity projection.
Images are then padded to a square (keeping their proportions) and resized.
"""
import os

import numpy as np
from PIL import Image

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DICOM_EXTENSIONS = {".dcm", ".dicom", ".dic"}
IGNORED_NAMES = {"thumbs.db", "desktop.ini", "dicomdir"}

# CT windows as (center, width) in Hounsfield units
CT_WINDOWS = {
    "lung": (-600, 1500),
    "soft_tissue": (40, 400),
    "bone": (400, 1800),
    "brain": (40, 80),
}
WINDOW_CHOICES = ["auto"] + list(CT_WINDOWS)
VOLUME_CHOICES = ["middle", "mip"]

_RGB_PHOTOMETRIC = ("RGB", "YBR_FULL", "YBR_FULL_422", "YBR_ICT", "YBR_RCT", "YBR_PARTIAL_420")


def _is_dicom(path):
    try:
        with open(path, "rb") as f:
            return f.read(132)[128:132] == b"DICM"
    except OSError:
        return False


def file_kind(path):
    """'raster', 'dicom' or 'nifti', or None for unsupported files. DICOM files are also
    recognised without an extension (common in PACS exports)."""
    name = os.path.basename(path).lower()
    if name.startswith(".") or name in IGNORED_NAMES:
        return None
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return "nifti"
    extension = os.path.splitext(name)[1]
    if extension in RASTER_EXTENSIONS:
        return "raster"
    if extension in DICOM_EXTENSIONS or _is_dicom(path):
        return "dicom"
    return None


def _window(array, center, width):
    low = center - width / 2.0
    return np.clip((array - low) / max(float(width), 1e-6), 0.0, 1.0)


def _percentiles(array, low=0.5, high=99.5):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((np.nan_to_num(array, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def _reduce_volume(volume, strategy, axis=0):
    """One 2D image from a volume: its middle slice or its maximum intensity projection."""
    if strategy == "mip":
        return volume.max(axis=axis)
    return np.take(volume, volume.shape[axis] // 2, axis=axis)


def _to_uint8(unit):
    return (np.clip(unit, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _first_value(value):
    """A DICOM window value, which may hold several windows (the first one is used)."""
    if value is None:
        return None
    try:
        if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
            value = value[0]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return None


def load_dicom(path, window="auto", volume="middle"):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, convert_color_space

    ds = pydicom.dcmread(path, force=True)
    if "PixelData" not in ds:
        raise ValueError("DICOM file without image data")
    array = ds.pixel_array
    photometric = str(ds.get("PhotometricInterpretation", "MONOCHROME2")).upper()
    frames = int(ds.get("NumberOfFrames", 1) or 1)

    if photometric in _RGB_PHOTOMETRIC or (array.ndim == 3 and array.shape[-1] == 3 and frames == 1) or array.ndim == 4:
        if photometric.startswith("YBR"):
            array = convert_color_space(array, photometric, "RGB")
        if frames > 1:
            array = _reduce_volume(array, volume, axis=0)
        if array.dtype != np.uint8:
            array = _to_uint8(_percentiles(array.astype(np.float32)))
        return array

    values = apply_modality_lut(array, ds).astype(np.float32)
    if frames > 1 and values.ndim == 3:
        values = _reduce_volume(values, volume, axis=0)
    modality = str(ds.get("Modality", "")).upper()
    center, width = _first_value(ds.get("WindowCenter")), _first_value(ds.get("WindowWidth"))
    if window in CT_WINDOWS and modality == "CT":
        unit = _window(values, *CT_WINDOWS[window])
    elif window == "auto" and center is not None and width:
        unit = _window(values, center, width)
    else:
        unit = _percentiles(values)
    if photometric == "MONOCHROME1":
        unit = 1.0 - unit
    return _to_uint8(unit)


def load_nifti(path, window="auto", volume="middle"):
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(path))
    values = np.asanyarray(image.dataobj).astype(np.float32)
    values = np.squeeze(values)
    if values.ndim == 4:  # time series or several channels: first volume
        values = values[..., 0]
    if values.ndim == 3:  # axial slices (RAS orientation)
        values = _reduce_volume(values, volume, axis=2)
    if values.ndim != 2:
        raise ValueError(f"unsupported NIfTI shape {image.shape}")
    values = np.rot90(values)  # anatomical display orientation (anterior at the top)
    # NIfTI has no modality: a CT window applies only when chosen explicitly
    unit = _window(values, *CT_WINDOWS[window]) if window in CT_WINDOWS else _percentiles(values)
    return _to_uint8(unit)


def load_raster(path):
    with Image.open(path) as image:
        image.seek(0)  # multi-page TIFF: first page
        if image.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
            return _to_uint8(_percentiles(np.array(image, dtype=np.float32)))
        if image.mode in ("1", "L", "LA", "I;8"):
            return np.array(image.convert("L"))
        rgb = np.array(image.convert("RGB"))
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
        return rgb[..., 0]  # grayscale stored as RGB
    return rgb


def load_image(path, window="auto", volume="middle"):
    """An 8-bit image, (H, W) for grayscale or (H, W, 3) for colour."""
    kind = file_kind(path)
    if kind == "dicom":
        array = load_dicom(path, window, volume)
    elif kind == "nifti":
        array = load_nifti(path, window, volume)
    elif kind == "raster":
        array = load_raster(path)
    else:
        raise ValueError("unsupported file type")
    if array.ndim not in (2, 3) or min(array.shape[:2]) < 1:
        raise ValueError(f"unexpected image shape {array.shape}")
    return np.ascontiguousarray(array)


def letterbox(array, size):
    """Pads to a square (keeping the proportions of the image) and resizes to size x size."""
    image = Image.fromarray(array)
    width, height = image.size
    side = max(width, height)
    square = Image.new(image.mode, (side, side), 0)
    square.paste(image, ((side - width) // 2, (side - height) // 2))
    return square.resize((size, size), Image.BILINEAR)


def image_info(path):
    """Cheap header information for the dataset summary (no full decoding when possible)."""
    kind = file_kind(path)
    info = {"kind": kind, "modality": None, "volume": False, "high_bit_depth": False, "size": None}
    if kind == "dicom":
        import pydicom
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        info["modality"] = str(ds.get("Modality", "")) or None
        info["volume"] = int(ds.get("NumberOfFrames", 1) or 1) > 1
        info["high_bit_depth"] = int(ds.get("BitsStored", 8) or 8) > 8
        if "Rows" in ds and "Columns" in ds:
            info["size"] = (int(ds.Columns), int(ds.Rows))
    elif kind == "nifti":
        import nibabel as nib
        shape = [d for d in nib.load(path).shape if d > 1]
        info["volume"] = len(shape) >= 3
        info["high_bit_depth"] = True
        info["size"] = tuple(shape[:2]) if len(shape) >= 2 else None
    elif kind == "raster":
        with Image.open(path) as image:
            info["size"] = image.size
            info["high_bit_depth"] = image.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")
    return info
