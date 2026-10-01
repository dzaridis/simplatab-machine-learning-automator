"""Images and volumes of the detection automator, as 8-bit arrays.

- 2D images: read like the image classification automator (Helpers.image.io): DICOM with the
  modality LUT and windows, NIfTI, 8/16-bit PNG and TIFF, JPEG, BMP.
- 3D volumes: NIfTI, multi-frame DICOM or a folder of DICOM slices (one series), returned as
  (slices, rows, columns) = (z, y, x). The voxel indices are those of the file: x and y are the
  column and the row of a slice, z the slice (for NIfTI, the first, second and third array axes;
  for a DICOM series, the slices sorted by position). 3D boxes and masks use these indices.
- 2.5D: the networks are 2D; a slice is given to them as an RGB image made of the slice and its
  two neighbours, which gives them the local 3D context.
"""
import os

import numpy as np
from PIL import Image

from Helpers.image import io as mio

VOLUME_EXTENSIONS = (".nii", ".nii.gz")


def is_nifti(path):
    return path.lower().endswith(VOLUME_EXTENSIONS)


def read_image(path, window="auto"):
    """A 2D image as an 8-bit (H, W, 3) array."""
    array = mio.load_image(path, window, "middle")
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return np.ascontiguousarray(array[..., :3])


def image_size(path):
    """(width, height) of a 2D image without decoding it when possible."""
    kind = mio.file_kind(path)
    if kind == "raster":
        with Image.open(path) as image:
            return image.size
    if kind == "dicom":
        import pydicom
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return int(ds.Columns), int(ds.Rows)
    array = read_image(path)
    return array.shape[1], array.shape[0]


# ---------------------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------------------

def _dicom_files(folder):
    import pydicom
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and mio.file_kind(path) == "dicom":
            files.append((pydicom.dcmread(path, force=True), path))
    return files


def _series_order(dataset):
    position = dataset.get("ImagePositionPatient")
    if position is not None and len(position) == 3:
        return float(position[2])
    return float(dataset.get("InstanceNumber", 0) or 0)


def _to_8bit(values, window, modality, header=None):
    """As for 2D images: a CT window when chosen (CT DICOM or NIfTI), else the window of the DICOM
    header when automatic, else the 0.5-99.5 percentiles of the whole volume."""
    if window in mio.CT_WINDOWS and modality in ("CT", "NIFTI"):
        return mio._to_uint8(mio._window(values, *mio.CT_WINDOWS[window]))
    if window == "auto" and header is not None:
        center, width = mio._first_value(header.get("WindowCenter")), mio._first_value(header.get("WindowWidth"))
        if center is not None and width:
            return mio._to_uint8(mio._window(values, center, width))
    return mio._to_uint8(mio._percentiles(values))


def read_volume(path, window="auto"):
    """A volume as an 8-bit (z, y, x) array. ``path`` is a NIfTI file, a multi-frame DICOM
    file or a folder of DICOM slices. The whole volume shares one intensity mapping."""
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    if os.path.isdir(path):
        files = sorted(_dicom_files(path), key=lambda item: _series_order(item[0]))
        if not files:
            raise ValueError("folder without DICOM files")
        values = np.stack([apply_modality_lut(ds.pixel_array, ds).astype(np.float32) for ds, _ in files])
        first = files[0][0]
        return _to_8bit(values, window, str(first.get("Modality", "")).upper(), first)
    if is_nifti(path):
        import nibabel as nib
        values = np.squeeze(np.asanyarray(nib.load(path).dataobj).astype(np.float32))
        if values.ndim == 4:
            values = values[..., 0]
        if values.ndim != 3:
            raise ValueError(f"not a 3D volume (shape {values.shape})")
        return _to_8bit(values.transpose(2, 1, 0), window, "NIFTI")
    if mio.file_kind(path) == "dicom":
        import pydicom
        ds = pydicom.dcmread(path, force=True)
        values = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        if values.ndim != 3:
            raise ValueError("single-frame DICOM: not a volume")
        return _to_8bit(values, window, str(ds.get("Modality", "")).upper(), ds)
    raise ValueError("unsupported volume format")


def volume_shape(path):
    """(z, y, x) without reading the voxels when possible."""
    if is_nifti(path):
        import nibabel as nib
        shape = [d for d in nib.load(path).shape]
        while len(shape) > 3 and shape[-1] == 1:
            shape.pop()
        x, y, z = shape[:3]
        return z, y, x
    if os.path.isdir(path):
        files = _dicom_files(path)
        return len(files), int(files[0][0].Rows), int(files[0][0].Columns)
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    return int(ds.get("NumberOfFrames", 1) or 1), int(ds.Rows), int(ds.Columns)


def slice_25d(volume, z):
    """Slice z with its neighbours as the three channels: (y, x, 3) uint8."""
    last = volume.shape[0] - 1
    return np.stack([volume[max(z - 1, 0)], volume[z], volume[min(z + 1, last)]], axis=-1)


# ---------------------------------------------------------------------------------------
# Masks -> boxes
# ---------------------------------------------------------------------------------------

def read_mask(path):
    """A label map: (y, x) for a 2D mask, (z, y, x) for a NIfTI volume."""
    if is_nifti(path):
        import nibabel as nib
        values = np.squeeze(np.asanyarray(nib.load(path).dataobj))
        if values.ndim == 3:
            return np.rint(values).astype(np.int32).transpose(2, 1, 0)
        if values.ndim == 2:
            return np.rint(values).astype(np.int32).T
        raise ValueError(f"unsupported mask shape {values.shape}")
    with Image.open(path) as image:
        values = np.array(image)
    if values.ndim == 3:  # colour mask: any non-black pixel is labelled with its first channel
        values = values[..., 0]
    return values.astype(np.int32)


def mask_boxes(mask):
    """One box per connected component of every label value (0 is background):
    [(label, box)], box = (x1, y1, x2, y2) or (x1, y1, z1, x2, y2, z2) in continuous
    coordinates (the last voxel of a component ends at index + 1)."""
    from scipy import ndimage
    boxes = []
    for value in sorted(int(v) for v in np.unique(mask) if v != 0):
        components, count = ndimage.label(mask == value)
        for region in ndimage.find_objects(components):
            if region is None:
                continue
            if mask.ndim == 2:
                ys, xs = region
                boxes.append((value, (xs.start, ys.start, xs.stop, ys.stop)))
            else:
                zs, ys, xs = region
                boxes.append((value, (xs.start, ys.start, zs.start, xs.stop, ys.stop, zs.stop)))
    return boxes
