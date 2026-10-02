"""Synthetic examples for the Image Segmentation automator. Writes, next to this script:

- Train.zip / Test.zip (2D, everyday photos): 128 x 128 aerial-like JPEG tiles of 20 areas
  (one folder per area, 3 tiles each in Train.zip) with "building" and "road" classes, as colour
  masks (PNG) and a labels.json naming the classes;
- Train3D.zip / Test3D.zip (3D, medical): prostate-like MRI cases with two series, a T2-weighted
  DICOM series and an ADC map (NIfTI on a coarser grid, another field of view), and a mask on the
  T2 grid with the gland ("prostate", 1) and a lesion inside it ("lesion", 2) in most cases:
      images/<patient>/t2/          single-frame MR DICOM slices
      images/<patient>/adc.nii.gz   ADC map
      masks/<patient>.nii.gz        labels 0, 1, 2

Run it to regenerate the zips: python Examples/image-segmentation/make_example.py
"""
import json
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 128
BUILDING, ROAD = (230, 25, 75), (255, 225, 25)  # mask colours: classes 1 and 2 (sorted by colour)


# ---- 2D: aerial-like tiles -----------------------------------------------------------------

def _ground(rng):
    """Fields and grass: low-frequency colour patches with fine texture."""
    coarse = rng.random((6, 6, 3))
    patches = np.asarray(Image.fromarray((coarse * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 255
    base = np.array([95, 120, 70], np.float32) + (patches - 0.5) * np.array([60, 50, 40], np.float32)
    return base + 10 * rng.standard_normal((SIZE, SIZE, 1))


def _tile(rng):
    image, labels = _ground(rng), np.zeros((SIZE, SIZE), np.uint8)
    y, x = np.mgrid[:SIZE, :SIZE]
    for _ in range(int(rng.integers(1, 3))):  # roads: straight strips, some slanted
        angle = rng.choice([0, np.pi / 2, rng.uniform(0, np.pi)])
        offset, width = rng.uniform(0.2, 0.8) * SIZE, rng.uniform(6, 11)
        distance = np.abs((x - SIZE / 2) * np.sin(angle) - (y - SIZE / 2) * np.cos(angle) - (offset - SIZE / 2))
        road = distance <= width / 2
        image[road] = np.array([120, 118, 112]) + 8 * rng.standard_normal((road.sum(), 1))
        labels[road] = 2
    for _ in range(int(rng.integers(3, 8)) * 4):  # buildings: rectangles with a shadow, off the roads
        if (labels == 1).sum() > 0.2 * SIZE * SIZE:
            break
        h, w = int(rng.integers(9, 22)), int(rng.integers(9, 22))
        y0, x0 = int(rng.integers(2, SIZE - h - 4)), int(rng.integers(2, SIZE - w - 4))
        if labels[y0:y0 + h + 3, x0:x0 + w + 3].any():
            continue
        image[y0 + 3:y0 + h + 3, x0 + 3:x0 + w + 3] *= 0.45  # shadow
        roof = [np.array([170, 75, 60]), np.array([160, 160, 165]), np.array([215, 210, 200])][int(rng.integers(3))]
        image[y0:y0 + h, x0:x0 + w] = roof + 6 * rng.standard_normal((h, w, 1))
        labels[y0:y0 + h, x0:x0 + w] = 1
    return np.clip(image, 0, 255).astype(np.uint8), labels


def make_2d(folder, areas, tiles, seed):
    rng = np.random.default_rng(seed)
    colours = np.array([(0, 0, 0), BUILDING, ROAD], np.uint8)
    for a in range(areas):
        for t in range(tiles):
            image, labels = _tile(rng)
            name = os.path.join(f"area_{seed}{a:02d}", f"tile_{t:02d}")
            for sub in ("images", "masks"):
                os.makedirs(os.path.join(folder, sub, os.path.dirname(name)), exist_ok=True)
            Image.fromarray(image).save(os.path.join(folder, "images", name + ".jpg"), quality=92)
            Image.fromarray(colours[labels]).save(os.path.join(folder, "masks", name + ".png"))
    with open(os.path.join(folder, "labels.json"), "w") as f:
        json.dump({"1": "building", "2": "road"}, f, indent=2)


# ---- 3D: prostate-like MRI -----------------------------------------------------------------

T2_SPACING, T2_SIZE = (0.6, 0.6, 3.0), (64, 64, 16)      # x, y, z (mm), columns, rows, slices
ADC_SPACING, ADC_SIZE = (1.2, 1.2, 3.0), (40, 40, 16)
T2_ORIGIN = (-19.2, -19.2, -24.0)                          # LPS position of the first voxel (mm)
ADC_ORIGIN = (-24.0, -24.0, -24.0)


def _grid(spacing, size, origin):
    axes = [origin[i] + np.arange(size[i]) * spacing[i] for i in range(3)]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return x, y, z


def _distance(grid, center, radii):
    x, y, z = grid
    return ((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2 + ((z - center[2]) / radii[2]) ** 2


def _case(rng, lesion):
    """T2 and ADC arrays (z, y, x) and the labels on the T2 grid."""
    gland = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-3, 3))
    radii = (rng.uniform(10, 14), rng.uniform(8, 11), rng.uniform(10, 14))
    t2_grid, adc_grid = _grid(T2_SPACING, T2_SIZE, T2_ORIGIN), _grid(ADC_SPACING, ADC_SIZE, ADC_ORIGIN)
    blob = lambda d: np.clip(1.2 - d, 0, 1) ** 0.4  # noqa: E731  (soft edge)
    t2 = 160 + 230 * blob(_distance(t2_grid, gland, radii)) + 25 * rng.standard_normal(T2_SIZE[::-1])
    adc = 900 + 700 * blob(_distance(adc_grid, gland, radii)) + 60 * rng.standard_normal(ADC_SIZE[::-1])
    labels = (_distance(t2_grid, gland, radii) <= 1).astype(np.uint8)
    if lesion:
        spot = tuple(gland[i] + rng.uniform(-0.4, 0.4) * radii[i] for i in range(3))
        size = rng.uniform(3.5, 5.5)
        shape = (size, size, size * 1.4)
        t2 -= 180 * blob(_distance(t2_grid, spot, shape))
        adc -= 950 * blob(_distance(adc_grid, spot, shape))
        labels[(_distance(t2_grid, spot, shape) <= 1) & (labels == 1)] = 2
    return np.clip(t2, 0, None), np.clip(adc, 0, None), labels


def _write_t2(folder, volume, patient):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    os.makedirs(folder)
    series, study = generate_uid(), generate_uid()
    for k in range(volume.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = MRImageStorage, generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
        ds.Modality, ds.PatientID, ds.SeriesDescription, ds.SeriesNumber = "MR", patient, "T2 TSE axial", 3
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [T2_ORIGIN[0], T2_ORIGIN[1], T2_ORIGIN[2] + k * T2_SPACING[2]]
        ds.PixelSpacing, ds.SliceThickness = [T2_SPACING[1], T2_SPACING[0]], T2_SPACING[2]
        ds.InstanceNumber, ds.Rows, ds.Columns = k + 1, volume.shape[1], volume.shape[2]
        ds.PhotometricInterpretation, ds.SamplesPerPixel = "MONOCHROME2", 1
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = volume[k].astype(np.uint16).tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(os.path.join(folder, f"IM{k + 1:04d}.dcm"), write_like_original=False)


def _write_nifti(path, volume, spacing, origin, dtype=np.float32):
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(volume.astype(dtype))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(image, path)


def make_3d(folder, patients, seed):
    rng = np.random.default_rng(seed)
    for p in range(patients):
        name = f"patient_{seed}{p:03d}"
        t2, adc, labels = _case(rng, lesion=p % 4 != 0)
        _write_t2(os.path.join(folder, "images", name, "t2"), t2, name)
        _write_nifti(os.path.join(folder, "images", name, "adc.nii.gz"), adc, ADC_SPACING, ADC_ORIGIN)
        _write_nifti(os.path.join(folder, "masks", name + ".nii.gz"), labels, T2_SPACING, T2_ORIGIN, np.uint8)
    with open(os.path.join(folder, "labels.json"), "w") as f:
        json.dump({"0": "background", "1": "prostate", "2": "lesion"}, f, indent=2)


def main():
    work = tempfile.mkdtemp()
    try:
        for name, maker, args, seed in (("Train", make_2d, (20, 3), 1), ("Test", make_2d, (20, 1), 2),
                                        ("Train3D", make_3d, (30,), 1), ("Test3D", make_3d, (10,), 2)):
            folder = os.path.join(work, name)
            maker(folder, *args, seed=seed)
            shutil.make_archive(os.path.join(HERE, name), "zip", folder)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
