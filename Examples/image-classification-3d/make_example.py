"""Synthetic example for the 3D image classification automator: prostate-like multi-parametric MRI.

Writes Train3D.zip (40 patients) and Test3D.zip (16 patients) next to this script, with the layout
    <class>/<patient>/study_1/t2/        T2-weighted series: single-frame MR DICOM slices
    <class>/<patient>/study_1/adc.nii.gz ADC map: NIfTI on a coarser grid, another field of view
Each study shows a gland in the middle of the pelvis. "malignant" studies have a lesion that is
dark on T2 and dark on ADC (restricted diffusion); a third of the "benign" studies have a nodule
that is dark on T2 only: telling them apart needs both series, aligned in patient space.

Run it to regenerate the zips: python Examples/image-classification-3d/make_example.py
"""
import os
import shutil
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T2_SPACING, T2_SIZE = (0.6, 0.6, 3.0), (64, 64, 16)      # x, y, z (mm), columns, rows, slices
ADC_SPACING, ADC_SIZE = (1.2, 1.2, 3.0), (40, 40, 16)
T2_ORIGIN = (-19.2, -19.2, -24.0)                          # LPS position of the first voxel (mm)
ADC_ORIGIN = (-24.0, -24.0, -24.0)


def _grid(spacing, size, origin):
    axes = [origin[i] + np.arange(size[i]) * spacing[i] for i in range(3)]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return x, y, z


def _blob(grid, center, radii):
    x, y, z = grid
    d = ((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2 + ((z - center[2]) / radii[2]) ** 2
    return np.clip(1.2 - d, 0, 1) ** 0.7


def _study(rng, kind):
    """T2 and ADC arrays (z, y, x) of a study: kind is "normal", "nodule" (benign) or "lesion"."""
    gland = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-3, 3))
    radii = (rng.uniform(11, 14), rng.uniform(8, 11), rng.uniform(10, 14))
    t2_grid, adc_grid = _grid(T2_SPACING, T2_SIZE, T2_ORIGIN), _grid(ADC_SPACING, ADC_SIZE, ADC_ORIGIN)
    t2 = 180 + 220 * _blob(t2_grid, gland, radii) + 25 * rng.standard_normal(T2_SIZE[::-1])
    adc = 900 + 700 * _blob(adc_grid, gland, radii) + 60 * rng.standard_normal(ADC_SIZE[::-1])
    if kind != "normal":
        spot = tuple(gland[i] + rng.uniform(-0.45, 0.45) * radii[i] for i in range(3))
        size = rng.uniform(3.5, 6.0)
        t2 -= 170 * _blob(t2_grid, spot, (size, size, size * 1.3))
        if kind == "lesion":
            adc -= 950 * _blob(adc_grid, spot, (size, size, size * 1.3))
    return np.clip(t2, 0, None), np.clip(adc, 0, None)


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


def _write_adc(path, volume):
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing(ADC_SPACING)
    image.SetOrigin(ADC_ORIGIN)
    sitk.WriteImage(image, path)


def make_split(folder, patients, seed):
    rng = np.random.default_rng(seed)
    for p in range(patients):
        malignant = p % 2 == 1
        kind = "lesion" if malignant else ("nodule" if p % 6 == 0 else "normal")
        name = f"patient_{seed}{p:03d}"
        study = os.path.join(folder, "malignant" if malignant else "benign", name, "study_1")
        t2, adc = _study(rng, kind)
        _write_t2(os.path.join(study, "t2"), t2, name)
        _write_adc(os.path.join(study, "adc.nii.gz"), adc)


def main():
    work = tempfile.mkdtemp()
    try:
        for name, patients, seed in (("Train3D", 40, 1), ("Test3D", 16, 2)):
            folder = os.path.join(work, name)
            make_split(folder, patients, seed)
            shutil.make_archive(os.path.join(HERE, name), "zip", folder)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
