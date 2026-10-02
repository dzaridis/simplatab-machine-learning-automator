"""Synthetic MRI-like studies for the tests of the 3D image classification automator.

Each study has a T2 series (a folder of single-frame MR DICOM slices) and an ADC series (a NIfTI
file with a coarser grid and a different field of view). "malignant" studies have a lesion,
bright on T2 and dark on ADC, at the same position in patient space in both series.
"""
import os

import numpy as np

CLASSES = ("benign", "malignant")
T2_SPACING = (0.8, 0.8, 3.0)   # x, y, z in mm
T2_SIZE = (40, 40, 12)          # columns, rows, slices
ADC_SPACING = (1.6, 1.6, 3.0)
ADC_SIZE = (24, 24, 12)
ORIGIN = (-16.0, -16.0, -18.0)  # LPS position of the first T2 voxel (mm)


def _lesion_mask(spacing, size, origin, center, radius):
    x = origin[0] + np.arange(size[0]) * spacing[0]
    y = origin[1] + np.arange(size[1]) * spacing[1]
    z = origin[2] + np.arange(size[2]) * spacing[2]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + ((zz - center[2]) / 1.5) ** 2) <= radius ** 2


def _study_arrays(rng, malignant):
    """T2 (z, y, x) and ADC (z, y, x) arrays of one study, and the lesion centre (mm)."""
    center = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-6, 6))
    t2 = 300 + 40 * rng.standard_normal(T2_SIZE[::-1])
    adc_origin = (ORIGIN[0] - 1.6, ORIGIN[1] - 1.6, ORIGIN[2])
    adc = 1400 + 60 * rng.standard_normal(ADC_SIZE[::-1])
    if malignant:
        t2[_lesion_mask(T2_SPACING, T2_SIZE, ORIGIN, center, 6)] += 500
        adc[_lesion_mask(ADC_SPACING, ADC_SIZE, adc_origin, center, 6)] -= 900
    return t2, adc, adc_origin, center


def write_dicom_series(folder, volume, spacing, origin, description="T2 TSE", series_uid=None, modality="MR",
                       study_uid=None):
    """A (z, y, x) volume as single-frame DICOM slices (files in shuffled order)."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, CTImageStorage, generate_uid
    os.makedirs(folder, exist_ok=True)
    series_uid = series_uid or generate_uid()
    study_uid = study_uid or generate_uid()
    for k in np.random.default_rng(len(folder)).permutation(volume.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage if modality == "MR" else CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.SeriesDescription, ds.SeriesNumber = description, 1
        ds.Modality, ds.PatientID = modality, "SYNTHETIC"
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + float(k) * spacing[2]]
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.SliceThickness = spacing[2]
        ds.InstanceNumber = int(k) + 1
        ds.Rows, ds.Columns = volume.shape[1:]
        ds.PhotometricInterpretation, ds.SamplesPerPixel = "MONOCHROME2", 1
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = np.clip(volume[k], 0, 65535).astype(np.uint16).tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(os.path.join(folder, f"{series_uid[-6:]}_{int(k):04d}"), write_like_original=False)


def write_nifti(path, volume, spacing, origin):
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(image, path)


def make_studies(root, per_class=4, seed=0, layout="full", extra_series=False):
    """Writes per_class studies of each class under root/<class>/. Layouts:
    - "full": <patient>/<study>/t2/ (DICOM) and <patient>/<study>/adc.nii.gz
    - "no_study": <patient>/t2/ and <patient>/adc.nii.gz
    - "pacs": <patient>/<study>/ holding both series as DICOM slices (told apart by their UID)
    - "single": <patient>.nii.gz (T2 only)
    Returns {study id: lesion centre or None}."""
    from pydicom.uid import generate_uid
    rng = np.random.default_rng(seed)
    truth = {}
    for c, name in enumerate(CLASSES):
        for i in range(per_class):
            patient = f"patient_{seed}{c}{i:02d}"
            t2, adc, adc_origin, center = _study_arrays(rng, malignant=c == 1)
            base = os.path.join(root, name, patient)
            if layout == "single":
                write_nifti(base + ".nii.gz", t2, T2_SPACING, ORIGIN)
                truth[f"{name}/{patient}.nii.gz"] = center if c else None
                continue
            study = os.path.join(base, "study_1") if layout in ("full", "pacs") else base
            if layout == "pacs":
                study_uid = generate_uid()
                write_dicom_series(study, t2, T2_SPACING, ORIGIN, "T2 TSE", study_uid=study_uid)
                write_dicom_series(study, adc, ADC_SPACING, adc_origin, "ADC", study_uid=study_uid)
            else:
                write_dicom_series(os.path.join(study, "t2"), t2, T2_SPACING, ORIGIN)
                write_nifti(os.path.join(study, "adc.nii.gz"), adc, ADC_SPACING, adc_origin)
                if extra_series and i % 2 == 0:
                    write_nifti(os.path.join(study, "dwi.nii.gz"), adc, ADC_SPACING, adc_origin)
            truth[os.path.relpath(study, root).replace(os.sep, "/")] = center if c else None
    return truth
