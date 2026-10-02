"""Overlap and distance metrics of predicted masks, per case and class.

For each class (background excluded): Dice, IoU (Jaccard), sensitivity (recall), precision, the
95th percentile Hausdorff distance (HD95) and the average symmetric surface distance (ASSD), in
the physical units of the image (mm for medical images, pixels otherwise).
A class absent from both the reference and the prediction of a case is not counted (NaN); a
class missed or hallucinated has Dice 0 and no distance (NaN).
"""
import numpy as np
from scipy import ndimage

METRICS = ["Dice", "IoU", "HD95", "ASSD", "Sensitivity", "Precision"]
HIGHER_IS_BETTER = {"Dice": True, "IoU": True, "HD95": False, "ASSD": False, "Sensitivity": True, "Precision": True}


def _surface(mask):
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    return mask & ~ndimage.binary_erosion(mask, structure, border_value=0)


def surface_distances(prediction, reference, spacing):
    """Distances from each surface voxel of one mask to the surface of the other (both ways)."""
    surface_p, surface_r = _surface(prediction), _surface(reference)
    to_r = ndimage.distance_transform_edt(~surface_r, sampling=spacing)
    to_p = ndimage.distance_transform_edt(~surface_p, sampling=spacing)
    return np.concatenate([to_r[surface_p], to_p[surface_r]])


def case_metrics(prediction, reference, num_classes, spacing):
    """{class index: {metric: value}} for one case. Arrays (z, y, x); spacing (z, y, x)."""
    if prediction.shape[0] == 1:  # 2D
        prediction, reference, spacing = prediction[0], reference[0], spacing[1:]
    out = {}
    for c in range(1, num_classes):
        p, r = prediction == c, reference == c
        tp, ps, rs = np.logical_and(p, r).sum(), p.sum(), r.sum()
        if ps == 0 and rs == 0:
            out[c] = {m: np.nan for m in METRICS}
            continue
        values = {"Dice": 2 * tp / (ps + rs), "IoU": tp / (ps + rs - tp),
                  "Sensitivity": tp / rs if rs else np.nan, "Precision": tp / ps if ps else np.nan,
                  "HD95": np.nan, "ASSD": np.nan}
        if ps and rs:
            distances = surface_distances(p, r, spacing)
            values["HD95"] = float(np.percentile(distances, 95))
            values["ASSD"] = float(distances.mean())
        out[c] = {k: float(v) for k, v in values.items()}
    return out


def summarise(per_case, classes):
    """Mean of each metric over cases and classes ("values"), per class ("per_class") and the
    per-case mean Dice ("case_dice")."""
    per_class = {}
    for c in range(1, len(classes)):
        per_class[classes[c]] = {m: float(np.nanmean([case[c][m] for case in per_case if c in case]))
                                 if any(np.isfinite(case[c][m]) for case in per_case if c in case) else np.nan
                                 for m in METRICS}
    values = {m: float(np.nanmean([v[m] for v in per_class.values()])) if any(np.isfinite(v[m]) for v in per_class.values())
              else np.nan for m in METRICS}
    case_dice = [float(np.nanmean([case[c]["Dice"] for c in case])) if any(np.isfinite(case[c]["Dice"]) for c in case)
                 else np.nan for case in per_case]
    return {"values": values, "per_class": per_class, "case_dice": case_dice}
