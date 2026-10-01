"""3D boxes from the 2D boxes of consecutive slices.

The networks detect objects slice by slice (2.5D). Boxes of the same class on consecutive
slices that overlap (IoU >= 0.5) are linked into one object; its 3D box spans the union of its
slice boxes and its first to last slice, and its score is the highest score of its slices.
"""
import numpy as np

from .metrics import iou_matrix

MIN_SCORE = 0.05     # slice boxes below are ignored (they would only add noise)
LINK_IOU = 0.5
MAX_OBJECTS = 100


def merge_slices(slice_detections, min_score=MIN_SCORE, link_iou=LINK_IOU):
    """``slice_detections``: {z: (boxes (n, 4), scores, labels)}. Returns 3D (boxes (n, 6)
    x1 y1 z1 x2 y2 z2, scores, labels)."""
    finished, active = [], []
    for z in sorted(slice_detections):
        boxes, scores, labels = (np.asarray(v) for v in slice_detections[z])
        keep = scores >= min_score
        boxes, scores, labels = boxes[keep].reshape(-1, 4), scores[keep], labels[keep]
        # Tracks not continued on the previous slice are finished
        finished += [t for t in active if t["last"] < z - 1]
        active = [t for t in active if t["last"] == z - 1]
        extended = set()
        for i in np.argsort(-scores, kind="stable"):
            candidates = [k for k, t in enumerate(active) if k not in extended and t["label"] == labels[i]]
            if candidates:
                ious = iou_matrix(boxes[i:i + 1], np.array([active[k]["boxes"][-1] for k in candidates]))[0]
                best = int(np.argmax(ious))
                if ious[best] >= link_iou:
                    track = active[candidates[best]]
                    track["boxes"].append(boxes[i])
                    track["scores"].append(scores[i])
                    track["last"] = z
                    extended.add(candidates[best])
                    continue
            active.append({"label": labels[i], "boxes": [boxes[i]], "scores": [scores[i]], "first": z, "last": z})
            extended.add(len(active) - 1)
    tracks = finished + active
    if not tracks:
        return np.zeros((0, 6)), np.zeros(0), np.zeros(0, int)
    out_boxes, out_scores, out_labels = [], [], []
    for t in tracks:
        b = np.array(t["boxes"])
        out_boxes.append([b[:, 0].min(), b[:, 1].min(), t["first"], b[:, 2].max(), b[:, 3].max(), t["last"] + 1])
        out_scores.append(max(t["scores"]))
        out_labels.append(int(t["label"]))
    order = np.argsort(-np.array(out_scores), kind="stable")[:MAX_OBJECTS]
    return np.array(out_boxes, float)[order], np.array(out_scores)[order], np.array(out_labels)[order]
