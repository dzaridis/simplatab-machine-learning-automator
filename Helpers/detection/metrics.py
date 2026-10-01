"""Detection metrics, for 2D boxes (x1 y1 x2 y2) and 3D boxes (x1 y1 z1 x2 y2 z2).

- AP (average precision) as in COCO: detections of a class sorted by score, each matched to the
  unmatched ground-truth box of the same class and image with the highest IoU above the
  threshold; precision made monotone and averaged at 101 recall levels. mAP averages the
  classes and the IoU thresholds: 0.50:0.95 in 2D (COCO), 0.10:0.50 in 3D (3D boxes overlap
  less for the same quality). At most 100 detections per image.
- AR: the highest recall, averaged likewise.
- FROC: sensitivity (share of the boxes found) against false positives per image (volume) at
  the main IoU (0.5 in 2D, 0.25 in 3D); CPM = mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and
  8 false positives per image, the score of the LUNA16 challenge.
- At an operating score threshold: precision, recall and F1 of the boxes, and image-level
  sensitivity and specificity (an image is positive when it has boxes, predicted positive when
  a detection reaches the threshold).
"""
import numpy as np

IOU_THRESHOLDS = {2: np.linspace(0.5, 0.95, 10), 3: np.linspace(0.1, 0.5, 9)}
MAIN_IOU = {2: 0.5, 3: 0.25}
FROC_POINTS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
MAX_DETECTIONS = 100
RECALL_LEVELS = np.linspace(0, 1, 101)


def metric_names(dim):
    """Metrics of the tables, in order."""
    if dim == 2:
        return ["mAP", "AP50", "AP75", "AR100", "FROC CPM", "Precision", "Recall", "F1", "Image sensitivity", "Image specificity"]
    return ["mAP", "AP10", "AP25", "AP50", "AR", "FROC CPM", "Precision", "Recall", "F1", "Volume sensitivity", "Volume specificity"]


def iou_matrix(a, b):
    """IoU between boxes a (n, 2d) and b (m, 2d), for 2D (d=2) or 3D (d=3) boxes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    d = a.shape[1] // 2
    low = np.maximum(a[:, None, :d], b[None, :, :d])
    high = np.minimum(a[:, None, d:], b[None, :, d:])
    inter = np.prod(np.clip(high - low, 0, None), axis=2)
    vol_a = np.prod(a[:, d:] - a[:, :d], axis=1)
    vol_b = np.prod(b[:, d:] - b[:, :d], axis=1)
    return inter / np.maximum(vol_a[:, None] + vol_b[None, :] - inter, 1e-12)


def _top(detections):
    """At most MAX_DETECTIONS per image, by score."""
    boxes, scores, labels = detections
    order = np.argsort(-np.asarray(scores), kind="stable")[:MAX_DETECTIONS]
    return np.asarray(boxes, float)[order], np.asarray(scores, float)[order], np.asarray(labels)[order]


def _match(gt_boxes, det_boxes, threshold):
    """COCO greedy matching (detections in score order): True for the matched detections."""
    matched = np.zeros(len(det_boxes), bool)
    if not len(gt_boxes) or not len(det_boxes):
        return matched
    ious = iou_matrix(det_boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), bool)
    for i in range(len(det_boxes)):
        candidates = np.where(~taken & (ious[i] >= threshold))[0]
        if len(candidates):
            j = candidates[np.argmax(ious[i, candidates])]
            taken[j] = True
            matched[i] = True
    return matched


def _ap(scores, matched, total):
    """COCO 101-point interpolated average precision."""
    if total == 0:
        return np.nan, np.nan
    if not len(scores):
        return 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(matched[order])
    fp = np.cumsum(~matched[order])
    recall = tp / total
    precision = tp / np.maximum(tp + fp, np.finfo(float).eps)
    for i in range(len(precision) - 2, -1, -1):  # monotone envelope
        precision[i] = max(precision[i], precision[i + 1])
    index = np.searchsorted(recall, RECALL_LEVELS, side="left")
    sampled = np.array([precision[i] if i < len(precision) else 0.0 for i in index])
    return float(sampled.mean()), float(recall[-1])


def average_precision(ground_truth, detections, num_classes, thresholds):
    """AP and recall per class (rows) and IoU threshold (columns). ``ground_truth`` and
    ``detections`` map an image to (boxes, labels) and (boxes, scores, labels)."""
    ap = np.full((num_classes, len(thresholds)), np.nan)
    ar = np.full((num_classes, len(thresholds)), np.nan)
    top = {image: _top(d) for image, d in detections.items()}
    for c in range(num_classes):
        total = sum(int(np.sum(np.asarray(labels) == c)) for _, labels in ground_truth.values())
        for t, threshold in enumerate(thresholds):
            scores, matched = [], []
            for image, (gt_boxes, gt_labels) in ground_truth.items():
                boxes, det_scores, det_labels = top.get(image, (np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
                keep = det_labels == c
                gt = np.asarray(gt_boxes, float)[np.asarray(gt_labels) == c]
                scores.append(det_scores[keep])
                matched.append(_match(gt, boxes[keep], threshold))
            ap[c, t], ar[c, t] = _ap(np.concatenate(scores) if scores else np.zeros(0),
                                     np.concatenate(matched) if matched else np.zeros(0, bool), total)
    return ap, ar


def _all_matches(ground_truth, detections, threshold):
    """Every detection (class-aware matching at ``threshold``): scores, matched flags, image ids."""
    scores, matched, images = [], [], []
    for image, (gt_boxes, gt_labels) in ground_truth.items():
        boxes, det_scores, det_labels = _top(detections.get(image, (np.zeros((0, 4)), np.zeros(0), np.zeros(0))))
        flags = np.zeros(len(boxes), bool)
        for c in np.unique(det_labels):
            keep = det_labels == c
            gt = np.asarray(gt_boxes, float)[np.asarray(gt_labels) == c]
            flags[keep] = _match(gt, boxes[keep], threshold)
        scores.append(det_scores)
        matched.append(flags)
        images += [image] * len(boxes)
    if not scores:
        return np.zeros(0), np.zeros(0, bool), []
    return np.concatenate(scores), np.concatenate(matched), images


def froc(ground_truth, detections, threshold):
    """FROC curve (false positives per image, sensitivity) and the CPM score."""
    total = sum(len(labels) for _, labels in ground_truth.values())
    scores, matched, _ = _all_matches(ground_truth, detections, threshold)
    if total == 0:
        return np.zeros(0), np.zeros(0), np.nan
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(matched[order])
    fp = np.cumsum(~matched[order])
    fppi = fp / max(1, len(ground_truth))
    sensitivity = tp / total
    points = [sensitivity[fppi <= p].max() if np.any(fppi <= p) else 0.0 for p in FROC_POINTS]
    return fppi, sensitivity, float(np.mean(points))


def best_threshold(ground_truth, detections, threshold):
    """Score threshold with the highest F1 of the boxes (at IoU ``threshold``)."""
    total = sum(len(labels) for _, labels in ground_truth.values())
    scores, matched, _ = _all_matches(ground_truth, detections, threshold)
    if not len(scores) or total == 0:
        return 0.5
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(matched[order])
    fp = np.cumsum(~matched[order])
    f1 = 2 * tp / np.maximum(tp + fp + total, 1)
    return float(scores[order][int(np.argmax(f1))])


def at_threshold(ground_truth, detections, score_threshold, iou_threshold):
    """Precision, recall, F1 of the boxes and image-level sensitivity / specificity."""
    total = sum(len(labels) for _, labels in ground_truth.values())
    scores, matched, images = _all_matches(ground_truth, detections, iou_threshold)
    keep = scores >= score_threshold
    tp, fp = int(matched[keep].sum()), int((~matched[keep]).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / total if total else np.nan
    f1 = 2 * tp / (tp + fp + total) if (tp + fp + total) else np.nan
    flagged = {image for image, k in zip(images, keep) if k}
    positives = [image for image, (_, labels) in ground_truth.items() if len(labels)]
    negatives = [image for image, (_, labels) in ground_truth.items() if not len(labels)]
    sensitivity = np.mean([image in flagged for image in positives]) if positives else np.nan
    specificity = np.mean([image not in flagged for image in negatives]) if negatives else np.nan
    return {"Precision": precision, "Recall": recall, "F1": f1, "sensitivity": float(sensitivity),
            "specificity": float(specificity)}


def evaluate(ground_truth, detections, num_classes, dim, score_threshold):
    """Every metric of ``metric_names(dim)``, the AP per class and the FROC curve."""
    thresholds = IOU_THRESHOLDS[dim]
    ap, ar = average_precision(ground_truth, detections, num_classes, thresholds)
    mean = lambda values: float(np.nanmean(values)) if np.isfinite(values).any() else np.nan  # noqa: E731
    column = lambda iou: int(np.argmin(np.abs(thresholds - iou)))  # noqa: E731
    main = MAIN_IOU[dim]
    fppi, sensitivity, cpm = froc(ground_truth, detections, main)
    point = at_threshold(ground_truth, detections, score_threshold, main)
    if dim == 2:
        values = {"mAP": mean(ap), "AP50": mean(ap[:, column(0.5)]), "AP75": mean(ap[:, column(0.75)]),
                  "AR100": mean(ar), "Image sensitivity": point["sensitivity"], "Image specificity": point["specificity"]}
    else:
        values = {"mAP": mean(ap), "AP10": mean(ap[:, column(0.1)]), "AP25": mean(ap[:, column(0.25)]),
                  "AP50": mean(ap[:, column(0.5)]), "AR": mean(ar),
                  "Volume sensitivity": point["sensitivity"], "Volume specificity": point["specificity"]}
    values.update({"FROC CPM": cpm, "Precision": point["Precision"], "Recall": point["Recall"], "F1": point["F1"]})
    return {"values": {m: values[m] for m in metric_names(dim)}, "per_class_ap": np.nanmean(ap, axis=1) if len(ap) else ap,
            "per_class_ap_main": ap[:, column(main)] if len(ap) else ap, "froc": (fppi, sensitivity)}


def precision_recall(ground_truth, detections, iou_threshold):
    """Precision-recall curve of all the boxes (class-aware matching)."""
    total = sum(len(labels) for _, labels in ground_truth.values())
    scores, matched, _ = _all_matches(ground_truth, detections, iou_threshold)
    if not len(scores) or not total:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(matched[order])
    fp = np.cumsum(~matched[order])
    return tp / total, tp / np.maximum(tp + fp, 1)
