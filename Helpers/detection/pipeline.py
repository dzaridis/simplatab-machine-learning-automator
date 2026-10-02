"""The object detection pipeline, following the other automators step by step:
1. images: annotations read (COCO, YOLO, VOC, CSV or masks), images and volumes cached in 8 bits;
2. validation of every network on Train.zip:
   - K-fold (default): grouped (patient) and stratified folds; each fold network is fine-tuned
     with early stopping on 15% of its training images, then scored on its fold; the final
     network is trained on all of Train.zip for the median best number of epochs;
   - hold-out (faster): one network fine-tuned on 80% of Train.zip with early stopping on the
     other 20%, scored on them and kept as the final network;
   the score threshold of the operating point (best F1 of the boxes) is chosen on the
   validation images;
3. the final networks detect the objects of Test.zip: metrics, curves, images with their
   detections, D-RISE explanations, predictions and the exported networks.
3D volumes are processed slice by slice (2.5D) and the slice boxes merged into 3D boxes.
Progress is printed in the format that web/jobs.py follows.
"""
import gc
import json
import os
import shutil
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import torch

from . import dataset as dd
from . import metrics as M
from . import plots
from Helpers.image import io as mio
from Helpers.splits import index_rows, write_splits
from .annotations import AnnotationError, load_split
from .explain import drise
from .models import BY_KEY, input_size
from .training import describe_device, ground_truth, image_tensor, predict_items, train

MATERIALS = "Materials"
FORMAT = "simplatab-detector"


def _banner(text):
    print("------------- \n", f"{text} \n", "-------------")


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def _image_stem(name):
    """File name part of an image (relative path without extension)."""
    for extension in (".nii.gz", ".nii"):
        if name.lower().endswith(extension):
            return _safe(name[:-len(extension)])
    return _safe(os.path.splitext(name)[0])


def _short(error):
    text = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    return text[:300]


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_detection_pipeline(input_folder, params):
    try:
        return _run(input_folder, params)
    except (AnnotationError, ValueError) as e:
        print(f"Error: {e}")
        return f"Error: {e}"
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"


# ---------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------

def _evaluate(model, spec, items, indices, classes, dim, params, threshold=None):
    """Detections of the items and their metrics (at ``threshold``, or the best-F1 one)."""
    size = input_size(spec, params["image_size"])
    detections = predict_items(model, items, indices, size, params["batch_size"])
    truth = ground_truth(items, indices)
    chosen = threshold if threshold is not None else M.best_threshold(truth, detections, M.MAIN_IOU[dim])
    return detections, M.evaluate(truth, detections, len(classes), dim, chosen), chosen


def _validate(spec, items, train_indices, classes, dim, params, log):
    """K-fold or hold-out validation of a network. Returns the fold results, the operating
    threshold and either the best number of epochs (K-fold) or the trained network (hold-out)."""
    folds = []
    if params["validation"] == "holdout":
        fit, val = dd.holdout_split(items, train_indices, params["holdout_fraction"], seed=params.get("seed", 0))
        model, epoch = train(spec, items, dd.units(items, fit, params["negative_ratio"]), len(classes), params,
                             val_units=dd.units(items, val, params["negative_ratio"], seed=1), log=log,
                             pretrained=params.get("pretrained", True))
        _, result, threshold = _evaluate(model, spec, items, val, classes, dim, params)
        folds.append({"result": result, "threshold": threshold, "epoch": epoch})
        return folds, threshold, model
    for k, (fit, val) in enumerate(dd.kfold_splits(items, train_indices, params["k_folds"], seed=params.get("seed", 0))):
        inner_fit, inner_val = dd.holdout_split(items, fit, 0.15, seed=k)
        log(f"{spec.name}: fold {k + 1}/{params['k_folds']}")
        model, epoch = train(spec, items, dd.units(items, inner_fit, params["negative_ratio"]), len(classes), params,
                             val_units=dd.units(items, inner_val, params["negative_ratio"], seed=1), log=log,
                             pretrained=params.get("pretrained", True))
        _, result, threshold = _evaluate(model, spec, items, val, classes, dim, params)
        folds.append({"result": result, "threshold": threshold, "epoch": epoch})
        del model
        _free()
    return folds, float(np.mean([f["threshold"] for f in folds])), int(np.median([f["epoch"] for f in folds]))


def _write_splits(items, train_indices, params):
    """Materials/Splits/: the images of each fold, as _validate splits them (same seeds)."""
    seed = params.get("seed", 0)
    if params["validation"] == "holdout":
        folds, sets = [dd.holdout_split(items, train_indices, params["holdout_fraction"], seed=seed)], ("train", "validation")
        description = (f"Hold-out split of Train.zip grouped by patient folder (about {params['holdout_fraction']:.0%} "
                       "for validation, also used for early stopping).")
    else:
        folds, sets = [], ("train", "early_stopping", "validation")
        for k, (fit, val) in enumerate(dd.kfold_splits(items, train_indices, params["k_folds"], seed=seed)):
            folds.append((*dd.holdout_split(items, fit, 0.15, seed=k), val))
        description = (f"{params['k_folds']}-fold cross-validation of Train.zip grouped by patient folder and stratified on "
                       "the main class of each image; 15% of each training fold stops the training early.")
    extra = {"patient": [it.group for it in items], "boxes": [len(it.labels) for it in items]}
    write_splits(index_rows(folds, [it.name for it in items], extra, sets=sets), materials=MATERIALS,
                 kind="holdout" if params["validation"] == "holdout" else "kfold",
                 description=description + " id: the image (or volume) of Train.zip.")


# ---------------------------------------------------------------------------------------
# Test outputs
# ---------------------------------------------------------------------------------------

def _slice_of(item, truth_boxes, det_boxes):
    """Slice to show for a volume: through the first ground-truth box, else the top detection."""
    for boxes in (truth_boxes, det_boxes):
        if len(boxes):
            return int((boxes[0][2] + boxes[0][5]) // 2)
    return item.shape[0] // 2


def _cross_section(boxes, z):
    """2D boxes of the 3D boxes crossing slice z (and which ones)."""
    boxes = np.asarray(boxes).reshape(-1, 6)
    inside = (boxes[:, 2] <= z) & (boxes[:, 5] > z)
    return boxes[inside][:, [0, 1, 3, 4]], inside


def _matches(truth, detections, threshold, iou):
    """Detections above the threshold, their correctness and the missed boxes of an image."""
    boxes, scores, labels = detections
    keep = scores >= threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    gt_boxes, gt_labels = truth
    matched = np.zeros(len(boxes), bool)
    found = np.zeros(len(gt_boxes), bool)
    for c in np.unique(labels):
        dets = np.where(labels == c)[0]
        gts = np.where(gt_labels == c)[0]
        if not len(gts):
            continue
        ious = M.iou_matrix(boxes[dets], gt_boxes[gts])
        for i in np.argsort(-scores[dets], kind="stable"):
            candidates = np.where(~found[gts] & (ious[i] >= iou))[0]
            if len(candidates):
                j = candidates[np.argmax(ious[i, candidates])]
                found[gts[j]] = True
                matched[dets[i]] = True
    return (boxes, scores, labels), matched, (gt_boxes[~found], gt_labels[~found])


def _figures(model, spec, items, detections, classes, dim, threshold, params, folder, explain_folder):
    """Images with their detections and D-RISE explanations of the top detections."""
    iou = M.MAIN_IOU[dim]
    positives = [i for i in sorted(detections) if len(items[i].labels)]
    negatives = [i for i in sorted(detections) if not len(items[i].labels)]
    shown = positives[:6] + negatives[:2]
    explained = 0
    size = input_size(spec, params["image_size"])
    for i in shown:
        item = items[i]
        (boxes, scores, labels), matched, missed = _matches((item.boxes, item.labels), detections[i], threshold, iou)
        z = None
        title = f"{spec.name} · {item.name}"
        if dim == 3:
            z = _slice_of(item, item.boxes, boxes)
            boxes2, inside = _cross_section(boxes, z)
            boxes, scores, labels, matched = boxes2, scores[inside], labels[inside], matched[inside]
            missed_boxes, missed_inside = _cross_section(missed[0], z)
            missed = (missed_boxes, missed[1][missed_inside])
            title += f" · slice {z}"
        tensor, image = image_tensor(items, i, z, size)
        display = image if dim == 2 else np.repeat(image[..., 1:2], 3, axis=2)
        plots.detections_figure(display, (boxes, scores, labels), matched, missed, classes, title,
                                os.path.join(folder, f"{_safe(spec.name)}_{_image_stem(item.name)}.png"))
        if explained < params["drise_images"] and len(boxes):
            top = int(np.argmax(scores))
            height, width = image.shape[:2]
            scaled = boxes[top] * np.array([size / width, size / height] * 2)
            saliency = drise(model, tensor, scaled, int(labels[top]), n_masks=params["drise_masks"],
                             batch_size=params["batch_size"], seed=explained)
            from PIL import Image
            saliency = np.asarray(Image.fromarray((saliency * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR)) / 255
            plots.drise_figure(display, saliency, boxes[top], classes[int(labels[top])], float(scores[top]), title,
                               os.path.join(explain_folder, f"{_safe(spec.name)}_{_image_stem(item.name)}.png"))
            explained += 1


def _predictions_table(items, detections, classes, dim, threshold):
    rows = []
    coords = ["x_min", "y_min", "x_max", "y_max"] if dim == 2 else ["x_min", "y_min", "z_min", "x_max", "y_max", "z_max"]
    iou = M.MAIN_IOU[dim]
    for i, (boxes, scores, labels) in sorted(detections.items()):
        (kept, kept_scores, kept_labels), matched, _ = _matches((items[i].boxes, items[i].labels), (boxes, scores, labels), 0.0, iou)
        for box, score, label, ok in zip(kept, kept_scores, kept_labels, matched):
            values = list(np.round(box, 2))
            if dim == 3:
                values[5] = values[5] - 1  # last slice, as in the annotations
            rows.append(dict(image=items[i].name, **{"class": classes[int(label)]}, score=round(float(score), 4),
                             **dict(zip(coords, values)), above_threshold=bool(score >= threshold), matches_a_box=bool(ok)))
    return pd.DataFrame(rows, columns=["image", "class", "score"] + coords + ["above_threshold", "matches_a_box"])


def export(model, spec, classes, dim, params, threshold, path):
    """torchvision networks: TorchScript (.pt) with the metadata; transformers networks: a zip of
    the save_pretrained folder and simplatab.json."""
    info = {"format": FORMAT, "version": 1, "network": spec.name, "library": spec.library, "classes": classes,
            "dim": dim, "image_size": input_size(spec, params["image_size"]), "threshold": threshold,
            "normalize": spec.normalize, "window": params.get("window", "auto"),
            "input": "RGB image resized to image_size x image_size, values in [0, 1]"
                     + ("; 3D: slice z with slices z-1 and z+1 as the three channels" if dim == 3 else "")}
    if spec.library == "torchvision":
        import warnings
        from torch.jit import _builtins
        # featurewiz (tabular automator) replaces warnings.warn: script with the function torch
        # registered as its "aten::warn" builtin when it was imported
        registered = next((fn for fn, op in _builtins._builtin_ops if op == "aten::warn"), warnings.warn)
        replaced, warnings.warn = warnings.warn, registered
        try:
            scripted = torch.jit.script(model.model.cpu().eval())
        finally:
            warnings.warn = replaced
        torch.jit.save(scripted, path + ".pt", _extra_files={"simplatab.json": json.dumps(info)})
        model.to(next(model.parameters()).device)
        return path + ".pt"
    workdir = tempfile.mkdtemp()
    try:
        folder = os.path.join(workdir, os.path.basename(path))
        model.model.save_pretrained(folder)
        with open(os.path.join(folder, "simplatab.json"), "w") as f:
            json.dump(info, f, indent=2)
        shutil.make_archive(path, "zip", workdir, os.path.basename(path))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return path + ".zip"


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------

def _run(input_folder, params):
    started = time.time()
    print("Preparing images")
    train_set, test_set = load_split(os.path.join(input_folder, "train")), load_split(os.path.join(input_folder, "test"))
    classes, dim = train_set.classes, train_set.dim
    cache = os.path.join(input_folder, "cache")
    items, failed = dd.cache_split(train_set, os.path.join(cache, "train"), classes, params.get("window", "auto"))
    test_items, test_failed = dd.cache_split(test_set, os.path.join(cache, "test"), classes, params.get("window", "auto"))
    if not items or not test_items:
        return "Error: no image of Train.zip or Test.zip could be read."
    print(f"{'Volumes' if dim == 3 else 'Images'}: {len(items)} train, {len(test_items)} test · classes: {', '.join(classes)} · "
          f"{describe_device()}")
    for folder in ("Models", "Detections", "Explainability", "Detection_Curves", "Predictions"):
        os.makedirs(os.path.join(MATERIALS, folder), exist_ok=True)
    models = [BY_KEY[key] for key in params["models"]]
    metric_names = M.metric_names(dim)
    train_indices = list(range(len(items)))
    state_folder = os.path.join(cache, "networks")
    os.makedirs(state_folder, exist_ok=True)

    # ---- Validation ------------------------------------------------------------------------
    try:
        _write_splits(items, train_indices, params)
    except Exception as e:  # e.g. too few patients for the folds: reported by the validation below
        print(f"The validation splits could not be written: {e}")
    holdout = params["validation"] == "holdout"
    stage = "Training on hold-out validation" if holdout else "Training on K-Fold cross validation"
    _banner(f"{stage} ({params['holdout_fraction']:.0%} of the images)" if holdout else f"{stage} ({params['k_folds']} folds)")
    validation, thresholds, epochs, skipped = {}, {}, {}, []
    for spec in models:
        print(f"{spec.name} is starting")
        try:
            t0 = time.time()
            folds, threshold, outcome = _validate(spec, items, train_indices, classes, dim, params, print)
            validation[spec.name] = folds
            thresholds[spec.name] = threshold
            if holdout:
                torch.save(outcome.state_dict(), os.path.join(state_folder, f"{spec.key}.pt"))
                epochs[spec.name] = folds[0]["epoch"]
                del outcome
            else:
                epochs[spec.name] = outcome
            print(f"{spec.name}: validation mAP {np.nanmean([f['result']['values']['mAP'] for f in folds]):.3f}, "
                  f"threshold {threshold:.2f} · {time.time() - t0:.0f} s")
            print(f"{spec.name} is completed successfully")
        except Exception as e:
            traceback.print_exc()
            print(f"{spec.name} failed and was skipped: {_short(e)}")
            skipped.append({"model": spec.name, "reason": _short(e)})
        _free()
    if not validation:
        return "Error: every network failed during the validation (see the log)."
    table = {}
    for name, folds in validation.items():
        values = pd.DataFrame([f["result"]["values"] for f in folds])[metric_names]
        table[name] = values.iloc[0].to_dict() if holdout else {
            m: f"{values[m].mean():.3f} ± {values[m].std(ddof=0):.3f}" for m in metric_names}
    validation_file = "holdout_results.xlsx" if holdout else f"{params['k_folds']}_fold_results.xlsx"
    pd.DataFrame(table).T[metric_names].to_excel(os.path.join(MATERIALS, validation_file))
    _banner(f"{stage} completed successfully")

    # ---- Test --------------------------------------------------------------------------------
    _banner("Evaluating algorithms on Test.zip")
    test_indices = list(range(len(test_items)))
    results, pr_curves, froc_curves, per_class, exported = {}, {}, {}, {}, {}
    from .models import build
    for spec in [s for s in models if s.name in validation]:
        print(f"{spec.name} is starting")
        try:
            size = input_size(spec, params["image_size"])
            if holdout:
                model = build(spec, len(classes), size, pretrained=False)
                model.load_state_dict(torch.load(os.path.join(state_folder, f"{spec.key}.pt"), map_location="cpu"))
                model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
            else:
                print(f"{spec.name}: final network on all of Train.zip ({epochs[spec.name]} epochs)")
                model, _ = train(spec, items, dd.units(items, train_indices, params["negative_ratio"]), len(classes), params,
                                 epochs=epochs[spec.name], log=print, pretrained=params.get("pretrained", True))
            threshold = thresholds[spec.name]
            detections, result, _ = _evaluate(model, spec, test_items, test_indices, classes, dim, params, threshold)
            results[spec.name] = result["values"]
            per_class[spec.name] = result["per_class_ap"]
            froc_curves[spec.name] = result["froc"]
            truth = ground_truth(test_items, test_indices)
            pr_curves[spec.name] = M.precision_recall(truth, detections, M.MAIN_IOU[dim])
            _predictions_table(test_items, detections, classes, dim, threshold).to_csv(
                os.path.join(MATERIALS, "Predictions", f"{_safe(spec.name)}_test_predictions.csv"), index=False)
            try:
                _figures(model, spec, test_items, detections, classes, dim, threshold, params,
                         os.path.join(MATERIALS, "Detections"), os.path.join(MATERIALS, "Explainability"))
            except Exception as e:
                traceback.print_exc()
                print(f"{spec.name}: figures could not be produced ({_short(e)})")
            try:
                exported[spec.name] = os.path.basename(export(model, spec, classes, dim, params, threshold,
                                                              os.path.join(MATERIALS, "Models", _safe(spec.name))))
            except Exception as e:
                traceback.print_exc()
                print(f"{spec.name}: the network could not be exported ({_short(e)})")
            print(f"{spec.name}: test mAP {result['values']['mAP']:.3f}")
            print(f"{spec.name} is completed successfully")
            del model
        except Exception as e:
            traceback.print_exc()
            print(f"{spec.name} failed and was skipped: {_short(e)}")
            skipped.append({"model": spec.name, "reason": _short(e)})
        _free()
    if not results:
        return "Error: every network failed on the test set (see the log)."

    test = pd.DataFrame(results).T[metric_names]
    test.to_excel(os.path.join(MATERIALS, "test_results.xlsx"))
    best = test["mAP"].idxmax()
    pd.DataFrame(per_class, index=classes).T.to_csv(os.path.join(MATERIALS, "Detection_Curves", "test_ap_per_class.csv"),
                                                   index_label="model")
    plots.metric_bars(test, [m for m in metric_names if m not in ("Precision",)], best,
                      os.path.join(MATERIALS, "Detection_Curves", "test_metrics.png"))
    plots.class_heatmap(np.array([per_class[n] for n in test.index]), list(test.index), classes,
                        "Test AP per class (averaged over the IoU thresholds)", os.path.join(MATERIALS, "Detection_Curves", "ap_per_class.png"))
    main = M.MAIN_IOU[dim]
    plots.curves_grid({n: pr_curves[n] for n in test.index}, best, "Recall", "Precision",
                      f"Precision-recall of the test boxes (IoU ≥ {main})", os.path.join(MATERIALS, "Detection_Curves", "PR_CURVES.png"))
    plots.curves_grid({n: froc_curves[n] for n in test.index}, best,
                      f"False positives per {'image' if dim == 2 else 'volume'}", "Sensitivity",
                      f"FROC of the test boxes (IoU ≥ {main})", os.path.join(MATERIALS, "Detection_Curves", "FROC_CURVES.png"),
                      xlog=True, xlim=(0.1, 16))
    info = {"automator": "object-detection", "dim": dim, "format": train_set.format, "classes": classes,
            "validation": params["validation"], "k_folds": params.get("k_folds"), "holdout_fraction": params.get("holdout_fraction"),
            "validation_file": validation_file, "image_size": params["image_size"], "epochs": params["epochs"],
            "best_epochs": epochs, "thresholds": thresholds, "best_model": best, "models": exported,
            "metrics": metric_names, "main_iou": main, "device": describe_device(), "skipped": skipped,
            "unreadable_images": len(failed) + len(test_failed), "train_images": len(items), "test_images": len(test_items),
            "window": params.get("window", "auto"), "minutes": round((time.time() - started) / 60, 1),
            # File kinds, for the code of the results page (which readers it needs)
            "kinds": sorted({"dicom series" if os.path.isdir(i.path) else (mio.file_kind(i.path) or "other")
                             for i in items + test_items})}
    with open(os.path.join(MATERIALS, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"Best network on Test.zip (mAP): {best}")
    print("Pipeline completed successfully.")
    return "Pipeline completed successfully"
