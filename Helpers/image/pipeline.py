"""The image classification pipeline, following the tabular one step by step:
1. prepare the images (read, window, reduce volumes, pad and resize, cache);
2. for each network, stratified K-fold cross-validation on the training images, with a
   decision threshold optimised on each validation fold (binary problems);
3. for each network, a final model trained on all the training images, evaluated on the
   test images with the mean threshold of the folds; Grad-CAM figures, predictions and the
   exported model are saved.
Outputs go to ./Materials in the same format as the tabular automator (Excel metrics,
confusion matrices, ROC and precision-recall curves), so the results page shows both.

Progress is printed in the format that web/jobs.py follows ("<model> is starting", ...).
"""
import gc
import json
import logging
import os
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from Helpers import MetricsReport, behave_metrics
from . import dataset
from .explain import save_gradcam_figures
from .inference import export_model
from .io import CT_WINDOWS
from .models import BY_KEY, ImageClassifier
from .training import (describe_device, extract_features, fine_tune, fit_linear_head, linear_head_weights,
                       predict_proba)

MATERIALS = "Materials"
MODES = {"features": "Feature extraction", "finetune": "Fine-tuning"}


class FixedProbabilities:
    """Precomputed class probabilities behind the ``predict_proba`` interface of the metric
    helpers: X is an array of sample indices."""

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, X):
        return self.probabilities[np.asarray(X, dtype=int)]


def _banner(text):
    print("------------- \n", f"{text} \n", "-------------")


def _model_line(text):
    print("-------------------- \n", f"{text} \n", "--------------------")


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def _predictions(probabilities, threshold):
    if probabilities.shape[1] == 2:
        return (probabilities[:, 1] > threshold).astype(int)
    return probabilities.argmax(axis=1)


def _fold_threshold(probabilities, labels, metric):
    optimizer = behave_metrics.ThresholdOptimizer(FixedProbabilities(probabilities), np.arange(len(labels)), labels)
    threshold = optimizer.find_optimal_threshold(metric_to_track=metric)
    return 0.5 if threshold is None else float(threshold)


def _scores(probabilities, labels, threshold):
    metrics = behave_metrics.Metrics(probabilities, labels)
    metrics.compute_metrics(threshold=threshold)
    return metrics.get_scores()


def _prepare(input_folder, params, log):
    """Cached images and integer labels of both splits. Class 1 is the positive class of
    binary problems."""
    classes = list(params["classes"])
    if len(classes) == 2 and params.get("positive_class") in classes:
        positive = params["positive_class"]
        classes = [c for c in classes if c != positive] + [positive]
    index = {c: i for i, c in enumerate(classes)}

    splits, formats = {}, {"dicom": False, "nifti": False, "high_bit_raster": False}
    for split in ("train", "test"):
        samples, _ = dataset.scan_split(os.path.join(input_folder, split))
        samples = [s for s in samples if s["class"] in index]
        formats["dicom"] |= any(s["kind"] == "dicom" for s in samples)
        formats["nifti"] |= any(s["kind"] == "nifti" for s in samples)
        formats["high_bit_raster"] |= any(_high_bit_raster(s["path"]) for s in samples if s["kind"] == "raster")
        ready, failed = dataset.preprocess(samples, os.path.join(input_folder, "cache", split),
                                           params["window"], params["volume"], log=log)
        for item in failed:
            logging.error(f"Image skipped: {item['path']}: {item['error']}")
            log(f"Could not read {os.path.basename(item['path'])}: {item['error']}")
        splits[split] = {
            "paths": np.array([s["cached"] for s in ready]),
            "files": [os.path.relpath(s["path"], os.path.join(input_folder, split)) for s in ready],
            "labels": np.array([index[s["class"]] for s in ready], dtype=int),
            "failed": len(failed),
        }
        log(f"{split.capitalize()}: {len(ready)} images ready, {len(failed)} skipped")
    return classes, splits, formats


def _high_bit_raster(path):
    """16-bit or float PNG/TIFF (reads the header only)."""
    from PIL import Image
    try:
        with Image.open(path) as image:
            return image.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")
    except Exception:
        return False


def run_image_pipeline(input_folder, params):
    log_file = os.path.join(MATERIALS, "error_log.log")
    os.makedirs(MATERIALS, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')
    try:
        return _run(input_folder, params)
    except Exception as e:
        logging.error(traceback.format_exc())
        print(f"Error in pipeline: {e}")
        return f"Error: {e}"


def _run(input_folder, params):
    k = int(params["k_folds"])
    metric = params["metric"]
    mode = params["mode"]
    pretrained = params.get("pretrained", True)
    settings = {"epochs": int(params["epochs"]), "learning_rate": float(params["learning_rate"]),
                "patience": int(params["patience"]), "batch_size": int(params["batch_size"]),
                "augmentation": params["augmentation"]}
    backbones = [BY_KEY[key] for key in params["models"]]

    _banner("Preparing images")
    print(f"Device: {describe_device()} · mode: {MODES[mode]}")
    classes, splits, formats = _prepare(input_folder, params, print)
    train, test = splits["train"], splits["test"]
    counts = np.bincount(train["labels"], minlength=len(classes))
    if (counts < k).any():
        small = ", ".join(c for c, n in zip(classes, counts) if n < k)
        return f"Error: every class needs at least {k} readable training images for {k}-fold cross-validation ({small})."
    if len(test["labels"]) == 0:
        return "Error: no readable test image."
    num_classes = len(classes)
    y_train, y_test = train["labels"], test["labels"]

    pd.DataFrame({"index": range(num_classes), "class": classes,
                  "train_images": counts, "test_images": np.bincount(y_test, minlength=num_classes)}
                 ).to_csv(os.path.join(MATERIALS, "classes.csv"), index=False)

    # ---- K-fold cross-validation ------------------------------------------------------
    _banner("Training on K-Fold cross validation")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    folds = list(skf.split(np.zeros(len(y_train)), y_train))
    scores_storage, thresholds, best_epochs, features = {}, {}, {}, {}
    for backbone in backbones:
        name = backbone.name
        _model_line(f"{name} is starting")
        try:
            fold_scores, fold_thresholds, epochs = {}, {}, []
            if mode == "features":
                model = ImageClassifier(backbone, num_classes, pretrained=pretrained)
                train_features = extract_features(model, train["paths"], settings["batch_size"], log=print)
                test_features = extract_features(model, test["paths"], settings["batch_size"], log=print)
                del model
                features[name] = (train_features, test_features)
            for i, (fit_idx, val_idx) in enumerate(folds, start=1):
                if mode == "features":
                    head = fit_linear_head(train_features[fit_idx], y_train[fit_idx])
                    probabilities = head.predict_proba(train_features[val_idx])
                else:
                    model, best_epoch = fine_tune(backbone, train["paths"][fit_idx], y_train[fit_idx], num_classes,
                                                  settings, log=print, label=f"{name} · fold {i}/{k} ·",
                                                  pretrained=pretrained)
                    probabilities = predict_proba(model, train["paths"][val_idx], settings["batch_size"])
                    epochs.append(best_epoch)
                    del model
                    _free_memory()
                threshold = _fold_threshold(probabilities, y_train[val_idx], metric)
                fold_thresholds[f"fold_{i}"] = threshold
                fold_scores[f"fold_{i}"] = _scores(probabilities, y_train[val_idx], threshold)
                print(f"{name} · fold {i}/{k}: AUC {fold_scores[f'fold_{i}']['AUC']:.3f}")
            scores_storage[name], thresholds[name], best_epochs[name] = fold_scores, fold_thresholds, epochs
            _model_line(f"{name} is completed successfully")
        except Exception as e:
            features.pop(name, None)
            logging.error(f"{name} failed and was skipped: {e}")
            logging.error(traceback.format_exc())
            _model_line(f"{name} failed and was skipped: {e}")
        _free_memory()
    if not scores_storage:
        return "Error: every network failed during the K-fold training (see the log)."
    MetricsReport.summary_results_excel(scores_storage, file=f"{k}_fold_results", conf_matrix_name=f"Internal_{k}_fold")
    _banner("Training on K-Fold cross validation completed successfully")

    # ---- Final models and external test -----------------------------------------------
    _banner("Evaluating algorithms on the test set")
    preprocessing = {"window": params["window"], "volume": params["volume"], "formats": formats}
    scores_test, probabilities_test = {}, {}
    for directory in ("Models", "Predictions", "GradCAM"):
        os.makedirs(os.path.join(MATERIALS, directory), exist_ok=True)
    for backbone in backbones:
        name = backbone.name
        if name not in scores_storage:
            continue
        _model_line(f"{name} is starting")
        try:
            threshold = float(np.mean(list(thresholds[name].values()))) if num_classes == 2 else 0.5
            if mode == "features":
                train_features, test_features = features[name]
                head = fit_linear_head(train_features, y_train)
                model = ImageClassifier(backbone, num_classes, pretrained=pretrained)
                model.set_linear_head(*linear_head_weights(head, num_classes))
                probabilities = head.predict_proba(test_features)
            else:
                epochs = max(1, int(round(float(np.median(best_epochs[name])))))
                print(f"{name}: final training on all the training images for {epochs} epoch(s)")
                model, _ = fine_tune(backbone, train["paths"], y_train, num_classes, settings, epochs=epochs,
                                     log=print, label=f"{name} · final ·", pretrained=pretrained)
                probabilities = predict_proba(model, test["paths"], settings["batch_size"])
            predictions = _predictions(probabilities, threshold)
            scores_test[name] = _scores(probabilities, y_test, threshold)
            probabilities_test[name] = probabilities
            print(f"{name}: test AUC {scores_test[name]['AUC']:.3f}")

            table = pd.DataFrame({"file": test["files"], "true_class": [classes[i] for i in y_test],
                                  "predicted_class": [classes[i] for i in predictions],
                                  "correct": predictions == y_test})
            for i, c in enumerate(classes):
                table[f"probability_{c}"] = probabilities[:, i]
            table.to_csv(os.path.join(MATERIALS, "Predictions", f"{_safe(name)}_test_predictions.csv"), index=False)

            export_model(model, os.path.join(MATERIALS, "Models", f"{_safe(name)}.pt"), classes,
                         threshold if num_classes == 2 else None, mode, preprocessing)
            try:
                save_gradcam_figures(model, test["paths"], y_test, probabilities, classes,
                                     os.path.join(MATERIALS, "GradCAM", _safe(name)), name, predictions=predictions)
            except Exception as e:
                with open(os.path.join(MATERIALS, "GradCAM_error_log.txt"), "a") as f:
                    f.write(f"{name}: {e}\n{traceback.format_exc()}\n")
            del model
            _model_line(f"{name} is completed successfully")
        except Exception as e:
            logging.error(f"{name} failed and was skipped: {e}")
            logging.error(traceback.format_exc())
            _model_line(f"{name} failed and was skipped: {e}")
        _free_memory()

    if scores_test:
        MetricsReport.external_summary(scores_test, file="test_results", conf_matrix_name="Test")
        try:
            curves = behave_metrics.ROCCurveEvaluator({n: FixedProbabilities(p) for n, p in probabilities_test.items()},
                                                      X_test=np.arange(len(y_test)), y_true=y_test)
            os.makedirs(os.path.join(MATERIALS, "ROC_Curves"), exist_ok=True)
            curves.plot_roc_curves(save_path=os.path.join(MATERIALS, "ROC_Curves"))
            curves.plot_pr_curves(save_path=os.path.join(MATERIALS, "ROC_Curves"))
        except Exception as e:
            print(f"Error here: {e}")

    with open(os.path.join(MATERIALS, "run_info.json"), "w") as f:
        json.dump({"automator": "image-classification", "mode": mode, "classes": classes,
                   "k_folds": k, "metric": metric, "device": describe_device(), "formats": formats,
                   "window": params["window"], "volume": params["volume"],
                   "ct_window": list(CT_WINDOWS[params["window"]]) if params["window"] in CT_WINDOWS else None,
                   "skipped_images": train["failed"] + test["failed"]}, f, indent=2)
    print("Pipeline completed successfully.")
    return "Pipeline completed successfully"


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
