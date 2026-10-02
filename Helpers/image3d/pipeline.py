"""The 3D image classification pipeline, the volumetric counterpart of Helpers/image/pipeline.py:
1. prepare the studies (read and align the chosen series, scale, crop, resize, cache);
2. for each network, K-fold cross-validation stratified by class and grouped by patient (all the
   studies of a patient fall in the same fold), with a decision threshold optimised on each
   validation fold (binary problems);
3. for each network, a final model trained on all the training studies, evaluated on the test
   studies with the mean threshold of the folds; 3D Grad-CAM figures, predictions and the
   exported model are saved.
The outputs have the format of the 2D automator, so the same results page shows them.
"""
import json
import logging
import os
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from Helpers import MetricsReport, behave_metrics
from Helpers.image.io import CT_WINDOWS
from Helpers.image.pipeline import (MATERIALS, MODES, FixedProbabilities, _banner, _fold_threshold, _free_memory,
                                    _model_line, _predictions, _safe, _scores)
from . import dataset, volumes
from .explain import save_gradcam_figures
from .inference import export_model
from .models import BY_KEY, VolumeClassifier
from .training import describe_device, extract_features, fine_tune, fit_linear_head, linear_head_weights, predict_proba


def run_image3d_pipeline(input_folder, params):
    log_file = os.path.join(MATERIALS, "error_log.log")
    os.makedirs(MATERIALS, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')
    try:
        return _run(input_folder, params)
    except Exception as e:
        logging.error(traceback.format_exc())
        print(f"Error in pipeline: {e}")
        return f"Error: {e}"


def _prepare(input_folder, params, classes, log):
    index = {c: i for i, c in enumerate(classes)}
    splits = {}
    for split in ("train", "test"):
        studies, _ = volumes.scan_split(os.path.join(input_folder, split))
        studies = [s for s in studies if s["class"] in index]
        ready, missing, failed = dataset.cache_studies(
            studies, params["channels"], params["shape"], params["crop"], params["window"],
            os.path.join(input_folder, "cache3d", split), log=log)
        for item in failed:
            logging.error(f"Study skipped: {item['study']}: {item['error']}")
            log(f"Could not read {item['study']}: {item['error']}")
        if missing:
            log(f"{split.capitalize()}: {len(missing)} studies skipped (without one of the series "
                f"{', '.join(params['channels'])}), e.g. {missing[0]['id']}")
        splits[split] = {
            "paths": np.array([s["cached"] for s in ready]),
            "studies": [s["id"] for s in ready],
            "patients": np.array([f"{s['patient']}" for s in ready]),
            "labels": np.array([index[s["class"]] for s in ready], dtype=int),
            "skipped": len(missing) + len(failed),
            "kinds": sorted({("nifti" if x["path"].lower().endswith((".nii", ".nii.gz")) else "dicom")
                             for s in ready for x in (volumes.series_of(s, params["channels"]) or [])}),
        }
        log(f"{split.capitalize()}: {len(ready)} studies ready, {len(missing) + len(failed)} skipped")
    return splits


def _run(input_folder, params):
    k = int(params["k_folds"])
    metric, mode = params["metric"], params["mode"]
    pretrained = params.get("pretrained", True)
    shape, channels = tuple(params["shape"]), list(params["channels"])
    settings = {"epochs": int(params["epochs"]), "learning_rate": float(params["learning_rate"]),
                "patience": int(params["patience"]), "batch_size": int(params["batch_size"]),
                "augmentation": params["augmentation"]}
    networks = [BY_KEY[key] for key in params["models"]]
    classes = list(params["classes"])
    if len(classes) == 2 and params.get("positive_class") in classes:
        classes = [c for c in classes if c != params["positive_class"]] + [params["positive_class"]]

    _banner("Preparing images")
    print(f"Device: {describe_device()} · mode: {MODES[mode]} · series: {', '.join(channels)} · "
          f"volume {shape[0]} x {shape[1]} x {shape[2]}")
    splits = _prepare(input_folder, params, classes, print)
    train, test = splits["train"], splits["test"]
    num_classes = len(classes)
    y_train, y_test = train["labels"], test["labels"]
    counts = np.bincount(y_train, minlength=num_classes)
    patients = [len(set(train["patients"][y_train == c])) for c in range(num_classes)]
    if min(patients) < k:
        small = ", ".join(c for c, n in zip(classes, patients) if n < k)
        return f"Error: every class needs at least {k} readable training patients for {k}-fold cross-validation ({small})."
    if len(y_test) == 0:
        return "Error: no readable test study."

    pd.DataFrame({"index": range(num_classes), "class": classes, "train_images": counts,
                  "test_images": np.bincount(y_test, minlength=num_classes), "train_patients": patients}
                 ).to_csv(os.path.join(MATERIALS, "classes.csv"), index=False)

    # ---- K-fold cross-validation, grouped by patient --------------------------------------
    _banner("Training on K-Fold cross validation")
    folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=10).split(
        np.zeros(len(y_train)), y_train, groups=train["patients"]))
    scores_storage, thresholds, best_epochs, features = {}, {}, {}, {}
    for spec in networks:
        name = spec.name
        _model_line(f"{name} is starting")
        try:
            fold_scores, fold_thresholds, epochs = {}, {}, []
            if mode == "features":
                model = VolumeClassifier(spec, len(channels), shape, num_classes, pretrained=pretrained)
                train_features = extract_features(model, train["paths"], settings["batch_size"], log=print)
                test_features = extract_features(model, test["paths"], settings["batch_size"], log=print)
                del model
                features[name] = (train_features, test_features)
            for i, (fit_idx, val_idx) in enumerate(folds, start=1):
                if mode == "features":
                    head = fit_linear_head(train_features[fit_idx], y_train[fit_idx])
                    probabilities = head.predict_proba(train_features[val_idx])
                else:
                    model, best_epoch = fine_tune(spec, train["paths"][fit_idx], y_train[fit_idx], num_classes,
                                                  len(channels), shape, settings, log=print,
                                                  label=f"{name} · fold {i}/{k} ·", pretrained=pretrained)
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

    # ---- Final models and external test -----------------------------------------------------
    _banner("Evaluating algorithms on the test set")
    kinds = sorted(set(train["kinds"]) | set(test["kinds"]))
    preprocessing = {"channels": channels, "shape": list(shape), "crop": params["crop"], "window": params["window"],
                     "kinds": {kind: True for kind in kinds}}
    scores_test, probabilities_test = {}, {}
    for directory in ("Models", "Predictions", "GradCAM"):
        os.makedirs(os.path.join(MATERIALS, directory), exist_ok=True)
    for spec in networks:
        name = spec.name
        if name not in scores_storage:
            continue
        _model_line(f"{name} is starting")
        try:
            threshold = float(np.mean(list(thresholds[name].values()))) if num_classes == 2 else 0.5
            if mode == "features":
                train_features, test_features = features[name]
                head = fit_linear_head(train_features, y_train)
                model = VolumeClassifier(spec, len(channels), shape, num_classes, pretrained=pretrained)
                model.set_linear_head(*linear_head_weights(head, num_classes))
                probabilities = head.predict_proba(test_features)
            else:
                epochs = max(1, int(round(float(np.median(best_epochs[name])))))
                print(f"{name}: final training on all the training studies for {epochs} epoch(s)")
                model, _ = fine_tune(spec, train["paths"], y_train, num_classes, len(channels), shape, settings,
                                     epochs=epochs, log=print, label=f"{name} · final ·", pretrained=pretrained)
                probabilities = predict_proba(model, test["paths"], settings["batch_size"])
            predictions = _predictions(probabilities, threshold)
            scores_test[name] = _scores(probabilities, y_test, threshold)
            probabilities_test[name] = probabilities
            print(f"{name}: test AUC {scores_test[name]['AUC']:.3f}")

            table = pd.DataFrame({"study": test["studies"], "patient": test["patients"],
                                  "true_class": [classes[i] for i in y_test],
                                  "predicted_class": [classes[i] for i in predictions],
                                  "correct": predictions == y_test})
            for i, c in enumerate(classes):
                table[f"probability_{c}"] = probabilities[:, i]
            table.to_csv(os.path.join(MATERIALS, "Predictions", f"{_safe(name)}_test_predictions.csv"), index=False)

            export_model(model, os.path.join(MATERIALS, "Models", f"{_safe(name)}.pt"), classes,
                         threshold if num_classes == 2 else None, mode, preprocessing)
            try:
                save_gradcam_figures(model, test["paths"], y_test, probabilities, classes,
                                     os.path.join(MATERIALS, "GradCAM", _safe(name)), name, predictions,
                                     channel_name=channels[0] if channels[0] != volumes.SINGLE else "")
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
        json.dump({"automator": "image-classification", "dim": 3, "mode": mode, "classes": classes,
                   "k_folds": k, "metric": metric, "device": describe_device(),
                   "channels": channels, "shape": list(shape), "crop": params["crop"], "window": params["window"],
                   "ct_window": list(CT_WINDOWS[params["window"]]) if params["window"] in CT_WINDOWS else None,
                   "kinds": {kind: True for kind in kinds}, "formats": {},
                   "skipped_images": train["skipped"] + test["skipped"]}, f, indent=2)
    print("Pipeline completed successfully.")
    return "Pipeline completed successfully"
