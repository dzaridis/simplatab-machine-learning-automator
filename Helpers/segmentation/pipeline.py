"""The segmentation pipeline:
1. prepare the cases (read, align, map the mask values to classes, cache) and the preprocessing
   plan of the networks trained by the automator (nnU-Net plans its own);
2. validation of each network: K-fold cross-validation grouped by patient (the final network is
   retrained on all the training cases) or a hold-out split (the network trained on it is final);
3. evaluation on Test.zip: Dice, IoU, HD95, ASSD, sensitivity and precision per case and class,
   predicted masks, overlays with the uncertainty maps, and the exported networks.
Progress is printed in the format web/jobs.py follows ("<network> is starting", ...).
"""
import gc
import json
import logging
import os
import re
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import torch

from . import data, figures, metrics as M, preprocessing as P
from .inference import export_model
from .models import BY_KEY, divisor, fit_patch
from .nnunet import NNUNet
from .training import PreparedCase, describe_device, predict, to_original, train

MATERIALS = "Materials"


def _banner(text):
    print("------------- \n", f"{text} \n", "-------------")


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "case"


def _short(error):
    text = str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__
    return text[:300]


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_segmentation_pipeline(input_folder, params):
    os.makedirs(MATERIALS, exist_ok=True)
    logging.basicConfig(filename=os.path.join(MATERIALS, "error_log.log"), level=logging.ERROR,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    try:
        return _run(input_folder, params)
    except Exception as e:
        logging.error(traceback.format_exc())
        print(f"Error in pipeline: {e}")
        return f"Error: {e}"


# ---------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------

def _strata(metas, k):
    """Stratification by the classes present in each case (rare combinations merged)."""
    keys = [",".join(map(str, m["present"])) for m in metas]
    counts = pd.Series(keys).value_counts()
    return np.array([key if counts[key] >= k else "other" for key in keys])


def kfold_splits(metas, k, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    groups = np.array([m["group"] for m in metas])
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(fit, val) for fit, val in splitter.split(np.zeros(len(metas)), _strata(metas, k), groups)]


def holdout_split(metas, fraction, seed=0):
    from sklearn.model_selection import GroupShuffleSplit
    groups = np.array([m["group"] for m in metas])
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    return next(splitter.split(np.zeros(len(metas)), groups=groups))


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _foreground_intensities(folder, metas, channels, limit=50, per_case=20000, seed=0):
    rng = np.random.default_rng(seed)
    samples = [[] for _ in range(channels)]
    for meta in metas[:limit]:
        image, labels = data.load_cached(folder, meta, channels)
        mask = labels > 0
        for c in range(channels):
            values = image[c][mask] if mask.any() else image[c].ravel()
            if len(values) > per_case:
                values = values[rng.choice(len(values), per_case, replace=False)]
            samples[c].append(values)
    return samples


def _evaluate(prediction, reference, num_classes, meta):
    return M.case_metrics(prediction, reference, num_classes, meta["spacing"])


def _fold_summary(per_case, classes):
    return M.summarise(per_case, classes)


def _save_mask(prediction, meta, folder, mapping, dim, path):
    """The predicted mask with the original mask values (2D PNG, 3D NIfTI on the case grid)."""
    import SimpleITK as sitk
    values = mapping["values"]
    if dim == 2:
        from PIL import Image
        if isinstance(values[0], list):
            rgb = np.array(values, np.uint8)[prediction[0]]
            Image.fromarray(rgb).save(path + ".png")
        else:
            lookup = np.array(values, np.int64)
            out = lookup[prediction[0]]
            Image.fromarray(out.astype(np.uint8 if out.max() < 256 else np.uint16)).save(path + ".png")
    else:
        lookup = np.array(values, np.int64)
        image = sitk.GetImageFromArray(lookup[prediction].astype(np.uint16 if lookup.max() > 255 else np.uint8))
        image.CopyInformation(sitk.ReadImage(os.path.join(folder, f"{meta['name']}.nii.gz")))
        sitk.WriteImage(image, path + ".nii.gz")


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------

def _run(input_folder, params):
    started = time.time()
    seed = int(params.get("seed", 0))
    mapping, dim = params["mapping"], int(params["dim"])
    classes, num_classes = mapping["classes"], len(mapping["classes"])
    holdout = params["validation"] == "holdout"
    k = int(params.get("k_folds", 5))
    tta = bool(params.get("tta", True))
    pretrained = params.get("pretrained", True)
    settings = {"epochs": params["epochs"], "iterations": params["iterations"], "batch_size": params["batch_size"],
                "learning_rate": params["learning_rate"], "augmentation": params["augmentation"],
                "nnunet_epochs": params.get("nnunet_epochs", 50), "nnunet_iterations": params.get("nnunet_iterations", 250)}
    specs = [BY_KEY[key] for key in params["models"]]

    _banner("Preparing images")
    caches, metas, skipped = {}, {}, []
    for split in ("train", "test"):
        cases, _, _ = data.scan_split(os.path.join(input_folder, split))
        caches[split] = os.path.abspath(os.path.join(input_folder, "cache", split))
        shutil.rmtree(caches[split], ignore_errors=True)
        metas[split], failed = data.cache_split(cases, caches[split], mapping, params.get("channels") or None, log=print)
        for error in failed:
            logging.error(f"Case skipped: {error}")
            print(f"Could not read {error}")
        skipped += failed
        print(f"{split.capitalize()}: {len(metas[split])} cases ready, {len(failed)} skipped")
    train_metas, test_metas = metas["train"], metas["test"]
    if not train_metas or not test_metas:
        return "Error: no readable training or test case."
    channels, rgb = train_metas[0]["channels"], train_metas[0]["rgb"]
    groups = len({m["group"] for m in train_metas})
    if not holdout and groups < k:
        return f"Error: {k}-fold cross-validation needs at least {k} training patients (cases): {groups} found."
    if holdout and groups < 2:
        return "Error: hold-out validation needs at least 2 training patients (cases)."
    unit = "mm" if dim == 3 or any(m["modality"] and m["modality"][0] not in ("RGB", "image") for m in train_metas) else "px"
    print(f"{dim}D · {channels} channel(s) · classes: {', '.join(classes[1:])} · {describe_device()}")

    folds = [holdout_split(train_metas, float(params.get("holdout_fraction", 0.2)), seed)] if holdout \
        else kfold_splits(train_metas, k, seed)

    # Preprocessing plan and prepared training cases of the automator's networks
    ours = [s for s in specs if s.family != "nnunet"]
    plan_, prepared = None, None
    if ours:
        modes = P.normalisation_modes(train_metas, channels, params.get("normalisation", "auto"))
        intensities = _foreground_intensities(caches["train"], train_metas, channels) if "ct" in modes else None
        plan_ = P.plan(train_metas, dim, channels, params.get("normalisation", "auto"), intensities)
        prepared = []
        for meta in train_metas:
            image, labels = P.prepare(*data.load_cached(caches["train"], meta, channels), meta, plan_)
            prepared.append(PreparedCase(image.astype(np.float16), labels))
        shapes = [case.labels.shape for case in prepared]
        print(f"Preprocessing: {', '.join(sorted(set(plan_['modes'])))} normalisation"
              + (f", spacing {' x '.join(f'{s:.2f}' for s in plan_['spacing'])} mm" if dim == 3 else ""))

    def patch_for(spec):
        return fit_patch(spec, P.patch_size(shapes, dim, divisor(spec, None)))

    # ---- Validation ----------------------------------------------------------------------
    stage = "Training on hold-out validation" if holdout else "Training on K-Fold cross validation"
    _banner(f"{stage} ({params.get('holdout_fraction', 0.2):.0%} of the cases)" if holdout else f"{stage} ({k} folds)")
    work = os.path.join(input_folder, "work")
    os.makedirs(work, exist_ok=True)
    validation, finals, failures, nnunet = {}, {}, [], None
    names = [m["name"] for m in train_metas]
    for spec in specs:
        print(f"{spec.name} is starting")
        try:
            t0 = time.time()
            fold_results = []
            if spec.family == "nnunet":
                if nnunet is None:
                    nnunet = NNUNet(os.path.abspath(os.path.join(work, "nnunet")), "2d" if dim == 2 else "3d_fullres", log=print)
                    os.makedirs(nnunet.workdir, exist_ok=True)
                    nn_modes = P.normalisation_modes(train_metas, channels, params.get("normalisation", "auto"))
                    nnunet.write_dataset(caches["train"], train_metas, nn_modes, classes)
                    nnunet.plan()
                    nnunet.set_splits([([names[i] for i in fit], [names[i] for i in val]) for fit, val in folds])
                for i, (fit, val) in enumerate(folds):
                    nnunet.train(i, settings, label=f"{spec.name} · fold {i + 1}/{len(folds)} ·")
                    val_metas = [train_metas[j] for j in val]
                    probabilities = nnunet.predict(i, caches["train"], val_metas, channels, tta)
                    per_case = [_evaluate(probabilities[m["name"]].argmax(0).astype(np.uint8),
                                          data.load_cached(caches["train"], m, channels)[1], num_classes, m) for m in val_metas]
                    fold_results.append(_fold_summary(per_case, classes))
                    print(f"{spec.name} · fold {i + 1}/{len(folds)}: Dice {fold_results[-1]['values']['Dice']:.3f}")
                finals[spec.name] = {"fold": 0 if holdout else "all"}
            else:
                patch = patch_for(spec)
                for i, (fit, val) in enumerate(folds):
                    model = train(spec, [prepared[j] for j in fit], channels, num_classes, patch, settings, log=print,
                                  label=f"{spec.name} · fold {i + 1}/{len(folds)} ·", pretrained=pretrained, rgb=rgb, seed=seed)
                    per_case = []
                    for j in val:
                        meta = train_metas[j]
                        p, u = predict(model, prepared[j].image.astype(np.float32), patch, tta)
                        per_case.append(_evaluate(to_original(p, u, meta["shape"])[0], data.load_cached(caches["train"], meta, channels)[1],
                                                  num_classes, meta))
                    fold_results.append(_fold_summary(per_case, classes))
                    print(f"{spec.name} · fold {i + 1}/{len(folds)}: Dice {fold_results[-1]['values']['Dice']:.3f}")
                    if holdout:
                        path = os.path.join(work, f"{spec.key}.pt")
                        torch.save(model.state_dict(), path)
                        finals[spec.name] = {"state": path, "patch": patch}
                    del model
                    _free()
                if not holdout:
                    finals[spec.name] = {"patch": patch}
            validation[spec.name] = fold_results
            print(f"{spec.name}: validation Dice {np.nanmean([f['values']['Dice'] for f in fold_results]):.3f} · {time.time() - t0:.0f} s")
            print(f"{spec.name} is completed successfully")
        except Exception as e:
            logging.error(f"{spec.name} failed and was skipped: {e}")
            logging.error(traceback.format_exc())
            print(f"{spec.name} failed and was skipped: {_short(e)}")
            failures.append({"model": spec.name, "reason": _short(e)})
            finals.pop(spec.name, None)
        _free()
    if not validation:
        return "Error: every network failed during the validation (see the log)."
    metric_names = M.METRICS
    table = {}
    for name, folds_ in validation.items():
        values = pd.DataFrame([f["values"] for f in folds_])[metric_names]
        table[name] = values.iloc[0].to_dict() if holdout else {
            m: f"{values[m].mean():.3f} ± {values[m].std(ddof=0):.3f}" for m in metric_names}
    validation_file = "holdout_results.xlsx" if holdout else f"{k}_fold_results.xlsx"
    pd.DataFrame(table).T[metric_names].to_excel(os.path.join(MATERIALS, validation_file))
    _banner(f"{stage} completed successfully")

    # ---- Test ----------------------------------------------------------------------------
    _banner("Evaluating algorithms on Test.zip")
    for folder in ("Models", "Predictions", "Overlays", "Segmentation_Plots", "Segmentation_Metrics"):
        os.makedirs(os.path.join(MATERIALS, folder), exist_ok=True)
    test_values, per_class_rows, per_case_rows, case_dice, exported = {}, [], [], {}, {}
    references = {m["name"]: data.load_cached(caches["test"], m, channels) for m in test_metas} if len(test_metas) <= 400 else None
    for spec in specs:
        if spec.name not in finals:
            continue
        print(f"{spec.name} is starting")
        try:
            final = finals[spec.name]
            outputs = os.path.join(work, "test", spec.key)
            os.makedirs(outputs, exist_ok=True)
            per_case = []
            if spec.family == "nnunet":
                if final["fold"] == "all":
                    nnunet.train("all", settings, label=f"{spec.name} · final ·")
                probabilities = nnunet.predict(final["fold"], caches["test"], test_metas, channels, tta)
                path = os.path.join(MATERIALS, "Models", f"{_safe(spec.name)}.zip")
                nnunet.export(final["fold"], path)
                exported[spec.name] = {"file": os.path.basename(path), "library": "nnunet", "fold": final["fold"],
                                       "configuration": nnunet.configuration}
            else:
                patch = final["patch"]
                if "state" in final:
                    from .models import build
                    model = build(spec, channels, num_classes, patch, pretrained=False, rgb=rgb)
                    model.load_state_dict(torch.load(final["state"], map_location="cpu"))
                else:
                    print(f"{spec.name}: final training on all the training cases")
                    model = train(spec, prepared, channels, num_classes, patch, settings, log=print,
                                  label=f"{spec.name} · final ·", pretrained=pretrained, rgb=rgb, seed=seed)
                model.eval()
            for meta in test_metas:
                image, reference = references[meta["name"]] if references else data.load_cached(caches["test"], meta, channels)
                if spec.family == "nnunet":
                    p = probabilities[meta["name"]]
                    prediction = p.argmax(0).astype(np.uint8)
                    uncertainty = (-(p * np.log(np.clip(p, 1e-8, 1))).sum(0) / np.log(p.shape[0])).astype(np.float32)
                else:
                    normalised, _ = P.prepare(image, None, meta, plan_)
                    p, u = predict(model, normalised, patch, tta)
                    prediction, uncertainty = to_original(p, u, meta["shape"])
                result = _evaluate(prediction, reference, num_classes, meta)
                per_case.append(result)
                np.savez_compressed(os.path.join(outputs, meta["name"] + ".npz"), prediction=prediction,
                                    uncertainty=uncertainty.astype(np.float16))
                masks = os.path.join(MATERIALS, "Predictions", _safe(spec.name))
                os.makedirs(masks, exist_ok=True)
                _save_mask(prediction, meta, caches["test"], mapping, dim, os.path.join(masks, _safe(meta["id"])))
                for c, values in result.items():
                    per_case_rows.append({"model": spec.name, "case": meta["id"], "class": classes[c], **values})
            summary = M.summarise(per_case, classes)
            test_values[spec.name] = summary["values"]
            case_dice[spec.name] = summary["case_dice"]
            for name, values in summary["per_class"].items():
                per_class_rows.append({"model": spec.name, "class": name, **values})
            print(f"{spec.name}: test Dice {summary['values']['Dice']:.3f}")

            if spec.family != "nnunet":
                path = os.path.join(MATERIALS, "Models", f"{_safe(spec.name)}.pt")
                export_model(model, path, {"network": spec.name, "dim": dim, "channels": channels, "rgb": rgb,
                                           "classes": classes, "mask_values": mapping["values"], "patch": patch,
                                           "modes": plan_["modes"], "stats": plan_["stats"], "spacing": plan_["spacing"],
                                           "tta": tta})
                exported[spec.name] = {"file": os.path.basename(path), "library": "torch", "patch": patch}
                del model

            # Overlays: the worst, median and best test cases
            order = [i for i in np.argsort(summary["case_dice"]) if np.isfinite(summary["case_dice"][i])]
            picks = list(dict.fromkeys([order[0], order[len(order) // 2], order[-1]])) if order else []
            folder = os.path.join(MATERIALS, "Overlays", _safe(spec.name))
            os.makedirs(folder, exist_ok=True)
            for rank, i in zip(("worst", "median", "best"), picks):
                meta = test_metas[i]
                image, reference = references[meta["name"]] if references else data.load_cached(caches["test"], meta, channels)
                with np.load(os.path.join(outputs, meta["name"] + ".npz")) as saved:
                    prediction, uncertainty = saved["prediction"], saved["uncertainty"].astype(np.float32)
                figures.overlay_figure(image, reference, prediction, uncertainty, classes, rgb,
                                       f"{spec.name} · {meta['id']} · Dice {summary['case_dice'][i]:.3f} ({rank} test case)",
                                       os.path.join(folder, f"{_safe(spec.name)}_{rank}_{_safe(meta['id'])}.png"))
            print(f"{spec.name} is completed successfully")
        except Exception as e:
            logging.error(f"{spec.name} failed and was skipped: {e}")
            logging.error(traceback.format_exc())
            print(f"{spec.name} failed and was skipped: {_short(e)}")
            failures.append({"model": spec.name, "reason": _short(e)})
        _free()
    if not test_values:
        return "Error: every network failed on the test set (see the log)."

    test = pd.DataFrame(test_values).T[metric_names]
    test.to_excel(os.path.join(MATERIALS, "test_results.xlsx"))
    pd.DataFrame(per_class_rows).to_csv(os.path.join(MATERIALS, "Segmentation_Metrics", "test_per_class.csv"), index=False)
    pd.DataFrame(per_case_rows).to_csv(os.path.join(MATERIALS, "Segmentation_Metrics", "test_per_case.csv"), index=False)
    best = test["Dice"].astype(float).idxmax()
    plots = os.path.join(MATERIALS, "Segmentation_Plots")
    figures.metric_bars(test[["Dice", "IoU", "Sensitivity", "Precision"]].astype(float), ["Dice", "IoU", "Sensitivity", "Precision"],
                        best, os.path.join(plots, "test_overlap_metrics.png"))
    figures.distance_bars(test[["HD95", "ASSD"]].astype(float), ["HD95", "ASSD"], best, os.path.join(plots, "test_distances.png"), unit)
    figures.dice_per_case(case_dice, best, os.path.join(plots, "test_dice_per_case.png"))
    if num_classes > 2:
        per_class = pd.DataFrame(per_class_rows).pivot(index="model", columns="class", values="Dice").reindex(columns=classes[1:])
        figures.class_heatmap(per_class.to_numpy(), list(per_class.index), classes[1:], "Test Dice per class",
                              os.path.join(plots, "test_dice_per_class.png"))
    print(f"Best network on Test.zip (Dice): {best}")

    with open(os.path.join(MATERIALS, "run_info.json"), "w") as f:
        json.dump({"automator": "image-segmentation", "dim": dim, "classes": classes, "mask_values": mapping["values"],
                   "channels": channels, "channel_names": params.get("channels") or [], "rgb": rgb,
                   "validation": params["validation"], "k_folds": None if holdout else k,
                   "holdout_fraction": params.get("holdout_fraction") if holdout else None,
                   "validation_file": validation_file, "metrics": metric_names, "best_model": best, "unit": unit,
                   "models": exported, "plan": plan_, "tta": tta, "epochs": params["epochs"], "iterations": params["iterations"],
                   "nnunet_epochs": settings["nnunet_epochs"], "nnunet_iterations": settings["nnunet_iterations"],
                   "device": describe_device(), "skipped": failures, "unreadable_cases": len(skipped),
                   "train_cases": len(train_metas), "test_cases": len(test_metas),
                   "kinds": sorted({k for m in train_metas for k in m["modality"]}),
                   "minutes": round((time.time() - started) / 60, 1)}, f, indent=2)
    print("Pipeline completed successfully.")
    return "Pipeline completed successfully"
