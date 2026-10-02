"""Example datasets of every automator (the examples of the repository, or small synthetic ones)
with a configuration that runs in minutes on a CPU. Python 3.9 compatible."""
import os
import shutil
import zipfile
from pathlib import Path

from .paths import simplatab_root

QUICK = {
    "tabular": {"models": ["logistic_regression", "random_forest", "xgboost"], "k_folds": 3,
                "hyperparameter_search": "none"},
    "image-classification": {"mode": "features", "k_folds": 3},
    "object-detection": {"models": ["fasterrcnn_mobilenet"], "validation": "holdout", "epochs": 3, "image_size": 320,
                         "drise_images": 1, "drise_masks": 50},
    "image-segmentation": {"validation": "holdout", "epochs": 3, "iterations": 10, "nnunet_epochs": 2,
                           "nnunet_iterations": 10},
    "time-series-forecasting": {"models": ["NHITS", "DLinear"], "horizon": 14, "k_folds": 2, "max_steps": 100,
                                "season": 7},
}


def _tabular(dest):
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "Target"})
    data.columns = [c.replace(" ", "_") for c in data.columns]
    data.insert(0, "ID", [f"P{i:04d}" for i in range(len(data))])
    train, test = train_test_split(data, test_size=0.25, random_state=0, stratify=data["Target"])
    train.to_csv(dest / "Train.csv", index=False)
    test.to_csv(dest / "Test.csv", index=False)
    return dest / "Train.csv", dest / "Test.csv", "Breast cancer (Wisconsin): 30 numeric features, Target 1 = benign."


def _image_2d(dest):
    """Small synthetic grayscale images: 'lesion' images hold a bright disc, 'normal' ones do not."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    for split, per_class in (("Train", 20), ("Test", 8)):
        with zipfile.ZipFile(dest / f"{split}.zip", "w") as archive:
            for label in ("normal", "lesion"):
                for i in range(per_class):
                    size = 64
                    y, x = np.mgrid[:size, :size]
                    image = 0.3 + 0.08 * rng.standard_normal((size, size))
                    if label == "lesion":
                        cy, cx = rng.integers(16, 48, 2)
                        image[(y - cy) ** 2 + (x - cx) ** 2 < 100] += 0.5
                    path = dest / "tmp.png"
                    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)
                    archive.write(path, f"{label}/patient_{i:02d}/image.png")
                    path.unlink()
    return dest / "Train.zip", dest / "Test.zip", "Synthetic 64x64 images: lesion (bright disc) vs. normal."


def _copy(folder, names, dest, description):
    source = simplatab_root() / "Examples" / folder
    paths = []
    for name in names:
        shutil.copyfile(source / name, dest / name)
        paths.append(dest / name)
    return paths[0], paths[1], description


def example(automator, variant, workspace):
    """Writes the example into <workspace>/examples/<automator>-<variant>/ and returns
    {train, test, description, config}."""
    variant = (variant or "2d").lower()
    dest = Path(workspace) / "examples" / f"{automator}-{variant}"
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    if automator == "tabular":
        train, test, description = _tabular(dest)
    elif automator == "image-classification" and variant == "3d":
        train, test, description = _copy("image-classification-3d", ["Train3D.zip", "Test3D.zip"], dest,
                                         "Prostate-like MRI studies: T2 DICOM series + ADC NIfTI, benign vs. malignant.")
    elif automator == "image-classification":
        train, test, description = _image_2d(dest)
    elif automator == "object-detection":
        names = ["Train3D.zip", "Test3D.zip"] if variant == "3d" else ["Train.zip", "Test.zip"]
        train, test, description = _copy("object-detection", names, dest,
                                         "CT-like volumes with CSV 3D boxes (nodules)." if variant == "3d" else
                                         "Radiograph-like images with COCO boxes: nodule and mass.")
    elif automator == "image-segmentation":
        names = ["Train3D.zip", "Test3D.zip"] if variant == "3d" else ["Train.zip", "Test.zip"]
        train, test, description = _copy("image-segmentation", names, dest,
                                         "Prostate-like MRI (T2 DICOM + ADC NIfTI) with gland and lesion masks." if variant == "3d"
                                         else "Aerial-like colour tiles with building and road colour masks.")
    elif automator == "time-series-forecasting":
        train, test, description = _copy("time-series-forecasting", ["Train.csv", "Test.csv"], dest,
                                         "Daily glucose of 40 patients (30 continued, 10 new in Test.csv): insulin and weekend known in advance, "
                                         "steps observed up to now, age/sex/BMI static.")
    else:
        raise KeyError(automator)
    config = dict(QUICK[automator])
    if automator == "time-series-forecasting":
        config["future_columns"] = ["Insulin_units", "Weekend"]
    if automator == "image-segmentation" and variant == "3d":
        config["models"] = ["nnunet_3d", "segresnet"]
    if automator == "image-segmentation" and variant != "3d":
        config["models"] = ["nnunet_2d", "unet_resnet34"]
    if automator == "image-classification" and variant == "3d":
        config["models"] = ["medicalnet_resnet10"]
    if automator == "image-classification" and variant != "3d":
        config["models"] = ["efficientnet_b0"]
    return {"train": str(train), "test": str(test), "description": description, "quick_config": config}
