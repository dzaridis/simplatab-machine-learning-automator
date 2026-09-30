# Simplatab

**No-code, self-hosted machine learning for research data.** Upload a training set and a test set in the
browser; Simplatab trains and compares many models with cross-validation, evaluates them on your test set,
explains their predictions and gives you the trained models. Your data never leaves the machine running it.

| Automator | Input | Models | Explanations |
|---|---|---|---|
| **Tabular Classification** | `Train.csv`, `Test.csv` | 7 classical (Logistic Regression, SVM, Random Forest, SGD, MLP, Decision Tree, XGBoost) and 4 deep learning (TabPFNv2, TabICL, TabTransformer, TabR) | SHAP |
| **Image Classification** | `Train.zip`, `Test.zip` of medical or other images (DICOM, NIfTI, PNG, JPEG, BMP, TIFF) | 10 pretrained networks: ResNet-50, EfficientNet-B0/V2-S, ConvNeXt(-V2)-Tiny, ViT-Small, DeiT III-Small, Swin-Tiny, MaxViT-Tiny, DINOv2-Small | Grad-CAM |

Binary and multiclass problems are supported. Image segmentation and longitudinal forecasting are planned.

## Quick start

```bash
docker run -p 7111:5000 dimzaridis/simplatab-machine-learning-automator:latest
```
Open **http://localhost:7111/automl/**. The image runs on Linux, Windows and macOS (Intel and Apple Silicon).

**NVIDIA GPU** (much faster to fine-tune image networks; needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):
```bash
docker run --gpus all --shm-size=4g -p 7111:5000 dimzaridis/simplatab-machine-learning-automator:latest-gpu
```

<details>
<summary>Other ways to run it: a specific version, building the image, from source</summary>

- **A specific version**: replace `latest` with a version from the
  [releases](https://github.com/dzaridis/simplatab-machine-learning-automator/releases) (`1.1.2`, or `1.1.2-gpu`).
- **Build the image**: `docker build -t simplatab .` (GPU: `docker build --build-arg DEVICE=gpu -t simplatab:gpu .`).
- **From source** (Python 3.9):
  ```bash
  git clone https://github.com/dzaridis/simplatab-machine-learning-automator.git
  cd simplatab-machine-learning-automator
  pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
  pip install -r requirements.txt
  python app.py   # then open http://localhost:5000/automl/
  ```
</details>

## How it works

1. **Upload** the training and test sets. They are checked in the browser (and on the server) before anything runs.
2. **Configure**: choose the models and the validation settings; every setting is explained on the page.
3. **Run**: follow each model live through the K-fold cross-validation and the external test.
4. **Results**: compare metrics and curves, inspect SHAP or Grad-CAM explanations, download everything.

Each model is trained and validated with **stratified K-fold cross-validation** on the training set. For binary
problems, the decision threshold that maximises the metric of your choice (balanced accuracy by default) is found
on the validation folds. The final model is trained on the whole training set and evaluated once on the test set,
with the mean threshold of the folds. The test set is never used for training or tuning.

## Your data

**Tabular**: two CSV files with the same columns:
- a numeric `Target` column: `0`/`1` for binary problems (1 = positive class), `0, 1, …, K-1` for multiclass;
- numeric and categorical features; an optional `ID` (or `patient_id`) column used as an identifier;
- rows with missing values are removed; categorical columns whose values differ between the files are dropped.

Optional steps: data bias assessment on a feature of your choice, correlation-based feature selection
(featurewiz), randomized or exhaustive hyperparameter search.

**Images**: two zip files (up to 5 GB each) with **one folder per class**, e.g. `Train.zip/benign/…`,
`Train.zip/malignant/…`; sub-folders (one per patient) are allowed and every image file is one sample.
- Medical formats: DICOM (compressed, multi-frame, colour, MONOCHROME1, without extension), NIfTI, 16-bit PNG/TIFF.
  CT windows (lung, soft tissue, bone, brain) or the DICOM window; 3D volumes reduced to the middle slice or the
  maximum intensity projection; images padded to a square and resized to 224 × 224.
- **Feature extraction** (fast, CPU friendly): the pretrained network is frozen and a logistic regression learns
  your classes from its features. **Fine-tuning** (GPU recommended): the whole network is retrained with data
  augmentation and early stopping.
- For binary problems, choose the positive class (the class to detect) on the configuration page.
- Keep all the images of a patient in the same zip; the upload warns about identical images in both zips.

## What you get

Everything is written to the `Materials` folder, shown on the results page and downloadable as one zip:

| Output | Files |
|---|---|
| Metrics (AUC, balanced accuracy, F-score, accuracy, sensitivity, specificity) | `<K>_fold_results.xlsx` (mean ± SD), `test_results.xlsx` |
| Curves and confusion matrices | `ROC_Curves/`, `ConfusionMatrices/` |
| Explanations | `Shap_Features/<model>/` (tabular), `GradCAM/<network>/` (images) |
| Trained models, usable without Simplatab | `Models/<model>_pipeline.pkl` + `Models/thresholds.json` (tabular), `Models/<network>.pt` (images) |
| Image predictions and classes | `Predictions/<network>_test_predictions.csv`, `classes.csv` |

Each run replaces the results of the previous one.

## Using the trained models

The models run **without Simplatab**, with pip packages only. The results page shows ready-to-copy code for
your best model, adapted to your data (identifier column, image formats, CT window).

**Tabular**: each `.pkl` file is a scikit-learn pipeline (feature selection, preprocessing, classifier).
```bash
pip install numpy==1.23.5 pandas==2.0.3 scikit-learn==1.3.1 xgboost==1.7.6   # deep learning models: + torch==2.8.0 cloudpickle==3.1.2 (and tabpfn / tabicl)
```
```python
import json, pickle
import pandas as pd

model = pickle.load(open("Materials/Models/XGBoost_pipeline.pkl", "rb"))
threshold = json.load(open("Materials/Models/thresholds.json"))["XGBoost"]   # binary problems
data = pd.read_csv("new_samples.csv", index_col="ID").dropna()               # columns of Train.csv
p = model.predict_proba(data)
print((p[:, 1] > threshold).astype(int))                                    # multiclass: p.argmax(1)
```

**Images**: each `.pt` file is a TorchScript model (normalisation, network, softmax) for 224 × 224 RGB images
in [0, 1], with its classes and decision threshold.
```bash
pip install torch numpy pillow   # + pydicom==2.4.4 pylibjpeg for DICOM, nibabel for NIfTI
```
```python
import json
import numpy as np, torch
from PIL import Image

meta = {"simplatab.json": ""}
model = torch.jit.load("Materials/Models/DINOv2-Small.pt", _extra_files=meta)
info = json.loads(meta["simplatab.json"])   # classes, threshold, preprocessing

image = Image.open("scan.png").convert("RGB")
side = max(image.size)                       # pad to a square, then resize as in training
square = Image.new("RGB", (side, side))
square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
square = square.resize((256, 256), Image.BILINEAR).resize((224, 224), Image.BILINEAR)
x = torch.from_numpy(np.asarray(square, dtype=np.float32) / 255).permute(2, 0, 1)[None]
p = model(x)[0].detach().numpy()
k = int(p[1] > info["threshold"]) if info["threshold"] is not None else int(p.argmax())
print(info["classes"][k], p)
```
The code of the results page also reads DICOM (modality LUT, CT or DICOM window, multi-frame) and NIfTI files
exactly as for training.

## Notes on the deep learning models

- **TabPFNv2 and TabICL** are pretrained foundation models (no training); TabPFNv2 is limited to 10,000 samples,
  500 features and 10 classes. **TabTransformer** and **TabR** are trained with early stopping.
- Pretrained weights (TabPFNv2, TabICL and the 10 image networks) are included in the Docker images; from source
  they are downloaded from the Hugging Face Hub on first use.
- A GPU is used automatically when available. A model that cannot run on a dataset is skipped and reported,
  and the others still complete.

## Releases

Every push to `main` whose tests pass is released automatically: the version is the latest release plus `0.0.1`,
and the Docker images (`<version>`, `latest`, `<version>-gpu`, `latest-gpu`) and the GitHub release get the same
version. For a new minor or major version, create a tag such as `1.2.0`; the next push is released as `1.2.1`.

## Development

```bash
python -m unittest discover tests
```
Code layout: `app.py` (web app), `Helpers/` (tabular pipeline), `Helpers/image/` (image pipeline), `web/`
(automator catalog and background jobs), `templates/` and `static/` (interface), `ci/` (release versioning).
`Examples/` holds the outputs of example runs on the Iris and breast cancer datasets.

## Authors

**Dimitrios Zaridis** (corresponding author), National Technical University of Athens; **Eugenia Mylona**, PhD;
**Vasileios B. Pezoulas**, PhD. With the assistance of **Charalampos Kalantzopoulos**, MSc; **Nikolaos S. Tachos**,
PhD; and **Dimitrios I. Fotiadis**, Professor of Biomedical Technology, University of Ioannina.

## License

[MIT](LICENSE)
