# Simplatab

**No-code, self-hosted machine learning for research data.** Upload a training set and a test set in the
browser; Simplatab trains and compares many models with cross-validation, evaluates them on your test set,
explains their predictions and gives you the trained models. Your data never leaves the machine running it.

| Automator | Input | Models | Explanations |
|---|---|---|---|
| **Tabular Classification** | `Train.csv`, `Test.csv` | 7 classical (Logistic Regression, SVM, Random Forest, SGD, MLP, Decision Tree, XGBoost) and 4 deep learning (TabPFNv2, TabICL, TabTransformer, TabR) | SHAP |
| **Image Classification** | `Train.zip`, `Test.zip` of medical or other images (DICOM, NIfTI, PNG, JPEG, BMP, TIFF) | 10 pretrained networks: ResNet-50, EfficientNet-B0/V2-S, ConvNeXt(-V2)-Tiny, ViT-Small, DeiT III-Small, Swin-Tiny, MaxViT-Tiny, DINOv2-Small | Grad-CAM |
| **3D Image Classification** (same automator) | `Train.zip`, `Test.zip` of studies with one or more series (DICOM series, NIfTI), e.g. T2 + ADC + DWI | 10 3D networks: MedicalNet ResNet-10/18/50, R3D-18, R(2+1)D-18, MC3-18, Video Swin-T, SwinUNETR Swin-ViT (self-supervised on CT), DenseNet-121 3D, DINOv2-Small 2.5D | 3D Grad-CAM |
| **Time Series Forecasting** | `Train.csv`, `Test.csv` in long format (e.g. repeated measurements of patients), with static, past and future covariates | 10 [neuralforecast](https://github.com/Nixtla/neuralforecast) networks: NHITS, NBEATSx, TiDE, KAN, DLinear, TFT, PatchTST, BiTCN, TCN, TimesNet | Integrated gradients |
| **Object Detection** | `Train.zip`, `Test.zip` of 2D images or 3D volumes (DICOM, NIfTI, PNG, JPEG, …) with boxes in COCO, YOLO, Pascal VOC, CSV or mask format | 10 pretrained detectors: Faster R-CNN v2, RetinaNet v2, FCOS, Faster R-CNN MobileNetV3, SSDLite (torchvision); RT-DETR, RT-DETRv2, D-FINE-M, Deformable DETR, Conditional DETR (transformers) | D-RISE |

Classification: binary and multiclass problems. Image segmentation is planned.

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
4. **Results**: compare metrics and curves, inspect SHAP, Grad-CAM, integrated gradients or D-RISE explanations, download everything.

Each model is trained and validated with **stratified K-fold cross-validation** on the training set. For binary
problems, the decision threshold that maximises the metric of your choice (balanced accuracy by default) is found
on the validation folds. The final model is trained on the whole training set and evaluated once on the test set,
with the mean threshold of the folds. The test set is never used for training or tuning.

Forecasting uses **rolling-origin (prequential) validation** instead: the last K horizons of every training series
are forecast one after the other, each by a network trained on the points before it (with optional tuning of the
lookback, learning rate and size on these windows). The final networks, trained on all the training series,
forecast the last H points of every test series.

For 3D studies, the folds are also **grouped by patient**: all the studies of a patient stay in the same fold.

Object detection offers **K-fold cross-validation** (folds grouped by patient folder, early stopping on a part of
each training fold, final networks retrained on all the images for the median best number of epochs) or a faster
**hold-out validation** (one split; the network trained on it is the final network). The score threshold that
maximises the F1 score on the validation images is used on the test set.

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

**3D studies**: zips of DICOM series and NIfTI volumes (without PNG/JPEG) are classified as 3D studies (they can
also be classified in 2D, one image per file). Layout: `<class>/<patient>/<study>/<series>`, e.g.
`Train.zip/malignant/patient_01/study_1/t2/` (DICOM slices) and `…/study_1/adc.nii.gz`:
- the study level is optional; a series is a folder of DICOM slices, or a NIfTI or multi-frame DICOM file; a folder
  holding several DICOM series (PACS export) is a study whose series are named after their description;
- each study is one sample; the series with the same name in every study (e.g. `t2`, `adc`, `dwi`) are chosen on
  the configuration page and become the input channels: they are read with SimpleITK, reoriented, **aligned in
  patient coordinates** on a reference series (resampled onto its grid), scaled to [0, 1] (CT window, DICOM window or
  percentiles), optionally cropped around the centre and resized (32 × 128 × 128 to 128 × 128 × 128 voxels);
- the first layer of the pretrained networks is adapted to the number of series; feature extraction or fine-tuning
  with 3D augmentation (rotations, scaling, gamma, contrast, optional flips), as in 2D.
- Example data: [`Examples/image-classification-3d`](Examples/image-classification-3d) (prostate-like MRI, T2 DICOM
  series and ADC NIfTI per study), also downloadable from the upload page.

**Time series**: two CSV files in long format, one row per series and time point: `ID` (or `patient_id`), `Time`
(dates at a regular frequency or integer steps; missing points are filled in) and the numeric `Target`.
- Other columns are covariates: constant within a series → **static** (categorical ones one-hot encoded); varying →
  **known in advance** (e.g. a scheduled dose, used over the horizon) or **observed up to now** (e.g. another
  measurement), chosen on the configuration page.
- Test.csv: the last H points (the horizon) of every series are forecast from the points before them. A series whose
  ID is in Train.csv **continues** it (temporal hold-out: Test.csv may hold only its next H points); a **new** ID is
  an unseen series (e.g. a new patient) with its own history.
- Example data: [`Examples/time-series-forecasting`](Examples/time-series-forecasting) (daily glucose of 40 patients),
  also downloadable from the upload page.

**Object detection**: two zip files (up to 5 GB each) with the images and their boxes in one of these formats
(detected automatically; boxes in pixels of the original image):

| Format | Layout |
|---|---|
| COCO JSON | `annotations.json` (`images`, `annotations` with `bbox = [x, y, width, height]`, `categories`) and the images |
| YOLO | `images/…` and `labels/…` (`class cx cy w h`, normalised), class names in `classes.txt` or `data.yaml` |
| Pascal VOC | one `.xml` file next to each image |
| CSV | `image, class, x_min, y_min, x_max, y_max` (+ `z_min, z_max`, first and last slice, for 3D boxes) |
| Masks | `masks/<image name>` label images (PNG or NIfTI); each connected region of a label is a box; names in `classes.txt` |

- **2D**: DICOM, NIfTI, PNG (8 or 16-bit), JPEG, BMP, TIFF. **3D**: NIfTI volumes, multi-frame DICOM or one folder of
  DICOM slices per series, with CSV boxes or NIfTI masks. 3D volumes are detected slice by slice with the
  neighbouring slices as context (2.5D), and the boxes of consecutive slices are merged into 3D boxes.
- Images without boxes (or listed in the CSV without box) are negatives. Sub-folders (one per patient) keep a
  patient's images in the same fold.
- Example data: [`Examples/object-detection`](Examples/object-detection) (2D radiograph-like images with COCO boxes,
  3D CT-like volumes with CSV boxes), also downloadable from the upload page.

## What you get

Everything is written to the `Materials` folder, shown on the results page and downloadable as one zip:

| Output | Files |
|---|---|
| Metrics (AUC, balanced accuracy, F-score, accuracy, sensitivity, specificity; forecasting: MAE, RMSE, sMAPE, MASE vs. a seasonal naive baseline; detection: mAP, AP at IoU 0.5/0.75 (3D: 0.1/0.25/0.5), recall, FROC, precision/recall/F1 and image-level sensitivity/specificity at the threshold) | `<K>_fold_results.xlsx` (mean ± SD) or `holdout_results.xlsx`, `test_results.xlsx`, `Metrics_Plots/` (forecasting) |
| Curves and confusion matrices | `ROC_Curves/`, `ConfusionMatrices/`, `Detection_Curves/` (precision-recall, FROC, AP per class) |
| Explanations | `Shap_Features/<model>/` (tabular), `GradCAM/<network>/` (images; 3D: the slices where the map is strongest), `Explainability/` (forecasting: integrated gradients; detection: D-RISE maps) |
| Trained models, usable without Simplatab | `Models/<model>_pipeline.pkl` + `Models/thresholds.json` (tabular), `Models/<network>.pt` (images, torchvision detectors), `Models/<model>.zip` (forecasting, transformers detectors) |
| Forecasts vs. observed values | `Forecasts/test_forecasts.csv`, `Forecasts/future_forecasts.csv` (beyond the data, without future covariates), `Forecast_Plots/` |
| Image predictions and classes | `Predictions/<network>_test_predictions.csv` (one row per image or 3D study), `classes.csv` |
| Detections drawn on test images (true positives, false positives, missed boxes) | `Detections/` |

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

**3D studies**: each `.pt` file takes a (1, C, D, H, W) volume in [0, 1] (the C series of a study, aligned and
resized as in training). The code of the results page reads, aligns and resizes the series with SimpleITK:
```bash
pip install torch numpy SimpleITK==2.5.2
```
```python
model = torch.jit.load("Materials/Models/MedicalNet_ResNet-10.pt", _extra_files=meta)
info = json.loads(meta["simplatab.json"])   # classes, threshold, series (channels), volume shape, crop, window
print(predict(["new_study/t2", "new_study/adc.nii.gz"]))   # predict() and its helpers: see the results page
```
The code of the results page also reads DICOM (modality LUT, CT or DICOM window, multi-frame) and NIfTI files
exactly as for training.

**Time series**: each `.zip` file is a [neuralforecast](https://github.com/Nixtla/neuralforecast) model with
`simplatab.json` (horizon, covariates, preprocessing).
```bash
pip install neuralforecast==3.1.2 pydantic pandas
```
```python
import shutil
import pandas as pd
from neuralforecast import NeuralForecast

shutil.unpack_archive("Materials/Models/NHITS.zip", "models")
nf = NeuralForecast.load("models/NHITS")
history = pd.read_csv("history.csv", parse_dates=["Time"])   # without parse_dates for integer time steps
history = history.rename(columns={"ID": "unique_id", "Time": "ds", "Target": "y"})
print(nf.predict(df=history))   # the next H points of every series
```
With covariates, the code of the results page also encodes them as in training and passes the static features
(`static_df`) and the covariates known in advance for the forecast period (`futr_df`).

**Object detection**: torchvision detectors are TorchScript `.pt` files; transformers detectors are `.zip` folders
for `from_pretrained` (transformers 4.57.6). Both carry `simplatab.json` (classes, image size, score threshold).
```bash
pip install torch torchvision numpy pillow
```
```python
import json
import numpy as np, torch
import torchvision   # registers the detection operators of the network
from PIL import Image

meta = {"simplatab.json": ""}
model = torch.jit.load("Materials/Models/Faster_R-CNN_R50-FPN_v2.pt", _extra_files=meta).eval()
info = json.loads(meta["simplatab.json"])
size = info["image_size"]

image = Image.open("scan.png").convert("RGB")
x = torch.from_numpy(np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255).permute(2, 0, 1)
with torch.no_grad():
    _, (out,) = model([x])
keep = (out["labels"] > 0) & (out["scores"] >= info["threshold"])   # label 0 is the background
boxes = out["boxes"][keep] * torch.tensor([image.width / size, image.height / size] * 2)
print(boxes, [info["classes"][k - 1] for k in out["labels"][keep]])
```
The code of the results page also covers the transformers detectors, DICOM and NIfTI images and 3D volumes
(slice-by-slice detection and merging into 3D boxes).

## Notes on the deep learning models

- **Forecasting networks** are trained from scratch on your series (no pretrained weights). On CPU most take
  seconds to a minute per training; TimesNet is much slower without a GPU and is not selected by default.
- **TabPFNv2 and TabICL** are pretrained foundation models (no training); TabPFNv2 is limited to 10,000 samples,
  500 features and 10 classes. **TabTransformer** and **TabR** are trained with early stopping.
- **3D networks**: MedicalNet ResNets are pretrained on 23 CT and MRI datasets, the video networks on Kinetics-400
  (slices play the role of frames), the SwinUNETR encoder is self-supervised on 5,050 CT volumes; DenseNet-121 3D
  is trained from scratch (fine-tuning only makes sense with it); the 2.5D model applies DINOv2 to up to 16 slices and
  combines them by attention pooling. Feature extraction takes seconds per study on CPU; fine-tuning needs a GPU.
- **Detectors** are pretrained on COCO and fine-tuned on your boxes, with images resized to a square (320 to
  1024 px). A GPU is strongly recommended: on CPU, fine-tuning takes minutes per epoch for the larger networks;
  Faster R-CNN MobileNetV3 and SSDLite are the fast choices. Hold-out validation trains each network once.
- Pretrained weights (TabPFNv2, TabICL, the 10 image networks, the 3D networks and the 10 detectors) are included
  in the Docker images; from source they are downloaded from the Hugging Face Hub, PyTorch or GitHub on first use.
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
Code layout: `app.py` (web app), `Helpers/` (tabular pipeline), `Helpers/image/` (image pipeline),
`Helpers/image3d/` (3D image pipeline),
`Helpers/forecasting/` (forecasting pipeline), `Helpers/detection/` (object detection pipeline), `web/`
(automator catalog and background jobs), `templates/` and `static/` (interface), `ci/` (release versioning).
`Examples/` holds the outputs of example runs on the Iris and breast cancer datasets, the example time series and
the example detection and 3D data. Set `SIMPLATAB_PRETRAINED=0` to run the detection and 3D automators without downloading weights
(randomly initialised networks, e.g. for tests).

## Authors

**Dimitrios Zaridis** (corresponding author), National Technical University of Athens; **Eugenia Mylona**, PhD;
**Vasileios B. Pezoulas**, PhD. With the assistance of **Charalampos Kalantzopoulos**, MSc; **Nikolaos S. Tachos**,
PhD; and **Dimitrios I. Fotiadis**, Professor of Biomedical Technology, University of Ioannina.

## License

[MIT](LICENSE)
