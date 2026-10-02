# Simplatab

**No-code, self-hosted machine learning for research data** (free for non-commercial research; see [License](#license)). Upload a training set and a test set in the
browser; Simplatab trains and compares many models with cross-validation, evaluates them on your test set,
explains their predictions and gives you the trained models. Your data never leaves the machine running it.

| Automator | Input | Models | Explanations |
|---|---|---|---|
| **Tabular Classification** | `Train.csv`, `Test.csv` | 7 classical (Logistic Regression, SVM, Random Forest, SGD, MLP, Decision Tree, XGBoost) and 4 deep learning (TabPFNv2, TabICL, TabTransformer, TabR) | SHAP |
| **Image Classification** | `Train.zip`, `Test.zip` of medical or other images (DICOM, NIfTI, PNG, JPEG, BMP, TIFF) | 10 pretrained networks: ResNet-50, EfficientNet-B0/V2-S, ConvNeXt(-V2)-Tiny, ViT-Small, DeiT III-Small, Swin-Tiny, MaxViT-Tiny, DINOv2-Small | Grad-CAM |
| **3D Image Classification** (same automator) | `Train.zip`, `Test.zip` of studies with one or more series (DICOM series, NIfTI), e.g. T2 + ADC + DWI | 18 3D networks. Pretrained: MedicalNet ResNet-10/18/50, R3D-18, R(2+1)D-18, MC3-18, Video Swin-T, SwinUNETR Swin-ViT (self-supervised on CT), DINOv2-Small 2.5D. From scratch: MedNeXt-S, ConvNeXt V2 3D, 3D UX-Net, nnU-Net ResEnc-M, SwinUNETR-V2, ViT-Small 3D (UNETR), SEResNeXt-50 3D, EfficientNet-B0 3D, DenseNet-121 3D | 3D Grad-CAM |
| **Time Series Forecasting** | `Train.csv`, `Test.csv` in long format (e.g. repeated measurements of patients), with static, past and future covariates | 10 [neuralforecast](https://github.com/Nixtla/neuralforecast) networks: NHITS, NBEATSx, TiDE, KAN, DLinear, TFT, PatchTST, BiTCN, TCN, TimesNet | Integrated gradients |
| **Object Detection** | `Train.zip`, `Test.zip` of 2D images or 3D volumes (DICOM, NIfTI, PNG, JPEG, …) with boxes in COCO, YOLO, Pascal VOC, CSV or mask format | 10 pretrained detectors: Faster R-CNN v2, RetinaNet v2, FCOS, Faster R-CNN MobileNetV3, SSDLite (torchvision); RT-DETR, RT-DETRv2, D-FINE-M, Deformable DETR, Conditional DETR (transformers) | D-RISE |
| **Survival Analysis** | `Train.csv`, `Test.csv` with `Time` (follow-up) and `Event` (1 event, 0 censored), e.g. overall survival or time to relapse | 8 time-to-event models: Cox PH, Weibull and log-normal AFT, XGBoost Cox and AFT, DeepSurv, DeepHit, Logistic-Hazard (Nnet-survival), with a Kaplan-Meier reference | Risk groups (Kaplan-Meier, log-rank), calibration, permutation importance, hazard ratios |
| **Clustering** | `Train.csv` (and an optional `Test.csv`), with or without a `Target` column of class labels to check the clusters against | 12 classical (K-Means, Bisecting K-Means, Gaussian mixture, Dirichlet-process Bayesian mixture, Ward agglomerative, BIRCH, spectral clustering, affinity propagation, DBSCAN, HDBSCAN, OPTICS, Mean Shift) and 6 deep learning and neural (DEC, IDEC, DCN, VaDE, SCARF + k-means, self-organising map) | Cluster profiles, PCA/t-SNE maps, SHAP |
| **Image Segmentation** | `Train.zip`, `Test.zip` of images and masks: 2D medical or everyday images (PNG, JPEG, DICOM, NIfTI, …), 3D volumes with one or more series (DICOM, NIfTI), or the nnU-Net raw format | 2D: the official **nnU-Net v2**, U-Net ResNet-34, U-Net++ EfficientNet-B4, DeepLabV3+ ResNet-50, FPN and UPerNet ConvNeXt-Tiny, SegFormer-B2, MA-Net ResNet-50 (pretrained), Attention U-Net, U-Net (nnU-Net-like). 3D: **nnU-Net v2** 3D full resolution, SwinUNETR (self-supervised on CT), SwinUNETR-V2, SegResNet, DynUNet, UNETR, MedNeXt-S, Attention U-Net, U-Net++, V-Net | Uncertainty maps (test-time augmentation) |

Classification: binary and multiclass problems. Segmentation: up to 32 classes. Clustering: supervised evaluation
(with labels) or unsupervised clustering (without).

## Three ways to run it

Whichever you choose, open **http://localhost:7111/automl/** (or port 5000 when run as Python) in a browser. The
app runs on Linux, Windows and macOS (Intel and Apple Silicon); your data stays on that machine.

### 1. Pull the image (recommended)

```bash
docker pull dimzaridis/simplatab-machine-learning-automator:latest
docker run -p 7111:5000 dimzaridis/simplatab-machine-learning-automator:latest
```
**NVIDIA GPU** (much faster for the image, detection and segmentation networks):
```bash
docker run --gpus all --shm-size=4g -p 7111:5000 dimzaridis/simplatab-machine-learning-automator:latest-gpu
```
Notes:
- Tags: `latest` and `latest-gpu`, or a version from the
  [releases](https://github.com/dzaridis/simplatab-machine-learning-automator/releases) (e.g. `1.1.8`, `1.1.8-gpu`).
  The CPU image is multi-architecture (amd64, arm64); the GPU image is amd64 and needs the NVIDIA driver and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- `--shm-size=4g` gives the data loaders of the networks enough shared memory.
- Pretrained weights are included in the images, so the automators work offline.
- The results are written inside the container (`/app/Materials`) and downloadable from the results page; add
  `-v "$PWD/Materials:/app/Materials"` to keep them on the host. Each run replaces the previous results.
- Port `7111` is any free port of your machine (`-p <port>:5000`).

### 2. Build the image from source

```bash
git clone https://github.com/dzaridis/simplatab-machine-learning-automator.git
cd simplatab-machine-learning-automator
docker build -t simplatab .                                # CPU
docker build --build-arg DEVICE=gpu -t simplatab:gpu .     # NVIDIA GPU (CUDA 12.8, linux/amd64)
docker run -p 7111:5000 simplatab                          # GPU: docker run --gpus all --shm-size=4g -p 7111:5000 simplatab:gpu
```

### 3. Run it as a Python application

Python 3.9 (the pinned libraries, e.g. numpy 1.23 and nnU-Net 2.4, target it):
```bash
git clone https://github.com/dzaridis/simplatab-machine-learning-automator.git
cd simplatab-machine-learning-automator
python3.9 -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu   # GPU: pip install torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt
python app.py                                             # then open http://localhost:5000/automl/
```
Pretrained weights are downloaded on first use; the results are written to `Materials/` in the current folder.

**For AI agents**: the [MCP server](mcp_server/README.md) exposes every automator as Model Context Protocol tools
(see [below](#mcp-server-for-ai-agents)).

## How it works

1. **Upload** the training and test sets. They are checked in the browser (and on the server) before anything runs.
2. **Configure**: choose the models and the validation settings; every setting is explained on the page.
3. **Run**: follow each model live through the K-fold cross-validation and the external test.
4. **Results**: compare metrics and curves, inspect SHAP, Grad-CAM, integrated gradients, D-RISE explanations or segmentation uncertainty maps, download everything.

Each model is trained and validated with **stratified K-fold cross-validation** on the training set. For binary
problems, the decision threshold that maximises the metric of your choice (balanced accuracy by default) is found
on the validation folds. The final model is trained on the whole training set and evaluated once on the test set,
with the mean threshold of the folds. The test set is never used for training or tuning.

Forecasting uses **rolling-origin (prequential) validation** instead: the last K horizons of every training series
are forecast one after the other, each by a network trained on the points before it (with optional tuning of the
lookback, learning rate and size on these windows). The final networks, trained on all the training series,
forecast the last H points of every test series.

Survival analysis uses **stratified K-fold cross-validation** on the event indicator (the preprocessing refitted in
each fold) and the external test set, with censoring handled by inverse probability of censoring weights: Harrell's
and Uno's C-index, integrated Brier score, and time-dependent AUC and Brier score at the horizons you choose. Test
patients are split into risk groups by the tertiles of the training risks (Kaplan-Meier curves, log-rank test).

Clustering has no training labels: every algorithm clusters Train.csv, its number of clusters given, set to the
number of `Target` classes or chosen in a range by the silhouette (or Calinski-Harabasz, Davies-Bouldin); the
algorithms that find it themselves (density-based, affinity propagation, Bayesian mixture) have their settings tuned
by the same criterion. **K-fold validation** then refits each algorithm on K-1 folds and assigns the held-out samples:
they are scored, and compared with the clusters found on all of Train.csv (**stability**). The final models assign
the Test.csv samples (optional). With a `Target`, the clusters are compared with the classes (ARI, AMI, NMI,
V-measure, purity, matched accuracy); the labels are never used to find the clusters.

For 3D studies, the folds are also **grouped by patient**: all the studies of a patient stay in the same fold.

Object detection offers **K-fold cross-validation** (folds grouped by patient folder, early stopping on a part of
each training fold, final networks retrained on all the images for the median best number of epochs) or a faster
**hold-out validation** (one split; the network trained on it is the final network). The score threshold that
maximises the F1 score on the validation images is used on the test set.

Image segmentation offers the same two modes: **K-fold cross-validation** (folds grouped by patient folder and
stratified on the classes present; final networks retrained on all of Train.zip, nnU-Net as its `all` fold) or
**hold-out validation** (the network trained on the split is the final network).

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

**Survival analysis**: two CSV files, one row per patient: `Time` (follow-up time, positive, any unit), `Event`
(1 if the event happened at `Time`, 0 if censored), an optional `ID`, and the features (numeric or categorical;
missing values imputed). At least 10 events. Example data: [`Examples/survival`](Examples/survival) (overall survival
after colorectal cancer surgery).

**Clustering**: one CSV file, `Train.csv`, one row per sample, and optionally a `Test.csv` with the same feature columns.
- `ID` (or `patient_id`): optional identifier, never a feature. `Target`: optional class labels (numbers or text).
  With it, the clusters are evaluated against the classes (supervised evaluation) and the number of clusters can be
  set to the number of classes; without it, the clustering is unsupervised (internal metrics).
- Every other column is a feature: numeric or categorical (one-hot encoded). Missing values are imputed; constant,
  mostly missing or identifier-like text columns are left out. Scaling (standard, robust, min-max) and an optional
  PCA are chosen on the configuration page, where columns can also be left out.
- Up to 100,000 rows; the quadratic algorithms are limited (spectral 10,000, affinity propagation 5,000 rows...).
- Example data: [`Examples/clustering`](Examples/clustering) (five subgroups of adult-onset diabetes, after Ahlqvist
  et al. 2018), also downloadable from the upload page.

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

**Image segmentation**: two zip files (up to 5 GB each) with the images and their masks, in one of two layouts:
- `images/` and `masks/` with mirrored paths, e.g. `images/case_01.png` and `masks/case_01.png` (mask names may
  end with `_mask`, `_seg`, …); a sub-folder per patient keeps a patient's cases in the same fold. For 3D cases with
  several series, `images/<case>/` holds them (e.g. `t2/` DICOM slices and `adc.nii.gz`) and `masks/<case>.nii.gz`;
- the **nnU-Net raw format**: `imagesTr/case_0000.nii.gz` (one file per channel), `labelsTr/case.nii.gz` and
  `dataset.json` (`imagesTs/`, `labelsTs/` in Test.zip).
- Masks are label images (0 = background; 0/255 binary masks are read as 0/1), palette PNGs or colour masks; the
  classes are named in `labels.json` (`{"1": "liver"}`), `classes.txt` (`1,liver`) or `dataset.json`.
- 3D series are reoriented and **aligned in patient coordinates** on a reference series (the input and output grid);
  the series with the same name in every case are the input channels.
- **nnU-Net** plans its own preprocessing, network and training from your data (a shorter schedule than its 1000
  epochs can be chosen). The other networks follow its recipe: resampling to the median spacing, CT or z-score
  normalisation, patches planned from the median size (a third centred on a structure), Dice + cross-entropy loss,
  sliding-window inference. **Test-time flips** give the uncertainty maps (entropy of the averaged probabilities).
- Example data: [`Examples/image-segmentation`](Examples/image-segmentation) (2D aerial-like tiles with building and
  road colour masks; 3D prostate-like MRI, T2 DICOM series and ADC NIfTI, with gland and lesion masks), also
  downloadable from the upload page.

## What you get

Everything is written to the `Materials` folder, shown on the results page and downloadable as one zip:

| Output | Files |
|---|---|
| Metrics (AUC, balanced accuracy, F-score, accuracy, sensitivity, specificity; forecasting: MAE, RMSE, sMAPE, MASE vs. a seasonal naive baseline; detection: mAP, AP at IoU 0.5/0.75 (3D: 0.1/0.25/0.5), recall, FROC, precision/recall/F1 and image-level sensitivity/specificity at the threshold; segmentation: Dice, IoU, HD95, ASSD, sensitivity, precision; clustering: silhouette, Calinski-Harabasz, Davies-Bouldin, stability and, with labels, ARI, AMI, NMI, V-measure, homogeneity, completeness, FMI, purity, matched accuracy) | `<K>_fold_results.xlsx` (mean ± SD) or `holdout_results.xlsx`, `test_results.xlsx`, `train_results.xlsx` (clustering), `Metrics_Plots/` (forecasting, clustering) |
| Validation splits: the samples of every fold (to reproduce the validation) | `Splits/splits.csv`, `Splits/splits.json` |
| Curves and confusion matrices | `ROC_Curves/`, `ConfusionMatrices/`, `Detection_Curves/` (precision-recall, FROC, AP per class), `Segmentation_Plots/` |
| Explanations | `Shap_Features/<model>/` (tabular), `GradCAM/<network>/` (images; 3D: the slices where the map is strongest), `Explainability/` (forecasting: integrated gradients; detection: D-RISE maps; clustering: SHAP of a random forest that recognises the clusters) |
| Trained models, usable without Simplatab | `Models/<model>_pipeline.pkl` + `Models/thresholds.json` (tabular), `Models/<network>.pt` (images, torchvision detectors, segmentation networks), `Models/<model>.zip` (forecasting, transformers detectors, nnU-Net model folders), `Models/<algorithm>.pkl` (clustering), `Models/<model>.pkl` (survival: `predict_risk`, `predict_survival`) |
| Forecasts vs. observed values | `Forecasts/test_forecasts.csv`, `Forecasts/future_forecasts.csv` (beyond the data, without future covariates), `Forecast_Plots/` |
| Clusters (clustering) | `Clusters/train_clusters.csv`, `test_clusters.csv` (cluster of every sample per algorithm; 0 the largest, -1 noise), `Cluster_Profiles/` (feature means per cluster), `Embeddings/` (PCA and t-SNE maps), `Metrics_Plots/` (clusters vs. classes, silhouettes, choice of k) |
| Image predictions and classes | `Predictions/<network>_test_predictions.csv` (one row per image or 3D study), `classes.csv` |
| Detections drawn on test images (true positives, false positives, missed boxes) | `Detections/` |
| Predicted test masks (original mask values; 3D: NIfTI on the grid of the first series), metrics per class and case, overlays with the uncertainty map (worst, median and best test cases) | `Predictions/<network>/`, `Segmentation_Metrics/`, `Overlays/<network>/` |

Each run replaces the results of the previous one.

## Validation splits

Every run writes the samples of each fold to `Materials/Splits/`, so that the validation can be reproduced exactly
(and is shown in the Downloads tab of the results page):
- `splits.csv`: one row per sample and fold: `fold`, `set` (`train`, `validation`, and `early_stopping` where a part
  of the training fold stops the training early) and `id`, plus `patient` (the group kept in one fold), `class`,
  `row` (the line of Train.csv, tabular and clustering) or the `start`/`end` times of each series (forecasting windows);
- `splits.json`: the ids of every fold, with a description of how the folds were made (seeds included).

`id` is what identifies a sample in your data: the `ID` (or `patient_id`) column of Train.csv, the image, study or
case path inside Train.zip, or the series ID.

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

**Clustering**: each `.pkl` file holds the preprocessing fitted on Train.csv and the clustering model (stored with
cloudpickle); `predict` gives the cluster of new rows, numbered as in `Clusters/train_clusters.csv`.
```bash
pip install numpy==1.23.5 pandas==2.0.3 scikit-learn==1.3.1 cloudpickle==3.1.2   # + torch==2.8.0 for DEC, IDEC, DCN, VaDE, SCARF
```
```python
import pickle
import pandas as pd

model = pickle.load(open("Materials/Models/K-Means.pkl", "rb"))
data = pd.read_csv("new_samples.csv", index_col="ID")   # the columns of Train.csv; Target not needed
print(model.predict(data))   # model.predict_proba(data): memberships (mixtures, deep networks)
```

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

**Image segmentation**: the `.pt` networks are TorchScript files (normalised patch → class logits) with
`simplatab.json` (patch size, normalisation, spacing, classes and mask values); the code of the results page reads,
aligns, normalises and resamples a new case, predicts by overlapping patches with MONAI and writes the mask. The
nnU-Net `.zip` files are nnU-Net v2 model folders, also usable with `nnUNetv2_predict`:
```bash
pip install nnunetv2==2.4.2 acvl-utils==0.2 "numpy<2" SimpleITK==2.5.2
```
```python
import functools, shutil
import torch
torch.load = functools.partial(torch.load, weights_only=False)   # nnU-Net 2.4 checkpoints hold training metadata
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

if __name__ == "__main__":   # nnU-Net starts worker processes
    shutil.unpack_archive("Materials/Models/nnU-Net_3D_full_resolution.zip", "nnunet_model")
    predictor = nnUNetPredictor(device=torch.device("cpu"), perform_everything_on_device=False)
    predictor.initialize_from_trained_model_folder("nnunet_model", use_folds=("all",))   # (0,) after a hold-out run
    predictor.predict_from_files([["case_0000.nii.gz", "case_0001.nii.gz"]],   # one file per series, aligned
                                 ["predictions/case"])                           # -> predictions/case.nii.gz
```
The class indices of nnU-Net masks follow the classes of the run (`run_info.json`); the code of the results page
maps them back to your mask values and aligns the series as in training.

## Notes on the deep learning models

- **Forecasting networks** are trained from scratch on your series (no pretrained weights). On CPU most take
  seconds to a minute per training; TimesNet is much slower without a GPU and is not selected by default.
- **Deep clustering networks** (DEC, IDEC, DCN, VaDE, SCARF) are MLPs trained from scratch on your table: an
  autoencoder (contrastive encoder for SCARF) is pretrained, then refined to cluster; with an automatic number of
  clusters, k is chosen on the pretrained embedding. They take seconds to a minute on CPU for a few thousand rows.
  The self-organising map (MiniSom) is a numpy model.
- **TabPFNv2 and TabICL** are pretrained foundation models (no training); TabPFNv2 is limited to 10,000 samples,
  500 features and 10 classes. **TabTransformer** and **TabR** are trained with early stopping.
- **3D networks**: MedicalNet ResNets are pretrained on 23 CT and MRI datasets, the video networks on Kinetics-400
  (slices play the role of frames), the SwinUNETR encoder is self-supervised on 5,050 CT volumes; the 2.5D model
  applies DINOv2 to up to 16 slices and combines them by attention pooling. Feature extraction takes seconds per
  study on CPU; fine-tuning needs a GPU.
- **3D architectures trained from scratch** (no pretrained weights, so fine-tuning only, ideally with a few hundred
  studies): the encoders of state-of-the-art medical networks, MedNeXt-S (MICCAI 2023), ConvNeXt V2 3D (2023), 3D UX-Net
  (ICLR 2023), nnU-Net ResEnc-M (2024), SwinUNETR-V2 (2023) and ViT-Small 3D (UNETR), and the SEResNeXt-50,
  EfficientNet-B0 and DenseNet-121 3D CNNs (MONAI).
- **Detectors** are pretrained on COCO and fine-tuned on your boxes, with images resized to a square (320 to
  1024 px). A GPU is strongly recommended: on CPU, fine-tuning takes minutes per epoch for the larger networks;
  Faster R-CNN MobileNetV3 and SSDLite are the fast choices. Hold-out validation trains each network once.
- **Segmentation**: nnU-Net is the reference of medical segmentation challenges but is slow on CPU (choose fewer
  epochs, or the GPU image); its 3D networks need a GPU for real data. The 2D encoders are pretrained on ImageNet
  (trained with 30% of the learning rate), SwinUNETR uses the CT self-supervised encoder; the other networks are
  trained from scratch with a fixed number of epochs × iterations, as nnU-Net.
- Pretrained weights (TabPFNv2, TabICL, the 10 image networks, the 3D networks, the 10 detectors and the 2D
  segmentation encoders) are included
  in the Docker images; from source they are downloaded from the Hugging Face Hub, PyTorch or GitHub on first use.
- A GPU is used automatically when available. A model that cannot run on a dataset is skipped and reported,
  and the others still complete.

## MCP server for AI agents

[`mcp_server/`](mcp_server/README.md) turns every automator into [Model Context Protocol](https://modelcontextprotocol.io)
tools, so that agents run experiments by themselves: the server suggests the automator from the data
(`inspect_data`), gives its **data contract** (`get_data_contract`: layout, formats, rules, configuration fields and
models), checks the data and the configuration, runs the experiment and returns the results (metrics, best model,
splits, figures, trained models). It runs locally in Docker, in the background:
```bash
cp mcp_server/.env.example mcp_server/.env                       # your data and output folders
docker compose -f mcp_server/compose.yaml up -d --build          # GPU: add -f mcp_server/compose.gpu.yaml
claude mcp add --transport http simplatab http://localhost:8000/mcp
```
It runs the same pipelines as the web application; see [mcp_server/README.md](mcp_server/README.md).

## Releases

Every push to `main` whose tests pass is released automatically: the version is the latest release plus `0.0.1`,
and the Docker images (`<version>`, `latest`, `<version>-gpu`, `latest-gpu`) and the GitHub release get the same
version. For a new minor or major version, create a tag such as `1.2.0`; the next push is released as `1.2.1`.

## Development

```bash
python -m unittest discover tests
```
Code layout: `app.py` (web app), `Helpers/` (tabular pipeline; `splits.py`: the validation splits of every automator), `Helpers/image/` (image pipeline),
`Helpers/image3d/` (3D image pipeline),
`Helpers/forecasting/` (forecasting pipeline), `Helpers/clustering/` (clustering pipeline), `Helpers/survival/` (survival pipeline), `Helpers/detection/` (object detection pipeline),
`Helpers/segmentation/` (segmentation pipeline; `nnunet_runner.py` runs nnU-Net in a separate process), `web/`
(automator catalog and background jobs), `templates/` and `static/` (interface), `ci/` (release versioning),
`mcp_server/` (MCP server, its Dockerfile and tests).
`Examples/` holds the outputs of example runs on the Iris and breast cancer datasets, the example time series and
the example detection, 3D, segmentation and clustering data. Set `SIMPLATAB_PRETRAINED=0` to run the detection, 3D and segmentation automators without downloading weights
(randomly initialised networks, e.g. for tests).

## Authors

**Dimitrios Zaridis** (corresponding author), National Technical University of Athens; **Eugenia Mylona**, PhD;
**Vasileios B. Pezoulas**, PhD. With the assistance of **Charalampos Kalantzopoulos**, MSc; **Nikolaos S. Tachos**,
PhD; and **Dimitrios I. Fotiadis**, Professor of Biomedical Technology, University of Ioannina.

## License

Simplatab is free for **non-commercial use only**: research, teaching, personal study, and use by universities,
hospitals, public research and other non-profit or public organisations. Commercial use (selling it, offering it as a
paid service or using it to make money) is not allowed. The terms are the
[PolyForm Noncommercial License 1.0.0](LICENSE). For any other use, contact the author.

Versions released before this change remain available under the MIT license they were published with.
