"""Training of the image classifiers: feature extraction + logistic regression, or
fine-tuning of the whole network. Uses the GPU when one is available."""
import copy
import os

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .models import INPUT_SIZE, ImageClassifier, normalization


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device():
    if torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return f"CPU ({os.cpu_count()} cores)"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transform(mean, std, train=False, augmentation=None):
    """Evaluation: resize to 224 x 224. Training: the selected augmentations."""
    if not train:
        return transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    augmentation = augmentation or {}
    steps = [transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1))
             if augmentation.get("crop", True) else transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if augmentation.get("horizontal_flip"):
        steps.append(transforms.RandomHorizontalFlip())
    if augmentation.get("vertical_flip"):
        steps.append(transforms.RandomVerticalFlip())
    if augmentation.get("rotation", True):
        steps.append(transforms.RandomRotation(10))
    if augmentation.get("intensity", True):
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class ImageDataset(Dataset):
    """Cached PNG images (grayscale images are repeated on the 3 input channels)."""

    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), np.asarray(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(self.labels[index])


def _shared_memory_bytes():
    try:
        stats = os.statvfs("/dev/shm")
        return stats.f_frsize * stats.f_blocks
    except (OSError, AttributeError):
        return 0


def _workers():
    """DataLoader worker processes: SIMPLATAB_NUM_WORKERS, else 4 on GPU (to keep it busy)
    and 0 on CPU (the network is the bottleneck there). Workers exchange batches through
    /dev/shm, which Docker limits to 64 MB unless the container is started with --shm-size:
    without enough shared memory, the images are loaded in the main process."""
    configured = os.environ.get("SIMPLATAB_NUM_WORKERS")
    if configured is not None:
        return max(0, int(configured))
    if not torch.cuda.is_available() or _shared_memory_bytes() < 1024 ** 3:
        return 0
    return min(4, os.cpu_count() or 1)


def make_loader(paths, labels, transform, batch_size, shuffle=False, drop_last=False):
    workers = _workers()
    return DataLoader(ImageDataset(paths, labels, transform), batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last and len(paths) > batch_size, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      multiprocessing_context="spawn" if workers > 0 else None)


def _autocast():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


# ---------------------------------------------------------------------------
# Feature extraction + logistic regression
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_features(model, paths, batch_size=32, log=None):
    """Pooled features of the frozen backbone for every image."""
    mean, std = normalization(model.backbone)
    loader = make_loader(paths, np.zeros(len(paths), dtype=int), build_transform(mean, std), batch_size)
    model.eval().to(device())
    features, done = [], 0
    for x, _ in loader:
        with _autocast():
            features.append(model.features(x.to(device(), non_blocking=True)).float().cpu().numpy())
        done += len(x)
        if log and (done // batch_size) % 20 == 0:
            log(f"Extracted features of {done}/{len(paths)} images")
    return np.concatenate(features) if features else np.zeros((0, model.backbone.num_features))


def fit_linear_head(features, labels, seed=10):
    """Standardised logistic regression with class weights; its regularisation is chosen by
    an internal 3-fold cross-validation on AUC."""
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    counts = np.bincount(labels)
    if counts[counts > 0].min() >= 3:
        scoring = "roc_auc" if len(np.unique(labels)) == 2 else "roc_auc_ovr"
        search = GridSearchCV(pipeline, {"lr__C": [0.001, 0.01, 0.1, 1.0]}, scoring=scoring,
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=1)
        search.fit(features, labels)
        return search.best_estimator_
    return pipeline.fit(features, labels)


def linear_head_weights(pipeline, num_classes):
    """The logistic regression (and its standardisation) as the weights of a linear layer
    whose softmax equals ``predict_proba``."""
    scaler, lr = pipeline.named_steps["scaler"], pipeline.named_steps["lr"]
    weight = lr.coef_ / scaler.scale_
    bias = lr.intercept_ - (lr.coef_ * scaler.mean_ / scaler.scale_).sum(axis=1)
    if weight.shape[0] == 1:  # binary: softmax([0, z]) = sigmoid(z)
        weight = np.vstack([np.zeros_like(weight), weight])
        bias = np.concatenate([[0.0], bias])
    if weight.shape[0] != num_classes:
        raise ValueError("the training data does not contain every class")
    return weight.astype(np.float32), bias.astype(np.float32)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_proba(model, paths, batch_size=32):
    mean, std = normalization(model.backbone)
    loader = make_loader(paths, np.zeros(len(paths), dtype=int), build_transform(mean, std), batch_size)
    model.eval().to(device())
    probabilities = []
    for x, _ in loader:
        with _autocast():
            logits = model(x.to(device(), non_blocking=True))
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def _score(labels, probabilities):
    """Early-stopping score: AUC (one-vs-rest for multiclass)."""
    try:
        if probabilities.shape[1] == 2:
            return roc_auc_score(labels, probabilities[:, 1])
        return roc_auc_score(labels, probabilities, multi_class="ovr", labels=list(range(probabilities.shape[1])))
    except ValueError:
        return float(np.mean(np.argmax(probabilities, axis=1) == labels))


def fine_tune(backbone, paths, labels, num_classes, settings, epochs=None, log=print, label="", pretrained=True,
              seed=10):
    """Trains the whole network. Without ``epochs``, 10% of the images are held out for early
    stopping and the best epoch is kept; with ``epochs``, trains for that many epochs on all
    the images. Returns the model and the number of epochs it was trained for."""
    torch.manual_seed(seed)
    labels = np.asarray(labels)
    paths = np.asarray(paths)
    early_stopping = epochs is None
    stop_paths = stop_labels = None
    if early_stopping:
        counts = np.bincount(labels, minlength=num_classes)
        # 10% of the images, and at least one per class (stratified split)
        held_out = max(num_classes, int(round(0.1 * len(labels))))
        if counts.min() >= 2 and len(labels) - held_out >= 2 * num_classes:
            paths, stop_paths, labels, stop_labels = train_test_split(
                paths, labels, test_size=held_out, stratify=labels, random_state=seed)
        else:
            log(f"{label} too few images to hold some out for early stopping: training for {epochs or settings['epochs']} epochs")
        epochs = settings["epochs"]

    model = ImageClassifier(backbone, num_classes, pretrained=pretrained).to(device())
    mean, std = normalization(model.backbone)
    loader = make_loader(paths, labels, build_transform(mean, std, train=True, augmentation=settings["augmentation"]),
                         settings["batch_size"], shuffle=True, drop_last=True)

    # The pretrained backbone learns 10 times slower than the new linear layer
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": settings["learning_rate"]},
        {"params": model.head.parameters(), "lr": settings["learning_rate"] * 10},
    ], weight_decay=0.05)
    steps = max(1, epochs * len(loader))
    warmup = min(len(loader), steps // 10 + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(step, steps) / steps)))
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = torch.tensor(len(labels) / (num_classes * np.maximum(counts, 1)), device=device())
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_state, best_score, best_epoch, waited = None, -np.inf, epochs, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device(), non_blocking=True), y.to(device(), non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(y)
            seen += len(y)
        message = f"{label} epoch {epoch}/{epochs}: training loss {total / max(seen, 1):.4f}"
        if stop_paths is not None:
            score = _score(stop_labels, predict_proba(model, stop_paths, settings["batch_size"]))
            message += f", early-stopping AUC {score:.3f}"
            if score > best_score:
                best_score, best_epoch, waited = score, epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                waited += 1
        log(message)
        if stop_paths is not None and waited >= settings["patience"]:
            log(f"{label} early stopping: best epoch {best_epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch
