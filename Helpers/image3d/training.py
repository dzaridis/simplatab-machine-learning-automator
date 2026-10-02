"""Training of the 3D classifiers on the cached volumes: feature extraction + logistic regression,
or fine-tuning with 3D augmentations. The logistic regression, the early-stopping score and the
device handling are those of the 2D automator."""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from Helpers.image.training import _autocast, _score, _workers, describe_device, device  # noqa: F401
from Helpers.image.training import fit_linear_head, linear_head_weights  # noqa: F401
from .models import VolumeClassifier


def augment(x, settings, generator):
    """Random 3D augmentations of a (C, D, H, W) volume in [0, 1]: left-right / anterior-posterior
    flips, in-plane rotation (±10°) and scaling (±10%), gamma and contrast changes."""
    settings = settings or {}
    if settings.get("horizontal_flip", True) and torch.rand(1, generator=generator) < 0.5:
        x = x.flip(-1)
    if settings.get("vertical_flip") and torch.rand(1, generator=generator) < 0.5:
        x = x.flip(-2)
    if settings.get("rotation", True):
        angle = (torch.rand(1, generator=generator).item() * 2 - 1) * np.pi / 18
        scale = 1 + (torch.rand(1, generator=generator).item() * 2 - 1) * 0.1
        cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
        theta = torch.tensor([[[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0]]], dtype=torch.float32)
        grid = F.affine_grid(theta, (1,) + tuple(x.shape), align_corners=False)
        x = F.grid_sample(x[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    if settings.get("intensity", True):
        gamma = float(np.exp((torch.rand(1, generator=generator).item() * 2 - 1) * 0.2))
        contrast = 1 + (torch.rand(1, generator=generator).item() * 2 - 1) * 0.1
        x = (x.clamp(0, 1) ** gamma - 0.5) * contrast + 0.5
    return x.clamp(0, 1)


class VolumeDataset(Dataset):
    """Cached (C, D, H, W) float16 volumes."""

    def __init__(self, paths, labels, augmentation=None, seed=0):
        self.paths, self.labels, self.augmentation = list(paths), np.asarray(labels), augmentation
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        x = torch.from_numpy(np.load(self.paths[index]).astype(np.float32))
        if self.augmentation is not None:
            x = augment(x, self.augmentation, self.generator)
        return x, int(self.labels[index])


def make_loader(paths, labels, batch_size, augmentation=None, shuffle=False, drop_last=False):
    workers = _workers()
    return DataLoader(VolumeDataset(paths, labels, augmentation), batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last and len(paths) > batch_size, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      multiprocessing_context="spawn" if workers > 0 else None)


@torch.inference_mode()
def extract_features(model, paths, batch_size=4, log=None):
    """Pooled features of the frozen encoder for every volume."""
    loader = make_loader(paths, np.zeros(len(paths), dtype=int), batch_size)
    model.eval().to(device())
    features, done = [], 0
    for x, _ in loader:
        with _autocast():
            features.append(model.features(x.to(device(), non_blocking=True)).float().cpu().numpy())
        done += len(x)
        if log and (done // batch_size) % 25 == 0:
            log(f"Extracted features of {done}/{len(paths)} volumes")
    return np.concatenate(features) if features else np.zeros((0, model.encoder.num_features))


@torch.inference_mode()
def predict_proba(model, paths, batch_size=4):
    loader = make_loader(paths, np.zeros(len(paths), dtype=int), batch_size)
    model.eval().to(device())
    probabilities = []
    for x, _ in loader:
        with _autocast():
            logits = model(x.to(device(), non_blocking=True))
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def fine_tune(spec, paths, labels, num_classes, channels, shape, settings, epochs=None, log=print, label="",
              pretrained=True, seed=10):
    """Trains the whole network. Without ``epochs``, 10% of the volumes are held out for early
    stopping and the best epoch is kept; with ``epochs``, trains for that many epochs on all
    the volumes. Returns the model and the number of epochs it was trained for."""
    torch.manual_seed(seed)
    labels, paths = np.asarray(labels), np.asarray(paths)
    early_stopping = epochs is None
    stop_paths = stop_labels = None
    if early_stopping:
        counts = np.bincount(labels, minlength=num_classes)
        held_out = max(num_classes, int(round(0.1 * len(labels))))
        if counts.min() >= 2 and len(labels) - held_out >= 2 * num_classes:
            paths, stop_paths, labels, stop_labels = train_test_split(
                paths, labels, test_size=held_out, stratify=labels, random_state=seed)
        else:
            log(f"{label} too few volumes to hold some out for early stopping: training for {settings['epochs']} epochs")
        epochs = settings["epochs"]

    model = VolumeClassifier(spec, channels, shape, num_classes, pretrained=pretrained).to(device())
    loader = make_loader(paths, labels, settings["batch_size"], augmentation=settings["augmentation"],
                         shuffle=True, drop_last=True)
    head = list(model.head.parameters())
    if spec.family == "2.5d":  # the attention pooling is new: it learns at the rate of the linear layer
        head += [p for n, p in model.encoder.named_parameters() if n.startswith("attention_")]
    head_ids = {id(p) for p in head}
    encoder = [p for p in model.parameters() if id(p) not in head_ids]
    # Pretrained encoders learn 10 times slower than the new layers; from scratch, CNNs at the same rate
    # and transformers (less stable at high rates) 3 times slower
    encoder_rate = settings["learning_rate"] * {"scratch": 10, "scratch_transformer": 3}.get(spec.family, 1)
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": encoder_rate},
                                   {"params": head, "lr": settings["learning_rate"] * 10}], weight_decay=0.05)
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
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
