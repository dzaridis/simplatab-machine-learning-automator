"""Training and inference of the detection networks."""
import copy
import os
import time

import numpy as np
import torch

from . import metrics
from .dataset import Augmentation, DetectionDataset, Unit, collate, load_unit_image
from .merge import merge_slices
from .models import build, input_size


def describe_device():
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return f"CPU ({os.cpu_count()} cores)"


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(items, units, size, batch_size, augmentation=None, shuffle=False, seed=0):
    dataset = DetectionDataset(items, units, size, augmentation, seed)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                                       num_workers=0, generator=generator)


@torch.no_grad()
def predict_units(model, items, units, size, batch_size=8):
    """Detections of each unit (image or slice), in the pixels of the original image."""
    model.eval()
    out = {}
    for images, _, indices in _loader(items, units, size, batch_size):
        for index, (boxes, scores, labels) in zip(indices, model.predict(images.to(device()))):
            unit = units[index]
            height, width = items[unit.item].shape[-2:]
            scale = np.array([width / size, height / size] * 2)
            out[index] = (boxes.cpu().numpy() * scale, scores.cpu().numpy(), labels.cpu().numpy())
    return out


def predict_volume(model, items, index, size, batch_size=8):
    """3D detections of a volume: every slice, then the slice boxes merged."""
    depth = items[index].shape[0]
    slices = [Unit(index, z, np.zeros((0, 4), np.float32), np.zeros(0, np.int64)) for z in range(depth)]
    detections = predict_units(model, items, slices, size, batch_size)
    return merge_slices({slices[k].z: d for k, d in detections.items()})


def predict_items(model, items, indices, size, batch_size=8):
    """Detections of whole images (2D) or volumes (3D): {item index: (boxes, scores, labels)}."""
    if items[indices[0]].dim == 3:
        return {i: predict_volume(model, items, i, size, batch_size) for i in indices}
    units = [Unit(i, None, items[i].boxes.astype(np.float32), items[i].labels) for i in indices]
    detections = predict_units(model, items, units, size, batch_size)
    return {units[k].item: d for k, d in detections.items()}


def ground_truth(items, indices):
    return {i: (items[i].boxes, items[i].labels) for i in indices}


def _unit_score(model, items, units, size, num_classes, batch_size):
    """mAP of the validation units (images or slices), for early stopping."""
    detections = predict_units(model, items, units, size, batch_size)
    truth = {k: (u.boxes, u.labels) for k, u in enumerate(units)}
    ap, _ = metrics.average_precision(truth, detections, num_classes, metrics.IOU_THRESHOLDS[2])
    return float(np.nanmean(ap)) if np.isfinite(ap).any() else 0.0


def train(spec, items, train_units, num_classes, params, val_units=None, epochs=None, log=print, pretrained=True):
    """Fine-tunes a network. With ``val_units``: early stopping on their mAP (the best epoch is
    kept). Without: ``epochs`` epochs. Returns the network and the best epoch (1-based)."""
    size = input_size(spec, params["image_size"])
    torch.manual_seed(params.get("seed", 0))
    model = build(spec, num_classes, size, pretrained=pretrained).to(device())
    augmentation = Augmentation(**params.get("augmentation", {}))
    loader = _loader(items, train_units, size, params["batch_size"], augmentation, shuffle=True, seed=params.get("seed", 0))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=spec.learning_rate * params.get("lr_scale", 1.0), weight_decay=1e-4)
    use_amp = device().type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    max_epochs = epochs or params["epochs"]
    best, best_epoch, best_state, waited = -1.0, max_epochs, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        started, losses = time.time(), []
        for images, targets, _ in loader:
            images = images.to(device())
            targets = [{k: v.to(device()) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device().type, enabled=use_amp):
                loss = model.loss(images, targets)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        message = f"{spec.name}: epoch {epoch}/{max_epochs}, loss {np.mean(losses) if losses else float('nan'):.3f}"
        if val_units:
            score = _unit_score(model, items, val_units, size, num_classes, params["batch_size"])
            message += f", validation mAP {score:.3f}"
            if score > best:
                best, best_epoch, waited = score, epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                waited += 1
        log(f"{message} ({time.time() - started:.0f} s)")
        if val_units and waited >= params["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def image_tensor(items, index, z, size):
    """(3, size, size) tensor in [0, 1] of an image or slice, and the image in 8 bits."""
    from PIL import Image
    image = load_unit_image(items[index], z)
    resized = np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255
    return torch.from_numpy(resized).permute(2, 0, 1), image
