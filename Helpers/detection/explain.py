"""D-RISE: which image regions a detection depends on (Petsiuk et al., CVPR 2021).

Model-agnostic: the image is shown to the network many times with random parts masked out;
each masked image is weighted by how well the network still finds the detection (IoU of its
best box of the same class x its score), and the weighted masks are averaged. Bright regions
are those the detection needs.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .metrics import iou_matrix


@torch.no_grad()
def drise(model, image, box, label, n_masks=300, grid=12, keep=0.5, batch_size=8, seed=0):
    """Saliency map (size x size, in [0, 1]) of the detection (``box`` x1 y1 x2 y2 in the pixels
    of the size x size ``image`` tensor, ``label``)."""
    device = next(model.parameters()).device
    size = image.shape[-1]
    cell = int(np.ceil(size / grid))
    generator = torch.Generator().manual_seed(seed)
    saliency = torch.zeros(size, size)
    total = 0.0
    model.eval()
    for start in range(0, n_masks, batch_size):
        count = min(batch_size, n_masks - start)
        coarse = (torch.rand(count, 1, grid, grid, generator=generator) < keep).float()
        upsampled = F.interpolate(coarse, size=(size + cell, size + cell), mode="bilinear", align_corners=False)
        shifts = torch.randint(0, cell, (count, 2), generator=generator)
        masks = torch.stack([upsampled[i, 0, int(dy):int(dy) + size, int(dx):int(dx) + size] for i, (dy, dx) in enumerate(shifts)])
        predictions = model.predict((image.unsqueeze(0) * masks.unsqueeze(1)).to(device))
        for mask, (boxes, scores, labels) in zip(masks, predictions):
            boxes, scores, labels = boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
            same = labels == label
            weight = float((iou_matrix(np.asarray([box]), boxes[same])[0] * scores[same]).max()) if same.any() else 0.0
            saliency += weight * mask
            total += weight
    saliency = saliency / max(n_masks, 1)
    low, high = float(saliency.min()), float(saliency.max())
    return ((saliency - low) / (high - low)).numpy() if high > low else np.zeros((size, size))
