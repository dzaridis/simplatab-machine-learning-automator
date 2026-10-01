"""The ten detection networks, pretrained on COCO and fine-tuned on the user's images, behind
one interface:
- ``loss(images, targets)``: training loss; images (B, 3, S, S) in [0, 1], targets with
  "boxes" (n, 4) x1 y1 x2 y2 in pixels of the S x S image and "labels" (n,) in 0..C-1;
- ``predict(images)``: [(boxes, scores, labels)] per image, at most 100 boxes, in the same space.
Images are resized to S x S (without keeping the proportions, like RT-DETR) before the networks.

torchvision (BSD): Faster R-CNN, RetinaNet, FCOS, Faster R-CNN MobileNetV3, SSDLite.
transformers (Apache 2.0): RT-DETR, RT-DETRv2, D-FINE, Deformable DETR, Conditional DETR.
Run this file to download the pretrained weights (done when building the Docker image).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

MAX_DETECTIONS = 100
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    name: str
    library: str                  # "torchvision" | "transformers"
    family: str                   # "cnn" | "transformer"
    description: str
    builder: str                  # torchvision function or transformers class
    checkpoint: Optional[str] = None
    weights: Optional[str] = None  # torchvision weights enum
    default: bool = False
    light: bool = False
    learning_rate: float = 1e-4
    normalize: bool = True         # ImageNet mean / std (transformers models without it take [0, 1])
    extra: Dict = field(default_factory=dict)


DETECTORS = [
    DetectorSpec("fasterrcnn_v2", "Faster R-CNN R50-FPN v2", "torchvision", "cnn",
                 "Two-stage detector (region proposals, then classification): the accurate reference, robust on small datasets.",
                 "fasterrcnn_resnet50_fpn_v2", weights="FasterRCNN_ResNet50_FPN_V2_Weights", default=True),
    DetectorSpec("retinanet_v2", "RetinaNet R50-FPN v2", "torchvision", "cnn",
                 "One-stage detector with focal loss: handles many easy background regions, e.g. small lesions.",
                 "retinanet_resnet50_fpn_v2", weights="RetinaNet_ResNet50_FPN_V2_Weights", default=True),
    DetectorSpec("fcos", "FCOS R50-FPN", "torchvision", "cnn",
                 "Anchor-free one-stage detector: predicts boxes from every location, no anchor tuning.",
                 "fcos_resnet50_fpn", weights="FCOS_ResNet50_FPN_Weights"),
    DetectorSpec("fasterrcnn_mobilenet", "Faster R-CNN MobileNetV3", "torchvision", "cnn",
                 "Light two-stage detector: several times faster, practical on CPU.",
                 "fasterrcnn_mobilenet_v3_large_fpn", weights="FasterRCNN_MobileNet_V3_Large_FPN_Weights", light=True),
    DetectorSpec("ssdlite", "SSDLite MobileNetV3", "torchvision", "cnn",
                 "The fastest detector (320 × 320 input): for quick experiments and CPU, less accurate on small objects.",
                 "ssdlite320_mobilenet_v3_large", weights="SSDLite320_MobileNet_V3_Large_Weights", light=True),
    DetectorSpec("rtdetr", "RT-DETR R50", "transformers", "transformer",
                 "Real-time detection transformer: end-to-end (no anchors or non-maximum suppression) at CNN speed (2024).",
                 "RTDetrForObjectDetection", checkpoint="PekingU/rtdetr_r50vd", normalize=False),
    DetectorSpec("rtdetr_v2", "RT-DETRv2 R50", "transformers", "transformer",
                 "Improved RT-DETR with better training recipe and sampling: state of the art among real-time detectors (2024).",
                 "RTDetrV2ForObjectDetection", checkpoint="PekingU/rtdetr_v2_r50vd", normalize=False, default=True),
    DetectorSpec("dfine", "D-FINE-M", "transformers", "transformer",
                 "DETR refining box edges as probability distributions: top COCO accuracy for its speed (2025).",
                 "DFineForObjectDetection", checkpoint="ustc-community/dfine-medium-coco", normalize=False, default=True),
    DetectorSpec("deformable_detr", "Deformable DETR", "transformers", "transformer",
                 "DETR with deformable attention on multi-scale features: good on small objects, slower to train.",
                 "DeformableDetrForObjectDetection", checkpoint="SenseTime/deformable-detr"),
    DetectorSpec("conditional_detr", "Conditional DETR R50", "transformers", "transformer",
                 "DETR with conditional spatial queries: converges several times faster than the original DETR.",
                 "ConditionalDetrForObjectDetection", checkpoint="microsoft/conditional-detr-resnet-50"),
]
BY_KEY = {d.key: d for d in DETECTORS}


# ---------------------------------------------------------------------------------------
# torchvision
# ---------------------------------------------------------------------------------------

def _torchvision_model(spec, num_classes, size, pretrained):
    import torchvision.models.detection as tvd
    builder = getattr(tvd, spec.builder)
    kwargs = {} if spec.key == "ssdlite" else {"min_size": size, "max_size": size}
    if not pretrained:
        return builder(weights=None, weights_backbone=None, num_classes=num_classes + 1, **kwargs)
    weights = getattr(tvd, spec.weights).DEFAULT
    model = builder(weights=weights, **kwargs)
    classes = num_classes + 1  # + background
    if spec.key.startswith("fasterrcnn"):
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, classes)
    elif spec.key == "retinanet_v2":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        head = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            head.conv[0][0].in_channels, head.num_anchors, classes, norm_layer=lambda c: nn.GroupNorm(32, c))
    elif spec.key == "fcos":
        from torchvision.models.detection.fcos import FCOSClassificationHead
        head = model.head.classification_head
        model.head.classification_head = FCOSClassificationHead(head.conv[0].in_channels, head.num_anchors, classes)
    elif spec.key == "ssdlite":
        from functools import partial
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        in_channels = [m[0][0].in_channels for m in model.head.classification_head.module_list]
        anchors = model.anchor_generator.num_anchors_per_location()
        norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head.classification_head = SSDLiteClassificationHead(in_channels, anchors, classes, norm)
    return model


class TorchvisionDetector(nn.Module):
    def __init__(self, spec, num_classes, size, pretrained=True):
        super().__init__()
        self.spec, self.num_classes, self.size = spec, num_classes, size
        self.model = _torchvision_model(spec, num_classes, size, pretrained)
        # Keep low-score boxes: the evaluation (mAP, FROC) needs the whole score range
        heads = self.model.roi_heads if hasattr(self.model, "roi_heads") else self.model
        heads.detections_per_img, heads.score_thresh = MAX_DETECTIONS, 0.001

    def loss(self, images, targets):
        targets = [{"boxes": t["boxes"], "labels": t["labels"] + 1} for t in targets]
        losses = self.model(list(images), targets)
        return sum(losses.values())

    @torch.no_grad()
    def predict(self, images):
        outputs = self.model(list(images))
        # Label 0 is the background (RetinaNet and FCOS score it like the other classes)
        return [(o["boxes"][o["labels"] > 0], o["scores"][o["labels"] > 0], o["labels"][o["labels"] > 0] - 1) for o in outputs]


# ---------------------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------------------

def _transformers_model(spec, num_classes, pretrained):
    import transformers
    cls = getattr(transformers, spec.builder)
    labels = {"num_labels": num_classes, "id2label": {i: str(i) for i in range(num_classes)},
              "label2id": {str(i): i for i in range(num_classes)}}
    if pretrained:
        kwargs = dict(labels, ignore_mismatched_sizes=True)
        if spec.key in ("deformable_detr", "conditional_detr"):
            kwargs["use_pretrained_backbone"] = False  # the checkpoint holds the backbone weights
        return cls.from_pretrained(spec.checkpoint, **kwargs)
    config_cls = getattr(transformers, spec.builder.replace("ForObjectDetection", "Config"))
    extra = {"use_pretrained_backbone": False} if spec.key in ("deformable_detr", "conditional_detr") else {}
    return cls(config_cls(**labels, **extra))


class TransformersDetector(nn.Module):
    """DETR-like networks with sigmoid class scores; boxes as (cx, cy, w, h) in [0, 1]."""

    def __init__(self, spec, num_classes, size, pretrained=True):
        super().__init__()
        self.spec, self.num_classes, self.size = spec, num_classes, size
        self.model = _transformers_model(spec, num_classes, pretrained)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _pixels(self, images):
        return (images - self.mean) / self.std if self.spec.normalize else images

    def loss(self, images, targets):
        labels = []
        for t in targets:
            b = t["boxes"] / self.size
            cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], dim=1)
            labels.append({"class_labels": t["labels"], "boxes": cxcywh.clamp(0, 1)})
        return self.model(pixel_values=self._pixels(images), labels=labels).loss

    @torch.no_grad()
    def predict(self, images):
        outputs = self.model(pixel_values=self._pixels(images))
        probabilities = outputs.logits.sigmoid()  # (B, queries, classes)
        results = []
        for probs, boxes in zip(probabilities, outputs.pred_boxes):
            scores, index = probs.flatten().topk(min(MAX_DETECTIONS, probs.numel()))
            query, label = index // probs.shape[1], index % probs.shape[1]
            cx, cy, w, h = boxes[query].unbind(-1)
            xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1).clamp(0, 1) * self.size
            results.append((xyxy, scores, label))
        return results


def build(spec, num_classes, size, pretrained=True):
    cls = TorchvisionDetector if spec.library == "torchvision" else TransformersDetector
    return cls(spec, num_classes, size, pretrained)


def input_size(spec, size):
    """SSDLite works at 320 x 320 whatever the chosen size."""
    return 320 if spec.key == "ssdlite" else size


def download_weights():
    """Downloads the pretrained weights of the ten networks (Docker image build)."""
    for spec in DETECTORS:
        try:
            build(spec, 2, 640, pretrained=True)
            print(f"{spec.name}: weights downloaded")
        except Exception as e:
            print(f"WARNING: {spec.name}: pretrained weights not downloaded ({type(e).__name__}: {e})")


if __name__ == "__main__":
    download_weights()
