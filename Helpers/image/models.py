"""Pretrained networks of the image classification automator (from timm).

Each network is used as a backbone that turns an image into a feature vector, followed by a
linear layer that outputs the class scores:
- in "feature extraction" mode the backbone is frozen and the linear layer is a logistic
  regression fitted on its features;
- in "fine-tuning" mode the backbone and the linear layer are trained together.
The same ``ImageClassifier`` module is exported in both modes, and explained with Grad-CAM.
"""
from dataclasses import dataclass, field

import torch
from torch import nn

INPUT_SIZE = 224


@dataclass(frozen=True)
class Backbone:
    key: str          # form field and file-name safe identifier
    name: str         # display name (also used in the result files)
    timm_name: str    # timm model and pretrained tag
    family: str       # "cnn" or "transformer"
    layout: str       # layout of forward_features: "nchw", "nhwc" or "tokens"
    description: str
    default: bool = False
    kwargs: dict = field(default_factory=dict)


BACKBONES = [
    Backbone("resnet50", "ResNet-50", "resnet50.a1_in1k", "cnn", "nchw",
             "The reference CNN of medical imaging studies, with the improved 2021 training recipe (ImageNet-1k).",
             default=True),
    Backbone("efficientnet_b0", "EfficientNet-B0", "efficientnet_b0.ra_in1k", "cnn", "nchw",
             "Compact CNN (5M parameters): the fastest network, a good first choice on CPU."),
    Backbone("efficientnetv2_s", "EfficientNetV2-S", "tf_efficientnetv2_s.in21k_ft_in1k", "cnn", "nchw",
             "Improved EfficientNet (2021) pretrained on ImageNet-21k: accurate and fast to fine-tune."),
    Backbone("convnext_tiny", "ConvNeXt-Tiny", "convnext_tiny.fb_in22k_ft_in1k", "cnn", "nchw",
             "Modern CNN with transformer design choices (2022), pretrained on ImageNet-22k."),
    Backbone("convnextv2_tiny", "ConvNeXt V2-Tiny", "convnextv2_tiny.fcmae_ft_in22k_in1k", "cnn", "nchw",
             "ConvNeXt with masked-autoencoder self-supervised pretraining (2023): strong transfer to new domains.",
             default=True),
    Backbone("vit_small", "ViT-Small", "vit_small_patch16_224.augreg_in21k_ft_in1k", "transformer", "tokens",
             "Vision transformer pretrained on ImageNet-21k: attends to the whole image at once."),
    Backbone("deit3_small", "DeiT III-Small", "deit3_small_patch16_224.fb_in22k_ft_in1k", "transformer", "tokens",
             "Vision transformer with the DeiT III training recipe (2022), ImageNet-22k."),
    Backbone("swin_tiny", "Swin-Tiny", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", "transformer", "nhwc",
             "Hierarchical transformer with shifted windows: multi-scale features, widely used in medical imaging."),
    Backbone("maxvit_tiny", "MaxViT-Tiny", "maxvit_tiny_tf_224.in1k", "transformer", "nchw",
             "Hybrid of convolutions and local + global attention (2022): state-of-the-art accuracy at its size."),
    Backbone("dinov2_small", "DINOv2-Small", "vit_small_patch14_dinov2.lvd142m", "transformer", "tokens",
             "Self-supervised foundation model (2023, 142M curated images): the best features without fine-tuning.",
             default=True, kwargs={"img_size": INPUT_SIZE}),
]

BY_KEY = {b.key: b for b in BACKBONES}
BY_NAME = {b.name: b for b in BACKBONES}


def create_backbone(backbone, pretrained=True):
    """The timm network without its classification layer (outputs pooled features)."""
    import timm
    return timm.create_model(backbone.timm_name, pretrained=pretrained, num_classes=0, **backbone.kwargs)


def normalization(model):
    """Mean and standard deviation the network was pretrained with."""
    cfg = model.pretrained_cfg
    return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))


class ImageClassifier(nn.Module):
    """Backbone + linear layer. Outputs class logits (2 columns for binary problems)."""

    def __init__(self, backbone, num_classes, pretrained=True, seed=0):
        super().__init__()
        self.spec = backbone
        # Deterministic initialisation: the feature-extraction mode creates the network twice
        # (features, then the exported model), which must be identical even without pretrained weights
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.backbone = create_backbone(backbone, pretrained)
            self.head = nn.Linear(self.backbone.num_features, num_classes)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x))

    def set_linear_head(self, weight, bias):
        with torch.no_grad():
            self.head.weight.copy_(torch.as_tensor(weight, dtype=self.head.weight.dtype))
            self.head.bias.copy_(torch.as_tensor(bias, dtype=self.head.bias.dtype))


def download_pretrained_weights():
    """Downloads the pretrained weights of every network into the local cache (used when
    building the Docker image, so that the automator works offline)."""
    for backbone in BACKBONES:
        try:
            create_backbone(backbone, pretrained=True)
            print(f"Downloaded {backbone.name}")
        except Exception as e:
            print(f"WARNING: could not download {backbone.name}: {e}")


if __name__ == "__main__":
    download_pretrained_weights()
