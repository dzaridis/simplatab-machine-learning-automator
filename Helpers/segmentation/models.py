"""Segmentation networks of the automator.

2D (medical and natural images): the official nnU-Net 2D, U-Net / U-Net++ / DeepLabV3+ / FPN /
UPerNet / SegFormer / MA-Net with ImageNet-pretrained encoders (segmentation_models_pytorch), and
Attention U-Net / a nnU-Net-style U-Net trained from scratch (MONAI).
3D (volumes, one or more series): the official nnU-Net 3D full resolution, SwinUNETR with its CT
self-supervised encoder, SwinUNETR-V2, SegResNet, DynUNet (the nnU-Net architecture), UNETR,
MedNeXt-S, Attention U-Net, U-Net++ and V-Net (MONAI, trained from scratch except SwinUNETR).

Every network (except nnU-Net, which runs its own pipeline) maps a normalised patch
(B, C, [D,] H, W) to class logits (B, K + 1, [D,] H, W).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SegNetwork:
    key: str
    name: str
    dim: int            # 2 or 3
    family: str         # "nnunet", "pretrained", "ssl" or "scratch"
    description: str
    builder: str        # how to build it (see build)
    encoder: str = ""
    default: bool = False
    light: bool = False


NETWORKS = [
    # ---- 2D ------------------------------------------------------------------------------
    SegNetwork("nnunet_2d", "nnU-Net 2D", 2, "nnunet",
               "The official self-configuring nnU-Net v2: plans its own U-Net, preprocessing and training "
               "from your data. The reference method of medical segmentation challenges.", "nnunet", default=True),
    SegNetwork("unet_resnet34", "U-Net ResNet-34", 2, "pretrained",
               "The classic U-Net with an ImageNet-pretrained ResNet-34 encoder: a strong, fast baseline.",
               "smp:Unet", "resnet34", default=True, light=True),
    SegNetwork("unetpp_effb4", "U-Net++ EfficientNet-B4", 2, "pretrained",
               "Nested dense skip connections (U-Net++) on an EfficientNet-B4 encoder: finer boundaries.",
               "smp:UnetPlusPlus", "efficientnet-b4"),
    SegNetwork("deeplabv3p_resnet50", "DeepLabV3+ ResNet-50", 2, "pretrained",
               "Atrous spatial pyramid pooling for multi-scale context, with a ResNet-50 encoder.",
               "smp:DeepLabV3Plus", "resnet50"),
    SegNetwork("fpn_convnext_tiny", "FPN ConvNeXt-Tiny", 2, "pretrained",
               "Feature pyramid network on a ConvNeXt-Tiny encoder (2022): robust multi-scale features.",
               "smp:FPN", "tu-convnext_tiny"),
    SegNetwork("upernet_convnext_tiny", "UPerNet ConvNeXt-Tiny", 2, "pretrained",
               "UPerNet decoder (pyramid pooling + FPN) on ConvNeXt-Tiny, a standard of semantic segmentation benchmarks.",
               "smp:UPerNet", "tu-convnext_tiny"),
    SegNetwork("segformer_b2", "SegFormer-B2", 2, "pretrained",
               "Hierarchical transformer with a light MLP decoder (2021): strong on natural images.",
               "smp:Segformer", "mit_b2", default=True),
    SegNetwork("manet_resnet50", "MA-Net ResNet-50", 2, "pretrained",
               "Multi-scale attention network (position and channel attention), designed for medical images.",
               "smp:MAnet", "resnet50"),
    SegNetwork("attention_unet_2d", "Attention U-Net 2D", 2, "scratch",
               "U-Net with attention gates on the skip connections, trained from scratch.", "monai:attention"),
    SegNetwork("unet_2d", "U-Net 2D (nnU-Net-like)", 2, "scratch",
               "The nnU-Net architecture (MONAI DynUNet: residual blocks, instance normalisation, deep "
               "supervision off) trained with this automator's recipe, from scratch.", "monai:dynunet", light=True),
    # ---- 3D ------------------------------------------------------------------------------
    SegNetwork("nnunet_3d", "nnU-Net 3D full resolution", 3, "nnunet",
               "The official self-configuring nnU-Net v2 (3d_fullres): the reference method of 3D medical "
               "segmentation challenges. Needs a GPU.", "nnunet", default=True),
    SegNetwork("swinunetr", "SwinUNETR", 3, "ssl",
               "Swin transformer encoder self-supervised on 5,050 CT volumes, with a convolutional decoder.",
               "monai:swinunetr", default=True),
    SegNetwork("swinunetr_v2", "SwinUNETR-V2", 3, "scratch",
               "SwinUNETR with residual convolution blocks in the encoder (MICCAI 2023), from scratch.",
               "monai:swinunetr_v2"),
    SegNetwork("segresnet", "SegResNet", 3, "scratch",
               "Residual encoder-decoder of the BraTS 2018 winning method: memory-efficient and robust.",
               "monai:segresnet", default=True, light=True),
    SegNetwork("dynunet_3d", "DynUNet 3D (nnU-Net architecture)", 3, "scratch",
               "The nnU-Net U-Net (MONAI DynUNet) trained with this automator's recipe.", "monai:dynunet"),
    SegNetwork("unetr", "UNETR", 3, "scratch",
               "Vision transformer encoder (16x16x16 patches) with a convolutional decoder; needs large datasets.",
               "monai:unetr"),
    SegNetwork("mednext_s", "MedNeXt-S", 3, "scratch",
               "ConvNeXt blocks for 3D medical images (MICCAI 2023), encoder and decoder: state of the art among CNNs.",
               "mednext"),
    SegNetwork("attention_unet_3d", "Attention U-Net 3D", 3, "scratch",
               "3D U-Net with attention gates on the skip connections.", "monai:attention"),
    SegNetwork("unetpp_3d", "U-Net++ 3D", 3, "scratch",
               "Nested U-Net (BasicUNetPlusPlus) with dense skip connections.", "monai:unetpp"),
    SegNetwork("vnet", "V-Net", 3, "scratch",
               "Residual volumetric network designed for prostate MRI (2016).", "monai:vnet", light=True),
]
BY_KEY = {n.key: n for n in NETWORKS}
BY_NAME = {n.name: n for n in NETWORKS}


def networks_for(dim):
    return [n for n in NETWORKS if n.dim == dim]


# ---------------------------------------------------------------------------------------
# MedNeXt U-Net
# ---------------------------------------------------------------------------------------

class MedNeXtUNet(nn.Module):
    """MedNeXt-S encoder (Helpers.image3d.architectures) with a symmetric decoder: transposed
    convolutions, skip additions and MedNeXt blocks."""

    def __init__(self, in_channels, out_channels, width=32, blocks=2):
        super().__init__()
        from Helpers.image3d.architectures import MedNeXtBlock, MedNeXtEncoder
        self.encoder = MedNeXtEncoder(in_channels, width=width)
        widths = [width * 2 ** i for i in range(len(self.encoder.stages))]
        self.up = nn.ModuleList([nn.ConvTranspose3d(widths[i + 1], widths[i], 2, stride=2) for i in range(len(widths) - 1)])
        self.decode = nn.ModuleList([nn.Sequential(*[MedNeXtBlock(widths[i]) for _ in range(blocks)])
                                     for i in range(len(widths) - 1)])
        self.out = nn.Conv3d(width, out_channels, 1)

    def forward(self, x):
        skips = []
        x = self.encoder.stem(x)
        for stage in self.encoder.stages:
            x = stage(x)
            skips.append(x)
        for i in reversed(range(len(self.up))):
            x = self.decode[i](self.up[i](x) + skips[i])
        return self.out(x)


# ---------------------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------------------

class Normalised(nn.Module):
    """ImageNet normalisation of RGB inputs in [0, 1] before a pretrained 2D encoder (inputs that
    are not RGB are already normalised by the preprocessing)."""

    def __init__(self, net, rgb):
        super().__init__()
        self.net, self.rgb = net, rgb
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        if self.rgb:
            x = (x - self.mean) / self.std
        return self.net(x)


class ChannelAdapter(nn.Module):
    """A 1x1x1 convolution in front of V-Net, whose input layer needs 1, 2, 4, 8 or 16 channels."""

    def __init__(self, net, in_channels, channels):
        super().__init__()
        self.adapt = nn.Conv3d(in_channels, channels, 1)
        self.net = net

    def forward(self, x):
        return self.net(self.adapt(x))


def _dynunet_strides(patch):
    """Kernel sizes and strides of DynUNet (nnU-Net style): halve each axis while it stays >= 8,
    at most 5 times; axes too small are not down-sampled."""
    sizes, strides = list(patch), [[1] * len(patch)]
    for _ in range(5):
        stride = [2 if s >= 8 else 1 for s in sizes]
        if not any(s == 2 for s in stride):
            break
        strides.append(stride)
        sizes = [s // st for s, st in zip(sizes, stride)]
    kernels = [[3] * len(patch) for _ in strides]
    return kernels, strides


def build(spec, in_channels, num_classes, patch, pretrained=True, rgb=False):
    """The network for patches of size ``patch`` with ``in_channels`` channels and ``num_classes``
    output classes (background included)."""
    dims = len(patch)
    kind = spec.builder
    if kind.startswith("smp:"):
        import segmentation_models_pytorch as smp
        weights = "imagenet" if pretrained else None
        net = getattr(smp, kind[4:])(encoder_name=spec.encoder, encoder_weights=weights, in_channels=in_channels,
                                     classes=num_classes)
        return Normalised(net, rgb and in_channels == 3)
    from monai.networks import nets
    if kind == "monai:dynunet":
        kernels, strides = _dynunet_strides(patch)
        return nets.DynUNet(dims, in_channels, num_classes, kernels, strides, strides[1:], norm_name="instance",
                            res_block=True)
    if kind == "monai:attention":
        return nets.AttentionUnet(dims, in_channels, num_classes, channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2))
    if kind == "monai:segresnet":
        return nets.SegResNet(spatial_dims=dims, init_filters=32, in_channels=in_channels, out_channels=num_classes,
                              blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
    if kind in ("monai:swinunetr", "monai:swinunetr_v2"):
        v2 = kind.endswith("v2")
        net = nets.SwinUNETR(img_size=tuple(patch), in_channels=in_channels, out_channels=num_classes, feature_size=48,
                             spatial_dims=dims, use_v2=v2)
        if not v2 and pretrained:
            # the CT self-supervised encoder, its input layer adapted to the number of channels
            from Helpers.image3d.models import SwinViTEncoder
            net.swinViT.load_state_dict(SwinViTEncoder(in_channels, pretrained=True).net.state_dict(), strict=True)
        return net
    if kind == "monai:unetr":
        return nets.UNETR(in_channels, num_classes, img_size=tuple(patch), feature_size=16, hidden_size=768,
                          mlp_dim=3072, num_heads=12, proj_type="conv", norm_name="instance", res_block=True)
    if kind == "monai:unetpp":
        return nets.BasicUNetPlusPlus(spatial_dims=dims, in_channels=in_channels, out_channels=num_classes,
                                      features=(32, 32, 64, 128, 256, 32))
    if kind == "monai:vnet":
        net = nets.VNet(spatial_dims=dims, in_channels=in_channels if 16 % in_channels == 0 else 4, out_channels=num_classes)
        return net if 16 % in_channels == 0 else ChannelAdapter(net, in_channels, 4)
    if kind == "mednext":
        return MedNeXtUNet(in_channels, num_classes)
    raise ValueError(f"unknown network {spec.key}")


def divisor(spec, patch):
    """Each patch side must be a multiple of this for the network."""
    if spec.builder in ("monai:swinunetr", "monai:swinunetr_v2"):
        return 32
    if spec.builder == "monai:unetr":
        return 16
    if spec.builder.startswith("smp:"):
        return 32
    return 16


def fit_patch(spec, patch):
    """The SwinUNETR bottleneck is the patch divided by 32: a patch of 32 on every side leaves a
    single voxel, which instance normalisation refuses in training. Its first side is doubled."""
    patch = list(patch)
    if spec.builder in ("monai:swinunetr", "monai:swinunetr_v2") and max(patch) <= 32:
        patch[0] = 64
    return patch


def logits(output):
    """Some networks return deep-supervision lists in training mode: the full-resolution output."""
    if isinstance(output, (list, tuple)):
        return output[0]
    if output.dim() == 6:  # DynUNet deep supervision (B, heads, C, ...)
        return output[:, 0]
    return output


def describe_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def resize_logits(output, size):
    return F.interpolate(output, size=size, mode="bilinear" if output.dim() == 4 else "trilinear", align_corners=False)


def download_pretrained_weights():
    """Downloads the ImageNet weights of the pretrained 2D encoders into the local caches (used when
    building the Docker image, so that the automator works offline). The SwinUNETR encoder is
    downloaded with the 3D classification networks."""
    for spec in NETWORKS:
        if spec.family != "pretrained":
            continue
        try:
            build(spec, 3, 2, (64, 64), pretrained=True, rgb=True)
            print(f"{spec.name}: weights downloaded")
        except Exception as e:
            print(f"WARNING: could not download {spec.name}: {e}")


if __name__ == "__main__":
    download_pretrained_weights()
