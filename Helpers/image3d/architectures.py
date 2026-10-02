"""Modern 3D architectures, written from their papers and trained from scratch (no pretrained
weights): the encoders of state-of-the-art medical networks, used as classification backbones.

- MedNeXt (Roy et al., MICCAI 2023): ConvNeXt blocks designed for 3D medical images, with
  residual ConvNeXt down-sampling blocks.
- ConvNeXt V2 (Woo et al., CVPR 2023) in 3D: depthwise 7x7x7 convolutions, LayerNorm, an
  inverted MLP and global response normalisation (GRN).
- 3D UX-Net (Lee et al., ICLR 2023): large-kernel (7x7x7) depthwise convolutions with
  depthwise-convolution scaling instead of the MLP.
- nnU-Net ResEnc (Isensee et al., MICCAI 2024): the residual encoder of the nnU-Net "revisited"
  benchmark (instance normalisation, leaky ReLU, residual basic blocks).

Each module exposes ``stages`` (an nn.ModuleList, the last stage being applied last), ``stem``
and ``num_features``, so that the 3D models can split them for Grad-CAM.
"""
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm3d(nn.Module):
    """LayerNorm over the channels of a (B, C, D, H, W) tensor."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


# ---------------------------------------------------------------------------------------
# MedNeXt
# ---------------------------------------------------------------------------------------

class MedNeXtBlock(nn.Module):
    """Depthwise conv -> GroupNorm (one group per channel) -> 1x1x1 expansion -> GELU ->
    1x1x1 compression, with a residual connection (strided and widened when down-sampling)."""

    def __init__(self, channels, out_channels=None, expansion=2, kernel=3, down=False):
        super().__init__()
        out_channels = out_channels or channels
        stride = 2 if down else 1
        self.depthwise = nn.Conv3d(channels, channels, kernel, stride=stride, padding=kernel // 2, groups=channels)
        self.norm = nn.GroupNorm(channels, channels)
        self.expand = nn.Conv3d(channels, expansion * channels, 1)
        self.compress = nn.Conv3d(expansion * channels, out_channels, 1)
        self.shortcut = nn.Conv3d(channels, out_channels, 1, stride=2) if down else None

    def forward(self, x):
        y = self.compress(F.gelu(self.expand(self.norm(self.depthwise(x)))))
        return y + (self.shortcut(x) if self.shortcut is not None else x)


class MedNeXtEncoder(nn.Module):
    """MedNeXt-S encoder: 32 channels doubled at each of 4 down-sampling blocks, 2 blocks per
    stage, expansion ratio 2, 3x3x3 kernels."""

    def __init__(self, in_channels, width=32, blocks=(2, 2, 2, 2, 2), expansion=2, kernel=3):
        super().__init__()
        self.stem = nn.Conv3d(in_channels, width, 1)
        stages, channels = [], width
        for i, count in enumerate(blocks):
            layers = []
            if i:
                layers.append(MedNeXtBlock(channels, 2 * channels, expansion, kernel, down=True))
                channels *= 2
            layers += [MedNeXtBlock(channels, None, expansion, kernel) for _ in range(count)]
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.num_features = channels


# ---------------------------------------------------------------------------------------
# ConvNeXt V2 3D
# ---------------------------------------------------------------------------------------

class GRN(nn.Module):
    """Global response normalisation of ConvNeXt V2 (channels-last input)."""

    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, channels))

    def forward(self, x):
        g = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        n = g / (g.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * n) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, channels, kernel=7):
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, 4 * channels)
        self.grn = GRN(4 * channels)
        self.compress = nn.Linear(4 * channels, channels)

    def forward(self, x):
        y = self.depthwise(x).permute(0, 2, 3, 4, 1)
        y = self.compress(self.grn(F.gelu(self.expand(self.norm(y)))))
        return x + y.permute(0, 4, 1, 2, 3)


class ConvNeXtV2Encoder(nn.Module):
    """ConvNeXt V2-Pico in 3D (64-512 channels, 2-2-6-2 blocks). The stem patchifies 2 x 4 x 4
    voxels (slices are usually thicker than the in-plane pixels)."""

    def __init__(self, in_channels, dims=(64, 128, 256, 512), depths=(2, 2, 6, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_channels, dims[0], (2, 4, 4), stride=(2, 4, 4)), LayerNorm3d(dims[0]))
        stages = []
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            layers = [] if i == 0 else [LayerNorm3d(dims[i - 1]), nn.Conv3d(dims[i - 1], dim, 2, stride=2)]
            layers += [ConvNeXtV2Block(dim) for _ in range(depth)]
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.norm = LayerNorm3d(dims[-1])
        self.num_features = dims[-1]


# ---------------------------------------------------------------------------------------
# 3D UX-Net
# ---------------------------------------------------------------------------------------

class UXBlock(nn.Module):
    """Depthwise 7x7x7 convolution, LayerNorm, then depthwise-convolution scaling (grouped 1x1x1
    convolutions, 4x expansion) with a layer scale."""

    def __init__(self, channels, kernel=7):
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.norm = LayerNorm3d(channels)
        self.expand = nn.Conv3d(channels, 4 * channels, 1, groups=channels)
        self.compress = nn.Conv3d(4 * channels, channels, 1, groups=channels)
        self.scale = nn.Parameter(1e-6 * torch.ones(1, channels, 1, 1, 1))

    def forward(self, x):
        return x + self.scale * self.compress(F.gelu(self.expand(self.norm(self.depthwise(x)))))


class UXNetEncoder(nn.Module):
    """3D UX-Net encoder: 48-384 channels, 2 blocks per stage."""

    def __init__(self, in_channels, dims=(48, 96, 192, 384), depths=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_channels, dims[0], 7, stride=2, padding=3), LayerNorm3d(dims[0]))
        stages = []
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            layers = [] if i == 0 else [LayerNorm3d(dims[i - 1]), nn.Conv3d(dims[i - 1], dim, 2, stride=2)]
            layers += [UXBlock(dim) for _ in range(depth)]
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.norm = LayerNorm3d(dims[-1])
        self.num_features = dims[-1]


# ---------------------------------------------------------------------------------------
# nnU-Net residual encoder (ResEnc)
# ---------------------------------------------------------------------------------------

def _conv_norm(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                         nn.InstanceNorm3d(out_channels, affine=True))


class ResidualBlock(nn.Module):
    """Basic residual block of nnU-Net ResEnc ("BasicBlockD": the shortcut down-samples by
    average pooling, then a 1x1x1 convolution)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = _conv_norm(in_channels, out_channels, stride)
        self.conv2 = _conv_norm(out_channels, out_channels)
        shortcut = []
        if stride != 1:
            shortcut.append(nn.AvgPool3d(stride, stride))
        if in_channels != out_channels:
            shortcut += [nn.Conv3d(in_channels, out_channels, 1, bias=False), nn.InstanceNorm3d(out_channels, affine=True)]
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        return F.leaky_relu(self.conv2(F.leaky_relu(self.conv1(x), 0.01)) + self.shortcut(x), 0.01)


class ResEncEncoder(nn.Module):
    """nnU-Net ResEnc-M encoder: 32-320 channels, 1-3-4-6-6 residual blocks. The first two
    down-samplings keep the slices (stride 1 x 2 x 2), as nnU-Net plans anisotropic data."""

    def __init__(self, in_channels, features=(32, 64, 128, 256, 320), blocks=(1, 3, 4, 6, 6),
                 strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2))):
        super().__init__()
        self.stem = nn.Sequential(_conv_norm(in_channels, features[0]), nn.LeakyReLU(0.01))
        stages, channels = [], features[0]
        for out_channels, count, stride in zip(features, blocks, strides):
            layers = [ResidualBlock(channels, out_channels, stride)]
            layers += [ResidualBlock(out_channels, out_channels) for _ in range(count - 1)]
            stages.append(nn.Sequential(*layers))
            channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.num_features = channels
