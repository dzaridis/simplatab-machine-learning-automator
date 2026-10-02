"""The 3D networks of the image classification automator.

Every network is an encoder that turns a (B, C, D, H, W) volume in [0, 1] into a feature map
(B, K, d, h, w) and a pooled feature vector, followed by a linear layer (as for 2D images: a
logistic regression on frozen features in "feature extraction" mode, or trained with the
encoder in "fine-tuning" mode). The input layer of a pretrained network is adapted to the
number of channels (series) of the studies: its weights are averaged over the channels it was
pretrained with and repeated, so that a volume repeated on every channel gives the same
activations as before.
"""
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

KINETICS_MEAN, KINETICS_STD = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SSL_URL = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
SSL_FILE = "swinvit_ssl_encoder.pt"
MAX_SLICES_25D = 16  # slices given to the 2D network of the 2.5D model


@dataclass(frozen=True)
class Network3D:
    key: str
    name: str
    family: str        # "medical", "video", "ssl", "2.5d", "scratch" (CNN) or "scratch_transformer"
    description: str
    weights: str       # what the network was pretrained on
    default: bool = False
    light: bool = False


NETWORKS = [
    Network3D("medicalnet_resnet10", "MedicalNet ResNet-10", "medical",
              "3D ResNet pretrained on 23 CT and MRI segmentation datasets (Med3D): small and fast, a strong first choice.",
              "MedicalNet (23 medical datasets)", default=True, light=True),
    Network3D("medicalnet_resnet18", "MedicalNet ResNet-18", "medical",
              "Deeper MedicalNet ResNet: more capacity for larger datasets.", "MedicalNet (23 medical datasets)",
              default=True),
    Network3D("medicalnet_resnet50", "MedicalNet ResNet-50", "medical",
              "Bottleneck MedicalNet ResNet (2048 features): the largest medical network, GPU recommended.",
              "MedicalNet (23 medical datasets)"),
    Network3D("r3d_18", "R3D-18", "video",
              "3D ResNet-18 pretrained on Kinetics-400 videos (slices play the role of frames).", "Kinetics-400", light=True),
    Network3D("r2plus1d_18", "R(2+1)D-18", "video",
              "Factorised 3D convolutions (2D in-plane + 1D across slices): suited to thick-slice MRI.", "Kinetics-400",
              default=True),
    Network3D("mc3_18", "MC3-18", "video",
              "Mixed convolutions: 3D in the first layers, 2D in-plane after; robust to few slices.", "Kinetics-400",
              light=True),
    Network3D("swin3d_t", "Video Swin-T", "video",
              "Video Swin transformer: shifted 3D windows of attention, the strongest video network (GPU recommended).",
              "Kinetics-400"),
    Network3D("swinvit_ssl", "SwinUNETR Swin-ViT", "ssl",
              "Encoder of SwinUNETR, self-supervised on 5,050 CT volumes: a medical 3D transformer.",
              "Self-supervised, 5,050 CT volumes"),
    Network3D("densenet121_3d", "DenseNet-121 3D", "scratch",
              "Densely connected 3D CNN (MONAI), the classic baseline of 3D medical classification.",
              "None (trained from scratch)"),
    Network3D("mednext_s", "MedNeXt-S", "scratch",
              "ConvNeXt blocks redesigned for 3D medical images (MICCAI 2023), with residual down-sampling: "
              "state of the art among medical CNNs.", "None (trained from scratch)"),
    Network3D("convnextv2_3d", "ConvNeXt V2-Pico 3D", "scratch",
              "ConvNeXt V2 (2023) in 3D: 7x7x7 depthwise convolutions and global response normalisation.",
              "None (trained from scratch)"),
    Network3D("uxnet_3d", "3D UX-Net", "scratch",
              "Large-kernel 3D CNN (ICLR 2023): 7x7x7 depthwise convolutions with the receptive field of a transformer.",
              "None (trained from scratch)"),
    Network3D("resenc_m", "nnU-Net ResEnc-M encoder", "scratch",
              "Residual encoder of nnU-Net, the strongest 3D medical baseline of the 2024 nnU-Net revisited benchmark.",
              "None (trained from scratch)"),
    Network3D("seresnext50_3d", "SEResNeXt-50 3D", "scratch",
              "Grouped residual convolutions with squeeze-and-excitation channel attention (MONAI).",
              "None (trained from scratch)"),
    Network3D("efficientnet_b0_3d", "EfficientNet-B0 3D", "scratch",
              "Compound-scaled mobile convolutions with squeeze-and-excitation (MONAI): the lightest network.",
              "None (trained from scratch)", light=True),
    Network3D("swinunetr_v2", "SwinUNETR-V2 encoder", "scratch_transformer",
              "Swin transformer with residual convolution blocks before each stage (MICCAI 2023).",
              "None (trained from scratch)"),
    Network3D("vit_3d", "ViT-Small 3D (UNETR)", "scratch_transformer",
              "Plain vision transformer on 8x16x16 patches, the encoder of UNETR: global attention from the first layer, "
              "needs large datasets.", "None (trained from scratch)"),
    Network3D("dinov2_25d", "DINOv2-Small 2.5D", "2.5d",
              f"2D foundation model applied to up to {MAX_SLICES_25D} slices, combined by attention pooling: "
              "strong features without 3D pretraining.", "LVD-142M images (2D)"),
]
BY_KEY = {n.key: n for n in NETWORKS}
BY_NAME = {n.name: n for n in NETWORKS}


def adapt_input(conv, channels):
    """A copy of the first convolution taking ``channels`` input channels (weights averaged over
    the pretrained input channels, repeated and rescaled)."""
    pretrained = conv.in_channels
    if pretrained == channels:
        return conv
    weight = conv.weight.detach().mean(dim=1, keepdim=True) * (pretrained / channels)
    new = type(conv)(channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                     conv.groups, conv.bias is not None, conv.padding_mode)
    with torch.no_grad():
        new.weight.copy_(weight.repeat(1, channels, *([1] * (weight.dim() - 2))))
        if conv.bias is not None:
            new.bias.copy_(conv.bias.detach())
    return new


def _zscore(x):
    """Per volume and channel: zero mean, unit variance (MedicalNet and from-scratch networks)."""
    dims = tuple(range(2, x.dim()))
    return (x - x.mean(dim=dims, keepdim=True)) / (x.std(dim=dims, keepdim=True) + 1e-5)


def _standardize(x, mean, std):
    """RGB statistics when the input has 3 channels, else their average on every channel."""
    shape = (1, -1) + (1,) * (x.dim() - 2)
    if x.shape[1] == len(mean):
        return (x - torch.tensor(mean, device=x.device).view(shape)) / torch.tensor(std, device=x.device).view(shape)
    return (x - sum(mean) / len(mean)) / (sum(std) / len(std))


class Encoder(nn.Module):
    """``early`` maps the normalised volume to the feature map of the penultimate stage
    (B, K, d, h, w), where Grad-CAM is computed (twice the resolution of the last stage); ``late``
    applies the last stage, and ``pool`` turns its output into the feature vector."""
    num_features = 0

    def normalize(self, x):
        return x

    def early(self, x):
        raise NotImplementedError

    def late(self, x):
        return x

    def pool(self, feature_map):
        return feature_map.mean(dim=(2, 3, 4))


class _ShortcutA(nn.Module):
    """Parameter-free shortcut of MedicalNet ResNet-18/34 (strided subsampling + zero channels).
    Replaces MONAI's version, which concatenates ``tensor.data`` and so detaches the shortcut
    from the graph (no gradient through it, and frozen by TorchScript tracing)."""

    def __init__(self, planes, stride):
        super().__init__()
        self.planes, self.stride = planes, stride

    def forward(self, x):
        out = F.avg_pool3d(x, kernel_size=1, stride=self.stride)
        return F.pad(out, (0, 0, 0, 0, 0, 0, 0, self.planes - out.shape[1]))


class MedicalNetEncoder(Encoder):
    def __init__(self, depth, channels, pretrained):
        super().__init__()
        import functools
        from monai.networks.nets import resnet
        from monai.networks.nets.resnet import get_medicalnet_pretrained_resnet_args
        bias_downsample, shortcut = get_medicalnet_pretrained_resnet_args(depth)
        self.net = getattr(resnet, f"resnet{depth}")(
            pretrained=pretrained, spatial_dims=3, n_input_channels=1, feed_forward=False,
            shortcut_type=shortcut, bias_downsample=bias_downsample == 1)
        for layer in (self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4):
            for block in layer:
                if isinstance(block.downsample, functools.partial):
                    block.downsample = _ShortcutA(block.downsample.keywords["planes"], block.downsample.keywords["stride"])
        self.net.conv1 = adapt_input(self.net.conv1, channels)
        self.net.fc = None
        self.num_features = 2048 if depth >= 50 else 512

    def normalize(self, x):
        return _zscore(x)

    def early(self, x):
        net = self.net
        x = net.act(net.bn1(net.conv1(x)))
        if not net.no_max_pool:
            x = net.maxpool(x)
        return net.layer3(net.layer2(net.layer1(x)))

    def late(self, x):
        return self.net.layer4(x)


class VideoResNetEncoder(Encoder):
    def __init__(self, name, channels, pretrained):
        super().__init__()
        from torchvision.models import video
        weights = {"r3d_18": "R3D_18_Weights", "r2plus1d_18": "R2Plus1D_18_Weights", "mc3_18": "MC3_18_Weights"}[name]
        self.net = getattr(video, name)(weights=getattr(video, weights).KINETICS400_V1 if pretrained else None)
        self.net.stem[0] = adapt_input(self.net.stem[0], channels)
        self.net.fc = nn.Identity()
        self.num_features = 512

    def normalize(self, x):
        return _standardize(x, KINETICS_MEAN, KINETICS_STD)

    def early(self, x):
        net = self.net
        return net.layer3(net.layer2(net.layer1(net.stem(x))))

    def late(self, x):
        return self.net.layer4(x)


class VideoSwinEncoder(Encoder):
    def __init__(self, channels, pretrained):
        super().__init__()
        from torchvision.models import video
        self.net = video.swin3d_t(weights=video.Swin3D_T_Weights.KINETICS400_V1 if pretrained else None)
        self.net.patch_embed.proj = adapt_input(self.net.patch_embed.proj, channels)
        self.net.head = nn.Identity()
        self.num_features = self.net.num_features

    def normalize(self, x):
        return _standardize(x, IMAGENET_MEAN, IMAGENET_STD)

    def early(self, x):
        net = self.net
        x = net.features[:-2](net.pos_drop(net.patch_embed(x)))  # (B, T, H, W, C), up to the third stage
        return x.permute(0, 4, 1, 2, 3)

    def late(self, x):
        net = self.net
        x = net.norm(net.features[-2:](x.permute(0, 2, 3, 4, 1)))
        return x.permute(0, 4, 1, 2, 3)


def ssl_weights_path():
    return os.path.join(torch.hub.get_dir(), "checkpoints", SSL_FILE)


def download_ssl_weights():
    """The self-supervised Swin-ViT encoder of SwinUNETR (MONAI, Apache 2.0): the full checkpoint
    (400 MB, with the self-supervision heads) is downloaded once and only the encoder is kept."""
    path = ssl_weights_path()
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    full = path + ".download"
    torch.hub.download_url_to_file(SSL_URL, full, progress=False)
    state = torch.load(full, map_location="cpu", weights_only=False)["state_dict"]
    encoder = {k[len("module."):].replace(".mlp.fc1.", ".mlp.linear1.").replace(".mlp.fc2.", ".mlp.linear2."): v
               for k, v in state.items() if k.startswith("module.")}
    torch.save({k: encoder[k] for k in _swin_vit(1).state_dict()}, path)
    os.remove(full)
    return path


def _swin_vit(channels, use_v2=False):
    from monai.networks.nets.swin_unetr import SwinTransformer
    return SwinTransformer(in_chans=channels, embed_dim=48, window_size=(7, 7, 7), patch_size=(2, 2, 2),
                           depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), spatial_dims=3, use_v2=use_v2)


class SwinViTEncoder(Encoder):
    """Swin-ViT encoder of SwinUNETR: self-supervised weights, or SwinUNETR-V2 (a residual
    convolution block before each stage) trained from scratch."""

    def __init__(self, channels, pretrained, use_v2=False):
        super().__init__()
        self.use_v2 = use_v2
        self.net = _swin_vit(1, use_v2)
        if pretrained and not use_v2:
            self.net.load_state_dict(torch.load(download_ssl_weights(), map_location="cpu"), strict=True)
        self.net.patch_embed.proj = adapt_input(self.net.patch_embed.proj, channels)
        self.num_features = 768

    def normalize(self, x):
        return _zscore(x) if self.use_v2 else x

    def _stage(self, x, i):
        net = self.net
        if self.use_v2:
            x = getattr(net, f"layers{i}c")[0](x.contiguous())
        return getattr(net, f"layers{i}")[0](x.contiguous())

    def early(self, x):
        net = self.net
        x = net.pos_drop(net.patch_embed(x))
        return self._stage(self._stage(self._stage(x, 1), 2), 3)

    def late(self, x):
        return self.net.proj_out(self._stage(x, 4), normalize=True)


class DenseNetEncoder(Encoder):
    def __init__(self, channels):
        super().__init__()
        from monai.networks.nets import DenseNet121
        self.net = DenseNet121(spatial_dims=3, in_channels=channels, out_channels=1)
        self.num_features = self.net.class_layers.out.in_features
        self.net.class_layers = None

    def normalize(self, x):
        return _zscore(x)

    def early(self, x):
        features = self.net.features
        return features[:features_index(features, "transition3")](x)  # up to the third dense block

    def late(self, x):
        features = self.net.features
        return F.relu(features[features_index(features, "transition3"):](x))


class StagedEncoder(Encoder):
    """Encoders of Helpers.image3d.architectures (stem + stages, optional final norm), trained from
    scratch: Grad-CAM on the output of the penultimate stage."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.num_features = net.num_features

    def normalize(self, x):
        return _zscore(x)

    def early(self, x):
        x = self.net.stem(x)
        for stage in self.net.stages[:-1]:
            x = stage(x)
        return x

    def late(self, x):
        x = self.net.stages[-1](x)
        return self.net.norm(x) if hasattr(self.net, "norm") else x


class SEResNeXtEncoder(Encoder):
    def __init__(self, channels):
        super().__init__()
        from monai.networks.nets import SEResNext50
        self.net = SEResNext50(spatial_dims=3, in_channels=channels, num_classes=1)
        self.num_features = self.net.last_linear.in_features
        self.net.last_linear = None

    def normalize(self, x):
        return _zscore(x)

    def early(self, x):
        net = self.net
        return net.layer3(net.layer2(net.layer1(net.layer0(x))))

    def late(self, x):
        return self.net.layer4(x)


class EfficientNetEncoder(Encoder):
    def __init__(self, channels):
        super().__init__()
        from monai.networks.nets import EfficientNetBN
        self.net = EfficientNetBN("efficientnet-b0", pretrained=False, spatial_dims=3, in_channels=channels, num_classes=1)
        self.num_features = self.net._fc.in_features
        self.net._fc = None
        # Grad-CAM before the last down-sampling block
        strided = [i for i, stage in enumerate(self.net._blocks) if max(stage[0]._depthwise_conv.stride) > 1]
        self.split = strided[-1]  # stages (each an nn.Sequential of blocks)

    def normalize(self, x):
        return _zscore(x)

    def early(self, x):
        net = self.net
        x = net._swish(net._bn0(net._conv_stem(net._conv_stem_padding(x))))
        return net._blocks[:self.split](x)

    def late(self, x):
        net = self.net
        x = net._blocks[self.split:](x)
        return net._swish(net._bn1(net._conv_head(net._conv_head_padding(x))))


class ViT3DEncoder(Encoder):
    """ViT-Small on 8 x 16 x 16 patches (the encoder of UNETR, MONAI); the patch tokens form the
    feature map."""
    patch = (8, 16, 16)

    def __init__(self, channels, shape):
        super().__init__()
        from monai.networks.nets import ViT
        self.grid = tuple(s // p for s, p in zip(shape, self.patch))
        self.net = ViT(in_channels=channels, img_size=tuple(shape), patch_size=self.patch, hidden_size=384,
                       mlp_dim=1536, num_layers=12, num_heads=6, classification=False)
        self.num_features = 384

    def normalize(self, x):
        return _zscore(x)

    def _to_map(self, tokens):
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], *self.grid)

    def early(self, x):
        x = self.net.patch_embedding(x)
        for block in self.net.blocks[:-1]:
            x = block(x)
        return self._to_map(x)

    def late(self, x):
        tokens = x.flatten(2).transpose(1, 2)
        return self._to_map(self.net.norm(self.net.blocks[-1](tokens)))


def features_index(sequential, name):
    return [n for n, _ in sequential.named_children()].index(name)


class SliceAttentionEncoder(Encoder):
    """2.5D: DINOv2 on evenly spaced slices; the slice embeddings (mean of the patch tokens) are
    combined by gated attention pooling. The attention starts uniform (a mean over slices), so
    the frozen features of the feature-extraction mode are the mean slice embedding."""
    size = 224

    def __init__(self, channels, pretrained, depth):
        super().__init__()
        import timm
        self.vit = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0,
                                     img_size=self.size)
        self.vit.patch_embed.proj = adapt_input(self.vit.patch_embed.proj, channels)
        self.num_features = self.vit.num_features
        count = min(depth, MAX_SLICES_25D)
        self.register_buffer("slices", torch.linspace(0, depth - 1, count).round().long(), persistent=False)
        self.grid = self.vit.patch_embed.grid_size
        self.attention_v = nn.Linear(self.num_features, 128)
        self.attention_u = nn.Linear(self.num_features, 128)
        self.attention_w = nn.Linear(128, 1)
        nn.init.zeros_(self.attention_w.weight)
        nn.init.zeros_(self.attention_w.bias)

    def early(self, x):
        b, c = x.shape[:2]
        x = x.index_select(2, self.slices)                                   # (B, C, S, H, W)
        s = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * s, c, *x.shape[3:])
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        tokens = self.vit.forward_features(_standardize(x, IMAGENET_MEAN, IMAGENET_STD))
        tokens = tokens[:, self.vit.num_prefix_tokens:, :]                   # (B*S, h*w, K)
        h, w = self.grid
        return tokens.reshape(b, s, h, w, -1).permute(0, 4, 1, 2, 3)          # (B, K, S, h, w)

    def pool(self, feature_map):
        slices = feature_map.mean(dim=(3, 4)).transpose(1, 2)                # (B, S, K)
        gate = torch.tanh(self.attention_v(slices)) * torch.sigmoid(self.attention_u(slices))
        weights = torch.softmax(self.attention_w(gate), dim=1)               # (B, S, 1)
        return (weights * slices).sum(dim=1)


def create_encoder(spec, channels, pretrained, shape):
    if spec.family == "medical":
        return MedicalNetEncoder(int(spec.key.rsplit("resnet", 1)[1]), channels, pretrained)
    if spec.key in ("r3d_18", "r2plus1d_18", "mc3_18"):
        return VideoResNetEncoder(spec.key, channels, pretrained)
    if spec.key == "swin3d_t":
        return VideoSwinEncoder(channels, pretrained)
    if spec.key == "swinvit_ssl":
        return SwinViTEncoder(channels, pretrained)
    if spec.key == "densenet121_3d":
        return DenseNetEncoder(channels)
    if spec.key == "dinov2_25d":
        return SliceAttentionEncoder(channels, pretrained, shape[0])
    from . import architectures
    staged = {"mednext_s": architectures.MedNeXtEncoder, "convnextv2_3d": architectures.ConvNeXtV2Encoder,
              "uxnet_3d": architectures.UXNetEncoder, "resenc_m": architectures.ResEncEncoder}
    if spec.key in staged:
        return StagedEncoder(staged[spec.key](channels))
    if spec.key == "seresnext50_3d":
        return SEResNeXtEncoder(channels)
    if spec.key == "efficientnet_b0_3d":
        return EfficientNetEncoder(channels)
    if spec.key == "swinunetr_v2":
        return SwinViTEncoder(channels, pretrained=False, use_v2=True)
    if spec.key == "vit_3d":
        return ViT3DEncoder(channels, shape)
    raise ValueError(f"unknown network {spec.key}")


class VolumeClassifier(nn.Module):
    """Encoder + linear layer. Input (B, C, D, H, W) in [0, 1]; output class logits."""

    def __init__(self, spec, channels, shape, num_classes, pretrained=True, seed=0):
        super().__init__()
        self.spec = spec
        # Deterministic initialisation: the feature-extraction mode creates the network twice
        # (features, then the exported model), which must be identical even without pretrained weights
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = create_encoder(spec, channels, pretrained, shape)
            self.head = nn.Linear(self.encoder.num_features, num_classes)

    def early(self, x):
        """Feature map of the penultimate stage (where Grad-CAM is computed)."""
        return self.encoder.early(self.encoder.normalize(x))

    def from_early(self, early):
        """Class logits from the output of ``early``."""
        return self.head(self.encoder.pool(self.encoder.late(early)))

    def feature_map(self, x):
        return self.encoder.late(self.early(x))

    def features(self, x):
        return self.encoder.pool(self.feature_map(x))

    def forward(self, x):
        return self.head(self.features(x))

    def set_linear_head(self, weight, bias):
        with torch.no_grad():
            self.head.weight.copy_(torch.as_tensor(weight, dtype=self.head.weight.dtype))
            self.head.bias.copy_(torch.as_tensor(bias, dtype=self.head.bias.dtype))


def download_pretrained_weights():
    """Downloads the pretrained weights of every network into the local caches (used when
    building the Docker image, so that the automator works offline)."""
    for spec in NETWORKS:
        if spec.family.startswith("scratch"):
            continue
        try:
            create_encoder(spec, 1, True, SHAPE_FOR_DOWNLOAD)
            print(f"{spec.name}: weights downloaded")
        except Exception as e:
            print(f"WARNING: could not download {spec.name}: {e}")


SHAPE_FOR_DOWNLOAD = (32, 128, 128)

if __name__ == "__main__":
    download_pretrained_weights()
