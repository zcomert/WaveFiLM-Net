"""The eleven torchvision backbones of the study, as feature extractors returning the last feature map."""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# name: (torchvision constructor, ImageNet weights, output channels)
BACKBONES = {
    "vgg-16": ("vgg16", "VGG16_Weights.IMAGENET1K_V1", 512),
    "vgg-19": ("vgg19", "VGG19_Weights.IMAGENET1K_V1", 512),
    "resnet-101": ("resnet101", "ResNet101_Weights.IMAGENET1K_V2", 2048),
    "densenet-169": ("densenet169", "DenseNet169_Weights.IMAGENET1K_V1", 1664),
    "densenet-201": ("densenet201", "DenseNet201_Weights.IMAGENET1K_V1", 1920),
    "mobilenet": ("mobilenet_v2", "MobileNet_V2_Weights.IMAGENET1K_V2", 1280),
    "inception-v3": ("inception_v3", "Inception_V3_Weights.IMAGENET1K_V1", 2048),
    "efficient-b7": ("efficientnet_b7", "EfficientNet_B7_Weights.IMAGENET1K_V1", 2560),
    "convnext-small": ("convnext_small", "ConvNeXt_Small_Weights.IMAGENET1K_V1", 768),
    "convnext-base": ("convnext_base", "ConvNeXt_Base_Weights.IMAGENET1K_V1", 1024),
    "convnext-large": ("convnext_large", "ConvNeXt_Large_Weights.IMAGENET1K_V1", 1536),
}


class _DenseNetFeatures(nn.Module):
    """torchvision DenseNet features followed by the final ReLU that its forward() applies before pooling."""

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        return F.relu(self.features(x), inplace=False)


class _InceptionV3Features(nn.Module):
    """Inception-v3 up to Mixed_7c, chained by hand (no auxiliary classifier, no input transform)."""

    def __init__(self, m):
        super().__init__()
        self.layers = nn.ModuleList([
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3, m.maxpool1,
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3, m.maxpool2,
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d,
            m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
            m.Mixed_7a, m.Mixed_7b, m.Mixed_7c,
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_backbone(name, pretrained=True):
    """Return (feature extractor, channels). Input [B, 3, 224, 224], ImageNet-normalized."""
    ctor, weights, channels = BACKBONES[name]
    cls_name, attr = weights.split(".")
    m = getattr(tvm, ctor)(weights=getattr(getattr(tvm, cls_name), attr) if pretrained else None)
    if name.startswith(("vgg", "efficient", "convnext")) or name == "mobilenet":
        return m.features, channels
    if name == "resnet-101":
        return nn.Sequential(*list(m.children())[:-2]), channels   # drop average pooling and fc
    if name.startswith("densenet"):
        return _DenseNetFeatures(m.features), channels
    return _InceptionV3Features(m), channels


def densenet169_features(pretrained=True):
    """DenseNet-169 feature extractor: [B, 3, 224, 224] -> [B, 1664, 7, 7]."""
    return build_backbone("densenet-169", pretrained)
