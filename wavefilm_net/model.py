"""WaveFiLM-Net and every head configuration of the study, built from one specification.

A specification has the fields below (defaults in DEFAULT_SPEC):

prior          none | threshold_mask | binary | soft | residual | film | blob_residual | ridge_blob_residual
subband        none | pooling_bands | fixed_haar | dhsr | dhsr_replicate | learned_pooling | no_subband
fusion         none | softmax_gate | sigmoid_gate | scalar_gate | two_way | sff
pooling        gap | context
prior_stream   False | True     separate HSP-map stream appended to the pooled vector
shuffle_prior  False | True     each image receives the prior map of the preceding image in its mini-batch
attention      none | se | cbam (only with all head modules set to none)
bilinear_rank  0 | r            low-rank bilinear head replacing the whole head (only on the plain backbone)
finetune       False | True     train the backbone (batch normalization in training mode)
lr_backbone    learning rate of the backbone when finetune is True

WaveFiLM-Net is {prior: residual, subband: dhsr, fusion: sff}; the LP is the default specification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .layers import DualStreamHaarRefinement, HessianStructurePrior, PriorGain, SpatialFiLMFusion
from .variants import (BilinearHead, CBAMBlock, ContextPooling, FixedHaarFusion, GatedFusion, HessianBlobPrior,
                       LearnedPoolingBandFusion, MultiPriorGain, NoSubbandControl, PoolingBandFusion, PriorFiLM,
                       PriorStream, SEBlock, SoftmaxGateFusion, ThresholdHessianMask, TwoWayFusion)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_SPEC = dict(prior="none", subband="none", fusion="none", pooling="gap", prior_stream=False,
                    shuffle_prior=False, attention="none", bilinear_rank=0, finetune=False, lr_backbone=None)
WAVEFILM_SPEC = dict(prior="residual", subband="dhsr", fusion="sff")
CHOICES = dict(prior={"none", "threshold_mask", "binary", "soft", "residual", "film", "blob_residual",
                      "ridge_blob_residual"},
               subband={"none", "pooling_bands", "fixed_haar", "dhsr", "dhsr_replicate", "learned_pooling",
                        "no_subband"},
               fusion={"none", "softmax_gate", "sigmoid_gate", "scalar_gate", "two_way", "sff"},
               pooling={"gap", "context"}, attention={"none", "se", "cbam"})


def resolve_spec(spec=None):
    s = {**DEFAULT_SPEC, **(spec or {})}
    unknown = set(s) - set(DEFAULT_SPEC)
    if unknown:
        raise KeyError(f"unknown specification fields: {sorted(unknown)}")
    for key, allowed in CHOICES.items():
        if s[key] not in allowed:
            raise ValueError(f"{key}={s[key]!r}, allowed: {sorted(allowed)}")
    plain = (s["prior"], s["subband"], s["fusion"], s["pooling"], s["prior_stream"]) == ("none", "none", "none", "gap", False)
    if s["attention"] != "none" and not plain:
        raise ValueError("attention blocks are used only with the plain backbone and GAP")
    if s["bilinear_rank"] and not (plain and s["attention"] == "none"):
        raise ValueError("the bilinear head is used only with the plain backbone")
    if bool(s["finetune"]) != (s["lr_backbone"] is not None):
        raise ValueError("finetune and lr_backbone are given together")
    return s


class HeadModel(nn.Module):
    """Backbone, then prior -> subband refinement -> fusion -> pooling -> linear classifier.

    Input images are in [0, 1]. ImageNet normalization is applied inside, and the prior maps are
    computed from the raw images. Without fusion the refined map is pooled directly, and without a
    subband module the fusion receives the prior-modulated stream S1.
    """

    def __init__(self, backbone, channels, num_classes, spec=None):
        super().__init__()
        self.spec = s = resolve_spec(spec)
        C = channels
        self.backbone = backbone
        self.finetune, self.lr_backbone = bool(s["finetune"]), s["lr_backbone"]
        for p in self.backbone.parameters():
            p.requires_grad = self.finetune
        self.attention = {"se": lambda: SEBlock(C), "cbam": lambda: CBAMBlock(C)}.get(s["attention"], lambda: None)()
        prior = s["prior"]
        self.threshold_mask = ThresholdHessianMask() if prior == "threshold_mask" else None
        needs_hsp = prior in {"binary", "soft", "residual", "film", "ridge_blob_residual"} or s["prior_stream"]
        self.prior_map = HessianStructurePrior() if needs_hsp else None
        self.blob_map = HessianBlobPrior() if prior in {"blob_residual", "ridge_blob_residual"} else None
        self.prior_gain = {"residual": PriorGain, "blob_residual": PriorGain,
                           "ridge_blob_residual": lambda: MultiPriorGain(2),
                           "film": lambda: PriorFiLM(C)}.get(prior, lambda: None)()
        self.subband = {"pooling_bands": PoolingBandFusion, "fixed_haar": FixedHaarFusion,
                        "dhsr": lambda: DualStreamHaarRefinement(C),
                        "dhsr_replicate": lambda: DualStreamHaarRefinement(C, "replicate"),
                        "learned_pooling": lambda: LearnedPoolingBandFusion(C),
                        "no_subband": lambda: NoSubbandControl(C)}.get(s["subband"], lambda: None)()
        self.fusion = {"softmax_gate": lambda: SoftmaxGateFusion(C),
                       "sigmoid_gate": lambda: GatedFusion(C, "sigmoid"),
                       "scalar_gate": lambda: GatedFusion(C, "scalar"),
                       "two_way": lambda: TwoWayFusion(C),
                       "sff": lambda: SpatialFiLMFusion(C)}.get(s["fusion"], lambda: None)()
        self.pool = ContextPooling(C) if s["pooling"] == "context" else None
        self.prior_stream = PriorStream() if s["prior_stream"] else None
        extra = self.prior_stream.dim + 1 if self.prior_stream is not None else 0
        self.bilinear = BilinearHead(C, s["bilinear_rank"], num_classes) if s["bilinear_rank"] else None
        self.classifier = None if self.bilinear is not None else nn.Linear(C + extra, num_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:
            self.backbone.eval()   # frozen backbone: batch normalization statistics stay fixed
        return self

    def param_groups(self, lr):
        """Optimizer groups: the head at lr, and with fine-tuning the backbone at lr_backbone."""
        head = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
        if not self.finetune:
            return [{"params": head, "lr": lr}]
        return [{"params": list(self.backbone.parameters()), "lr": self.lr_backbone}, {"params": head, "lr": lr}]

    def _maybe_shuffle(self, m):
        # shuffled-prior control: the map of the preceding image in the mini-batch, cyclically
        return torch.roll(m, shifts=1, dims=0) if self.spec["shuffle_prior"] and m.shape[0] > 1 else m

    def extract(self, x):
        x = (x - self.mean) / self.std
        if self.finetune:
            return self.backbone(x)
        with torch.no_grad():
            feat = self.backbone(x)
        return feat.detach()

    def forward(self, x):
        feat = self.extract(x)
        if self.bilinear is not None:
            return self.bilinear(feat)
        if self.attention is not None:
            return self.classifier(self.attention(feat).mean(dim=(2, 3)))
        size, prior = feat.shape[-2:], self.spec["prior"]

        def down(m):
            return F.interpolate(m, size=size, mode="bilinear", align_corners=False)

        hsp = None
        if self.prior_map is not None and (prior != "binary" or self.prior_stream is not None):
            hsp = self._maybe_shuffle(self.prior_map(x))
        blob = self._maybe_shuffle(self.blob_map(x)) if self.blob_map is not None else None

        s1 = feat
        if prior == "threshold_mask":
            s1 = feat * down(self._maybe_shuffle(self.threshold_mask(x)))
        elif prior == "binary":
            s1 = feat * down(self._maybe_shuffle(self.prior_map(x, binary=True)))
        elif prior == "soft":
            s1 = feat * (0.5 + 0.5 * down(hsp))
        elif prior in {"residual", "film"}:
            s1 = self.prior_gain(feat, down(hsp))
        elif prior == "blob_residual":
            s1 = self.prior_gain(feat, down(blob))
        elif prior == "ridge_blob_residual":
            s1 = self.prior_gain(feat, down(torch.cat([hsp, blob], dim=1)))

        refined = self.subband(s1, feat) if self.subband is not None else s1
        fused = self.fusion(refined, feat) if self.fusion is not None else refined
        pooled = self.pool(fused) if self.pool is not None else fused.mean(dim=(2, 3))
        if self.prior_stream is not None:
            pooled = torch.cat([pooled, self.prior_stream(hsp)], dim=1)
        return self.classifier(pooled)


def build_model(num_classes, spec=None, backbone="densenet-169", pretrained=True):
    net, channels = build_backbone(backbone, pretrained)
    return HeadModel(net, channels, num_classes, spec)


def build_wavefilm_net(num_classes, pretrained=True, use_prior=True, use_dhsr=True, use_sff=True,
                       backbone="densenet-169"):
    """WaveFiLM-Net. Setting use_prior, use_dhsr, or use_sff to False removes that module (ablations)."""
    spec = dict(prior="residual" if use_prior else "none", subband="dhsr" if use_dhsr else "none",
                fusion="sff" if use_sff else "none")
    return build_model(num_classes, spec, backbone, pretrained)


def build_linear_probe(num_classes, pretrained=True, backbone="densenet-169"):
    """Reference head: GAP and a linear classifier on the same frozen features."""
    return build_model(num_classes, None, backbone, pretrained)


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
