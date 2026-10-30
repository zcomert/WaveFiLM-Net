"""Unit tests that need no image data and no pretrained weights."""
import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavefilm_net.backbones import densenet169_features  # noqa: E402
from wavefilm_net.layers import (DualStreamHaarRefinement, HaarDWT2d, HessianStructurePrior,  # noqa: E402
                                 SpatialFiLMFusion)
from wavefilm_net.model import HeadModel, trainable_parameters  # noqa: E402
from wavefilm_net.stats import decide, holm, macro_f1_batch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "configs", "experiments.yaml"), encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)

# trainable parameters on frozen DenseNet-169 with four classes (manuscript Tables 3, 4, and 7)
EXPECTED = {"lp": 6_660, "gf_head": 11_087_236, "hsp": 6_661, "blob_prior": 6_661, "hsp_dhsr": 5_564_421,
            "hsp_no_subband": 5_564_421, "hsp_dhsr_softmax_gate": 16_644_997, "hsp_dhsr_two_way": 5_567_751,
            "wavefilm_net": 11_105_541, "wavefilm_net_without_hsp": 11_105_540,
            "wavefilm_net_without_dhsr": 5_547_781, "wavefilm_net_shuffled_prior": 11_105_541,
            "wavefilm_net_blob_prior": 11_105_541, "wavefilm_net_context_pool": 11_107_206,
            "wavefilm_net_prior_stream": 11_130_154, "wavefilm_net_no_subband_control": 11_105_541,
            "wavefilm_net_replicate_padding": 11_105_541, "bilinear_512": 1_707_012, "bilinear_3328": 11_095_556}


def frozen_densenet(spec, num_classes=4):
    return HeadModel(*densenet169_features(pretrained=False), num_classes, spec)


def test_haar_roundtrip_even_and_odd():
    for pad_mode in ("zero", "replicate"):
        dwt = HaarDWT2d(pad_mode)
        for size in (7, 8):
            x = torch.randn(2, 3, size, size)
            ll, lh, hl, hh, shape = dwt(x)
            assert torch.allclose(dwt.inverse(ll, lh, hl, hh, shape), x, atol=1e-6)


def test_zero_init_head_is_linear_probe_on_pooled_features():
    feat = torch.randn(2, 16, 7, 7).relu()
    dhsr, sff = DualStreamHaarRefinement(16), SpatialFiLMFusion(16)
    out = sff(dhsr(feat, feat), feat)
    assert torch.allclose(out, feat, atol=1e-6)


def test_prior_map_range_and_shape():
    x = torch.rand(2, 3, 224, 224)
    m = HessianStructurePrior()(x)
    assert m.shape == (2, 1, 224, 224) and float(m.min()) >= 0 and float(m.max()) <= 1 + 1e-6


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_trainable_parameter_count_matches_manuscript(name):
    model = frozen_densenet(REGISTRY["configurations"][name]["spec"])
    assert trainable_parameters(model) == EXPECTED[name]


def test_every_configuration_builds_and_runs():
    x = torch.rand(2, 3, 224, 224)
    backbone, channels = densenet169_features(pretrained=False)
    for name, entry in REGISTRY["configurations"].items():
        model = HeadModel(backbone, channels, 3, entry["spec"]).eval()
        with torch.no_grad():
            assert model(x).shape == (2, 3), name


def test_registered_experiments_use_known_configurations():
    for e in REGISTRY["experiments"]:
        assert e["configuration"] in REGISTRY["configurations"]
        assert e["reference"] is None or e["reference"] in REGISTRY["configurations"]


def test_backbone_frozen_and_batchnorm_in_inference_mode():
    model = frozen_densenet(REGISTRY["configurations"]["wavefilm_net"]["spec"]).train()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not any(m.training for m in model.backbone.modules())
    tuned = frozen_densenet(REGISTRY["configurations"]["finetuned"]["spec"]).train()
    assert all(p.requires_grad for p in tuned.backbone.parameters()) and tuned.backbone.training
    assert [g["lr"] for g in tuned.param_groups(1e-3)] == [1e-4, 1e-3]


def test_stats_helpers():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert float(macro_f1_batch(y, y, 4)) == 1.0
    assert np.allclose(holm([0.01, 0.04]), [0.02, 0.04])
    assert decide(2.0, 0.5, 3.5, 0.01, 5, 5) == "Improves"
    assert decide(0.1, -0.8, 1.0, 0.5, 2, 5) == "Equivalent"
