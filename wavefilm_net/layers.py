"""Components of WaveFiLM-Net: HSP, DHSR, SFF.

HSP   Hessian structure prior: a parameter-free vesselness map computed from the raw image,
      injected into the frozen features through a single learned residual gain.
DHSR  Dual-stream Haar subband refinement of the prior-modulated and raw feature streams.
SFF   Spatial feature-wise linear modulation (position-wise FiLM) of the raw stream.

The prior gain lambda, the approximation convolution W_A and pointwise convolution W_P of
DHSR, and the SFF convolution are zero-initialized. Because SFF is zero-initialized, H = F at
initialization, and the model starts as a linear probe (LP) on GAP(F).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_DXX = torch.tensor([[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]]).view(1, 1, 3, 3)
_DYY = _DXX.transpose(-1, -2).contiguous()
_DXY = (torch.tensor([[1., 0., -1.], [0., 0., 0.], [-1., 0., 1.]]) / 4.0).view(1, 1, 3, 3)


def _gaussian_kernel(sigma, device, dtype):
    radius = int(math.ceil(3 * sigma))
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k1 = torch.exp(-(axis ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    return (k1[:, None] * k1[None, :]).view(1, 1, 2 * radius + 1, 2 * radius + 1)


class HessianStructurePrior(nn.Module):
    """Multi-scale Frangi vesselness map m in [0, 1] (no parameters).

    Input: images in [0, 1], shape [B, 3, H, W]. Output: [B, 1, H, W], the scale-averaged
    response divided by its per-image maximum. gamma is half the per-image, per-scale maximum
    Hessian norm. The response is zeroed where the larger-magnitude eigenvalue is positive,
    which restricts it to bright ridge-like structures.
    """

    def __init__(self, sigmas=(1.0, 2.0, 3.0), beta=0.5, threshold=0.1, eps=1e-8):
        super().__init__()
        self.sigmas, self.beta, self.threshold, self.eps = tuple(sigmas), beta, threshold, eps

    @torch.no_grad()
    def response(self, x):
        """Scale-averaged vesselness before normalization, [B, 1, H, W]."""
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        device, dtype = x.device, x.dtype
        responses = []
        for s in self.sigmas:
            k = _gaussian_kernel(s, device, dtype)
            blurred = F.conv2d(F.pad(gray, [k.shape[-1] // 2] * 4, mode="replicate"), k)
            padded = F.pad(blurred, [1, 1, 1, 1], mode="replicate")
            hxx = F.conv2d(padded, _DXX.to(device, dtype)) * s ** 2
            hyy = F.conv2d(padded, _DYY.to(device, dtype)) * s ** 2
            hxy = F.conv2d(padded, _DXY.to(device, dtype)) * s ** 2
            trace, det = hxx + hyy, hxx * hyy - hxy ** 2
            root = torch.sqrt(F.relu(trace ** 2 - 4 * det))
            l1, l2 = (trace - root) / 2, (trace + root) / 2
            swap = l1.abs() > l2.abs()
            l1, l2 = torch.where(swap, l2, l1), torch.where(swap, l1, l2)   # |l1| <= |l2|
            blob_ratio = l1.abs() / (l2.abs() + self.eps)
            norm = torch.sqrt(l1 ** 2 + l2 ** 2)
            gamma = 0.5 * norm.amax(dim=(2, 3), keepdim=True) + self.eps
            v = torch.exp(-blob_ratio ** 2 / (2 * self.beta ** 2)) * (1 - torch.exp(-norm ** 2 / (2 * gamma ** 2)))
            responses.append(torch.where(l2 > 0, torch.zeros_like(v), v))
        return torch.stack(responses, 0).mean(0)

    @torch.no_grad()
    def forward(self, x, binary=False):
        v = self.response(x)
        if binary:   # binary-prior variant: 1 where the unnormalized response exceeds the threshold, else 0.5
            return torch.where(v > self.threshold, torch.ones_like(v), torch.full_like(v, 0.5))
        return v / (v.amax(dim=(2, 3), keepdim=True) + self.eps)


class PriorGain(nn.Module):
    """S1 = F * (1 + lambda * m) with a single zero-initialized scalar lambda."""

    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.zeros(1))

    def forward(self, feat, prior):
        return feat * (1 + self.lam * prior)


class HaarDWT2d(nn.Module):
    """Orthonormal single-level 2D Haar transform. Odd sizes are padded on the right and bottom
    (zeros, or a copy of the last row and column with pad_mode="replicate" for the padding
    control), and the inverse transform crops back (exact reconstruction)."""

    def __init__(self, pad_mode="zero"):
        super().__init__()
        self.pad_mode = pad_mode
        ll = torch.tensor([[1., 1.], [1., 1.]]) / 2
        lh = torch.tensor([[1., 1.], [-1., -1.]]) / 2
        hl = torch.tensor([[1., -1.], [1., -1.]]) / 2
        hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2
        self.register_buffer("k", torch.stack([ll, lh, hl, hh]).unsqueeze(1))   # [4, 1, 2, 2]

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.pad(x, [0, w % 2, 0, h % 2], mode="replicate" if self.pad_mode == "replicate" else "constant")
        y = F.conv2d(x.reshape(b * c, 1, *x.shape[-2:]), self.k, stride=2)
        y = y.reshape(b, c, 4, *y.shape[-2:])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3], (h, w)

    def inverse(self, ll, lh, hl, hh, size):
        b, c, h, w = ll.shape
        y = torch.stack([ll, lh, hl, hh], 2).reshape(b * c, 4, h, w)
        x = F.conv_transpose2d(y, self.k, stride=2).reshape(b, c, 2 * h, 2 * w)
        return x[..., :size[0], :size[1]]


class DualStreamHaarRefinement(nn.Module):
    """DHSR. Approximation bands of the two streams are averaged and detail bands summed, then
    refined with a residual 1x1 convolution (approximation) and a depthwise 3x3 plus pointwise
    1x1 pair shared across orientations (details). Output Z has the shape of the input maps."""

    def __init__(self, channels, pad_mode="zero"):
        super().__init__()
        self.dwt = HaarDWT2d(pad_mode)
        self.ll_conv = nn.Conv2d(channels, channels, 1)
        self.hf_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.hf_pw = nn.Conv2d(channels, channels, 1)
        for m in (self.ll_conv, self.hf_pw):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def merge(self, stream_prior, stream_raw):
        a1, *d1, size = self.dwt(stream_prior)
        a0, *d0, _ = self.dwt(stream_raw)
        return 0.5 * (a1 + a0), [p + q for p, q in zip(d1, d0)], size

    def forward(self, stream_prior, stream_raw):
        ll, details, size = self.merge(stream_prior, stream_raw)
        ll = ll + self.ll_conv(ll)
        details = [d + self.hf_pw(self.hf_dw(d)) for d in details]
        return self.dwt.inverse(ll, *details, size)


class SpatialFiLMFusion(nn.Module):
    """SFF. (alpha, beta) = W Z + b with channel- and position-dependent coefficients, output
    (1 + alpha) * F + beta. W and b are zero-initialized, so the output equals F at the start."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2 * channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, refined, raw):
        alpha, beta = self.conv(refined).chunk(2, dim=1)
        return (1 + alpha) * raw + beta
