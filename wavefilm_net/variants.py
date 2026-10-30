"""Alternatives evaluated against the WaveFiLM-Net components, comparators, and controls.

Grouped by development stage (manuscript Sections 2.3 and 2.5, Tables 3, 4, 6, and 7):

prior     ThresholdHessianMask (fixed-threshold Hessian mask of the GF head), HessianBlobPrior,
          MultiPriorGain (ridge plus blob), PriorFiLM
subband   PoolingBandFusion (2 x 2 average-pooling band fusion of the GF head), FixedHaarFusion,
          LearnedPoolingBandFusion, NoSubbandControl
fusion    SoftmaxGateFusion (channel-softmax gated fusion of the GF head), GatedFusion (sigmoid or
          scalar gates in the same structure), TwoWayFusion (per-location two-way softmax)
head      ContextPooling, PriorStream, BilinearHead
attention SEBlock, CBAMBlock (standard initialization, applied before GAP)

Learnable alternatives to a WaveFiLM-Net component zero-initialize their last layer, so each
variant starts from the output of its reference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import HaarDWT2d, _DXX, _DXY, _DYY, _gaussian_kernel

# ---------------------------------------------------------------- prior


def _fixed_gaussian_kernel(sigma, ksize=5, device=None, dtype=torch.float32):
    ax = torch.arange(-(ksize // 2), ksize // 2 + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).view(1, 1, ksize, ksize)


_GF_DXX = torch.tensor([[1., -2., 1.], [1., -2., 1.], [1., -2., 1.]]).view(1, 1, 3, 3)
_GF_DYY = _GF_DXX.transpose(-1, -2).contiguous()
_GF_DXY = torch.tensor([[1., 0., -1.], [0., 0., 0.], [-1., 0., 1.]]).view(1, 1, 3, 3)


class ThresholdHessianMask(nn.Module):
    """Mask of the GF head: 1 where the scale-averaged Frangi response exceeds 0.1, 0.5 elsewhere.

    Fixed gamma = 15, no sigma^2 factor, 5 x 5 Gaussian kernels, the stencils
    Dxx = [1, 1, 1]^T [1, -2, 1], Dyy = Dxx^T, Dxy = [1, 0, -1]^T [1, 0, -1], zero padding.
    """

    def __init__(self, sigmas=(1.0, 2.0, 3.0), beta=0.5, gamma=15.0, threshold=0.1):
        super().__init__()
        self.sigmas, self.beta, self.gamma, self.threshold = sigmas, beta, gamma, threshold

    def forward(self, x):
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        responses = []
        for sigma in self.sigmas:
            blurred = F.conv2d(gray, _fixed_gaussian_kernel(sigma, device=x.device, dtype=x.dtype), padding="same")
            dxx = F.conv2d(blurred, _GF_DXX.to(x.device, x.dtype), padding="same")
            dyy = F.conv2d(blurred, _GF_DYY.to(x.device, x.dtype), padding="same")
            dxy = F.conv2d(blurred, _GF_DXY.to(x.device, x.dtype), padding="same")
            trace, det = dxx + dyy, dxx * dyy - dxy ** 2
            root = torch.sqrt(F.relu(trace ** 2 - 4 * det))
            l1, l2 = (trace - root) / 2, (trace + root) / 2
            swap = l1.abs() > l2.abs()
            l1, l2 = torch.where(swap, l2, l1), torch.where(swap, l1, l2)
            blob_ratio = l1.abs() / (l2.abs() + 1e-10)
            norm = torch.sqrt(l1 ** 2 + l2 ** 2)
            v = torch.exp(-(blob_ratio ** 2) / (2 * self.beta ** 2)) * (1 - torch.exp(-(norm ** 2) / (2 * self.gamma ** 2)))
            responses.append(torch.where(l2 > 0, torch.zeros_like(v), v))
        binary = (torch.stack(responses, dim=0).mean(dim=0) > self.threshold).to(x.dtype)
        return torch.where(binary == 0.0, torch.full_like(binary, 0.5), binary)


class HessianBlobPrior(nn.Module):
    """Bright-blob map (parameter-free): the square root of the positive scale-normalized Hessian
    determinant where the trace is negative, maximized over scales 4, 8, and 16, divided by its
    per-image 99.5th percentile and clipped to [0, 1]."""

    def __init__(self, sigmas=(4.0, 8.0, 16.0), q=0.995, eps=1e-6):
        super().__init__()
        self.sigmas, self.q, self.eps = tuple(sigmas), q, eps

    @torch.no_grad()
    def forward(self, x):
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        device, dtype = x.device, x.dtype
        out = torch.zeros_like(gray)
        for s in self.sigmas:
            k = _gaussian_kernel(s, device, dtype)
            blurred = F.conv2d(F.pad(gray, [k.shape[-1] // 2] * 4, mode="replicate"), k)
            padded = F.pad(blurred, [1, 1, 1, 1], mode="replicate")
            hxx = F.conv2d(padded, _DXX.to(device, dtype)) * s ** 2
            hyy = F.conv2d(padded, _DYY.to(device, dtype)) * s ** 2
            hxy = F.conv2d(padded, _DXY.to(device, dtype)) * s ** 2
            magnitude = torch.sqrt(F.relu(hxx * hyy - hxy ** 2))
            out = torch.maximum(out, magnitude * ((hxx + hyy) < 0).to(dtype))
        ref = torch.quantile(out.flatten(1), self.q, dim=1).view(-1, 1, 1, 1)
        return (out / (ref + self.eps)).clamp(0, 1)


class MultiPriorGain(nn.Module):
    """S1 = F * (1 + sum_i lambda_i m_i) with one zero-initialized scalar per map; m: [B, n, h, w]."""

    def __init__(self, n_maps):
        super().__init__()
        self.lam = nn.Parameter(torch.zeros(n_maps))

    def forward(self, feat, maps):
        return feat * (1 + (self.lam.view(1, -1, 1, 1) * maps).sum(dim=1, keepdim=True))


class PriorFiLM(nn.Module):
    """(alpha, beta) from the prior map through a two-layer 1 x 1 network, S1 = (1 + alpha) * F + beta.
    The output layer is zero-initialized."""

    def __init__(self, channels, hidden=32):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, hidden, 1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(hidden, 2 * channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat, prior):
        alpha, beta = self.out(self.body(prior)).chunk(2, dim=1)
        return (1 + alpha) * feat + beta

# ---------------------------------------------------------------- subband


class PoolingBandFusion(nn.Module):
    """Band fusion of the GF head (parameter-free). B(X) = U(P(X)) with 2 x 2 average pooling in
    ceiling mode and nearest-neighbor upsampling (half-pixel centers), R(X) = X - B(X), and
    Q = (B(M) + B(F)) / 2 + R(M) + R(F)."""

    @staticmethod
    def split(x):
        approx = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        return approx, x - F.interpolate(approx, size=x.shape[-2:], mode="nearest-exact")

    @staticmethod
    def merge(approx, detail):
        return F.interpolate(approx, size=detail.shape[-2:], mode="nearest-exact") + detail

    def forward(self, stream_prior, stream_raw):
        a1, d1 = self.split(stream_prior)
        a0, d0 = self.split(stream_raw)
        return self.merge(0.5 * (a1 + a0), d1 + d0)


class LearnedPoolingBandFusion(PoolingBandFusion):
    """Pooling band fusion with a learnable coarse band: a zero-initialized residual depthwise 3 x 3
    convolution on the merged approximation."""

    def __init__(self, channels):
        super().__init__()
        self.approx_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        nn.init.zeros_(self.approx_conv.weight)
        nn.init.zeros_(self.approx_conv.bias)

    def forward(self, stream_prior, stream_raw):
        a1, d1 = self.split(stream_prior)
        a0, d0 = self.split(stream_raw)
        approx = 0.5 * (a1 + a0)
        return self.merge(approx + self.approx_conv(approx), d1 + d0)


class FixedHaarFusion(nn.Module):
    """Haar fusion without learned refinement: averaged approximations, summed details, inverse transform."""

    def __init__(self, pad_mode="zero"):
        super().__init__()
        self.dwt = HaarDWT2d(pad_mode)

    def forward(self, stream_prior, stream_raw):
        a1, *d1, size = self.dwt(stream_prior)
        a0, *d0, _ = self.dwt(stream_raw)
        return self.dwt.inverse(0.5 * (a1 + a0), *[p + q for p, q in zip(d1, d0)], size)


class NoSubbandControl(nn.Module):
    """The layers and parameter count of DHSR without the Haar transform:
    S + W_A S + W_P(DW(S)) with S = (S1 + S0) / 2, W_A and W_P zero-initialized."""

    def __init__(self, channels):
        super().__init__()
        self.ll_conv = nn.Conv2d(channels, channels, 1)
        self.hf_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.hf_pw = nn.Conv2d(channels, channels, 1)
        for m in (self.ll_conv, self.hf_pw):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, stream_prior, stream_raw):
        z = 0.5 * (stream_prior + stream_raw)
        return z + self.ll_conv(z) + self.hf_pw(self.hf_dw(z))

# ---------------------------------------------------------------- fusion


class SoftmaxGateFusion(nn.Module):
    """Fusion of the GF head: alpha_Q = softmax(W_Q GAP(Q)) and alpha_F = softmax(W_F GAP(F)) over
    channels, output alpha_Q Q + alpha_F F + W_c [alpha_Q Q; alpha_F F] (4 C^2 + 3 C parameters)."""

    def __init__(self, channels):
        super().__init__()
        self.gate_q = nn.Linear(channels, channels)
        self.gate_f = nn.Linear(channels, channels)
        self.conv = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def gates(self, q, f):
        return (F.softmax(self.gate_q(q.mean(dim=(2, 3))), dim=-1)[..., None, None],
                F.softmax(self.gate_f(f.mean(dim=(2, 3))), dim=-1)[..., None, None])

    def forward(self, q, f):
        gq, gf = self.gates(q, f)
        a, b = q * gq, f * gf
        return a + b + self.conv(torch.cat([a, b], dim=1))


class GatedFusion(SoftmaxGateFusion):
    """The GF fusion structure with independent per-channel sigmoid gates (mode="sigmoid") or one
    sigmoid gate per stream (mode="scalar")."""

    def __init__(self, channels, mode):
        super().__init__(channels)
        assert mode in ("sigmoid", "scalar")
        if mode == "scalar":
            self.gate_q = nn.Linear(channels, 1)
            self.gate_f = nn.Linear(channels, 1)

    def gates(self, q, f):
        return (torch.sigmoid(self.gate_q(q.mean(dim=(2, 3))))[..., None, None],
                torch.sigmoid(self.gate_f(f.mean(dim=(2, 3))))[..., None, None])


class TwoWayFusion(nn.Module):
    """Per-location two-way softmax fusion computed from the stream average (zero-initialized,
    so the output starts at (S1 + S0) / 2)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, s1, s0):
        w = torch.softmax(self.conv(0.5 * (s1 + s0)), dim=1)
        return w[:, 0:1] * s1 + w[:, 1:2] * s0

# ---------------------------------------------------------------- head


class ContextPooling(nn.Module):
    """A 1 x 1 convolution and spatial softmax replace GAP (zero-initialized, so it starts as GAP)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        w = torch.softmax(self.conv(x).flatten(2), dim=-1)
        return torch.bmm(x.flatten(2), w.transpose(1, 2)).squeeze(-1)


class PriorStream(nn.Module):
    """Separate prior stream: a three-block encoder of the HSP map with a scalar gate and 20% map
    dropout during training. Output [gated encoding (64), map-present flag (1)], appended to the
    pooled vector."""

    def __init__(self, dim=64, p_drop=0.2):
        super().__init__()
        blocks, cin = [], 1
        for cout in (16, 32, dim):
            blocks += [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            cin = cout
        self.enc = nn.Sequential(*blocks)
        self.gate = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
        self.p_drop, self.dim = p_drop, dim

    def forward(self, m):
        flag = torch.ones(m.shape[0], 1, device=m.device, dtype=m.dtype)
        if self.training and self.p_drop > 0:
            keep = (torch.rand(m.shape[0], 1, device=m.device) >= self.p_drop).to(m.dtype)
            m, flag = m * keep[..., None, None], keep
        h = self.enc(m).mean(dim=(2, 3))
        return torch.cat([torch.sigmoid(self.gate(h)) * h, flag], dim=1)


class BilinearHead(nn.Module):
    """Low-rank bilinear pooling: two learned 1 x 1 projections to rank r multiplied at each position,
    averaged, signed square root, l2 normalization, linear classifier."""

    def __init__(self, channels, rank, num_classes):
        super().__init__()
        self.u = nn.Conv2d(channels, rank, 1)
        self.v = nn.Conv2d(channels, rank, 1)
        self.fc = nn.Linear(rank, num_classes)

    def forward(self, x):
        z = (self.u(x) * self.v(x)).mean(dim=(2, 3))
        z = torch.sign(z) * torch.sqrt(z.abs() + 1e-6)
        return self.fc(F.normalize(z, dim=1))

# ---------------------------------------------------------------- attention


class SEBlock(nn.Module):
    """Squeeze-and-excitation [Hu2018], reduction ratio 16."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean(dim=(2, 3))))[..., None, None]


class CBAMBlock(nn.Module):
    """Convolutional block attention module [Woo2018]: channel attention (average and max pooling,
    shared MLP, reduction ratio 16), then spatial attention (7 x 7 convolution)."""

    def __init__(self, channels, reduction=16, kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        x = x * torch.sigmoid(self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3))))[..., None, None]
        return x * torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
