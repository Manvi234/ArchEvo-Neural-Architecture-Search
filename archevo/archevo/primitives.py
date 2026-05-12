"""
archevo/primitives.py
---------------------
Primitive neural architecture operations used as building blocks in the search space.
Each op is an nn.Module with forward(x, stride=1) -> x interface.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class NormStrategy(nn.Module):
    """Wraps a normalisation layer; supports 'bn', 'ln', 'gn'."""

    def __init__(self, num_features: int, strategy: str = 'bn', num_groups: int = 8):
        super().__init__()
        strategy = strategy.lower()
        if strategy == 'bn':
            self.norm = nn.BatchNorm2d(num_features)
        elif strategy == 'ln':
            self.norm = nn.GroupNorm(1, num_features)  # LayerNorm over C,H,W
        elif strategy == 'gn':
            groups = min(num_groups, num_features)
            # ensure divisibility
            while num_features % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, num_features)
        else:
            raise ValueError(f"Unknown norm strategy: {strategy}. Choose 'bn','ln','gn'.")
        self.strategy = strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# ---------------------------------------------------------------------------
# ConvBlock: depthwise-separable convolution
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Depthwise-separable convolution block with BN + ReLU.

    kernel_size: 3 or 5
    For stride=2 the depthwise conv uses stride=2 (spatial downsampling).
    """

    def __init__(self, C_in: int, C_out: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size not in (3, 5):
            raise ValueError("kernel_size must be 3 or 5")
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        pad = kernel_size // 2

        # Strided version (stride=2)
        self.dw_s2 = nn.Conv2d(C_in, C_in, kernel_size, stride=2,
                                padding=pad, groups=C_in, bias=False)
        self.pw_s2 = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.bn_s2 = nn.BatchNorm2d(C_out)

        # Stride-1 version
        self.dw_s1 = nn.Conv2d(C_in, C_in, kernel_size, stride=1,
                                padding=pad, groups=C_in, bias=False)
        self.pw_s1 = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.bn_s1 = nn.BatchNorm2d(C_out)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, stride: int = 1) -> torch.Tensor:
        if stride == 2:
            out = self.dw_s2(x)
            out = self.pw_s2(out)
            out = self.bn_s2(out)
        else:
            out = self.dw_s1(x)
            out = self.pw_s1(out)
            out = self.bn_s1(out)
        return self.relu(out)


# ---------------------------------------------------------------------------
# SelfAttentionBlock: windowed multi-head attention (Swin-style)
# ---------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """
    Windowed multi-head self-attention block (Swin-style).

    For stride=2: strided average-pool after attention to downsample.
    Uses nn.MultiheadAttention internally with window partitioning.
    """

    def __init__(self, C_in: int, C_out: int, num_heads: int = 4, window_size: int = 4):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.num_heads = num_heads
        self.window_size = window_size

        # Project to C_out before attention
        self.proj_in = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.norm1 = nn.LayerNorm(C_out)
        self.attn = nn.MultiheadAttention(C_out, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(C_out)

        # Feed-forward within the block
        self.ffn = nn.Sequential(
            nn.Linear(C_out, C_out * 4),
            nn.GELU(),
            nn.Linear(C_out * 4, C_out),
        )

        self.proj_out = nn.Conv2d(C_out, C_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        self.pool = nn.AvgPool2d(2, stride=2)

    def _pad_to_window(self, x: torch.Tensor):
        """Pad H,W so they are divisible by window_size."""
        _, _, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, H, W

    def forward(self, x: torch.Tensor, stride: int = 1) -> torch.Tensor:
        B, C, H_orig, W_orig = x.shape
        # Project channels
        out = self.proj_in(x)  # (B, C_out, H, W)
        out, H, W = self._pad_to_window(out)
        _, _, Hp, Wp = out.shape
        ws = self.window_size

        nH, nW = Hp // ws, Wp // ws

        # Partition into windows: (B*nH*nW, ws*ws, C_out)
        out_rearr = rearrange(out, 'b c (nh wh) (nw ww) -> (b nh nw) (wh ww) c',
                              nh=nH, nw=nW, wh=ws, ww=ws)

        # Self-attention within windows
        normed = self.norm1(out_rearr)
        attn_out, _ = self.attn(normed, normed, normed)
        out_rearr = out_rearr + attn_out

        # FFN
        out_rearr = out_rearr + self.ffn(self.norm2(out_rearr))

        # Reverse window partition
        out = rearrange(out_rearr, '(b nh nw) (wh ww) c -> b c (nh wh) (nw ww)',
                        b=B, nh=nH, nw=nW, wh=ws, ww=ws)

        # Crop back to original (before padding)
        out = out[:, :, :H, :W]

        out = self.proj_out(out)
        out = self.bn(out)

        if stride == 2:
            out = self.pool(out)

        return F.relu(out)


# ---------------------------------------------------------------------------
# MLPMixerBlock: token-mixing MLP + channel-mixing MLP
# ---------------------------------------------------------------------------

class MLPMixerBlock(nn.Module):
    """
    MLP-Mixer block: token-mixing MLP + channel-mixing MLP.
    For stride=2: average-pool after mixing.
    Input: (B, C, H, W) – treated as (B, H*W tokens, C channels).
    """

    def __init__(self, C_in: int, C_out: int, image_size: int = 8):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.image_size = image_size  # default expected spatial size (used for token_dim)
        num_tokens = image_size * image_size

        # Channel projection if needed
        self.proj_in = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.bn_in = nn.BatchNorm2d(C_out)

        # Token mixing MLP (operates on spatial dim)
        self.norm1 = nn.LayerNorm(C_out)
        self.token_mix = nn.Sequential(
            nn.Linear(num_tokens, num_tokens * 2),
            nn.GELU(),
            nn.Linear(num_tokens * 2, num_tokens),
        )

        # Channel mixing MLP
        self.norm2 = nn.LayerNorm(C_out)
        self.channel_mix = nn.Sequential(
            nn.Linear(C_out, C_out * 4),
            nn.GELU(),
            nn.Linear(C_out * 4, C_out),
        )

        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x: torch.Tensor, stride: int = 1) -> torch.Tensor:
        B, C, H, W = x.shape
        out = F.relu(self.bn_in(self.proj_in(x)))  # (B, C_out, H, W)
        num_tokens = H * W

        # Reshape to (B, tokens, channels)
        tokens = rearrange(out, 'b c h w -> b (h w) c')  # (B, T, C_out)

        # Token mixing (operate on spatial positions)
        normed = self.norm1(tokens)  # (B, T, C_out)

        # We need to mix across T dimension; use a linear that accepts T tokens
        # Build a dynamic token mix if spatial size differs from default
        if num_tokens != self.image_size * self.image_size:
            # Fallback: use adaptive average pooling along token dim via interpolation
            normed_t = normed.transpose(1, 2)  # (B, C_out, T)
            # Create a temporary linear layer on the fly for arbitrary sizes
            target_t = self.image_size * self.image_size
            normed_t_interp = F.interpolate(normed_t, size=target_t, mode='linear', align_corners=False)
            mixed_t = self.token_mix(normed_t_interp)
            mixed_t = F.interpolate(mixed_t, size=num_tokens, mode='linear', align_corners=False)
            tokens = tokens + mixed_t.transpose(1, 2)
        else:
            normed_t = normed.transpose(1, 2)  # (B, C_out, T)
            mixed_t = self.token_mix(normed_t)   # (B, C_out, T)
            tokens = tokens + mixed_t.transpose(1, 2)

        # Channel mixing
        tokens = tokens + self.channel_mix(self.norm2(tokens))

        # Reshape back
        out = rearrange(tokens, 'b (h w) c -> b c h w', h=H, w=W)

        if stride == 2:
            out = self.pool(out)

        return out


# ---------------------------------------------------------------------------
# SkipConnection
# ---------------------------------------------------------------------------

class SkipConnection(nn.Module):
    """
    Identity for stride=1; 1x1 conv with stride for channel/spatial projection at stride=2.
    """

    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.proj = nn.Conv2d(C_in, C_out, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x: torch.Tensor, stride: int = 1) -> torch.Tensor:
        if stride == 2:
            return self.bn(self.proj(x))
        if self.C_in != self.C_out:
            # Channel-only projection (no spatial downsampling)
            proj1 = nn.Conv2d(self.C_in, self.C_out, 1, bias=False).to(x.device)
            bn1 = nn.BatchNorm2d(self.C_out).to(x.device)
            return bn1(proj1(x))
        return x


# ---------------------------------------------------------------------------
# ZeroOp
# ---------------------------------------------------------------------------

class ZeroOp(nn.Module):
    """Returns a zero tensor of the correct output shape (handles stride)."""

    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out

    def forward(self, x: torch.Tensor, stride: int = 1) -> torch.Tensor:
        B, C, H, W = x.shape
        out_H = H // stride if stride > 1 else H
        out_W = W // stride if stride > 1 else W
        return torch.zeros(B, self.C_out, out_H, out_W, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
# OPS_REGISTRY
# ---------------------------------------------------------------------------

def _make_conv3(C_in, C_out):
    return ConvBlock(C_in, C_out, kernel_size=3)

def _make_conv5(C_in, C_out):
    return ConvBlock(C_in, C_out, kernel_size=5)

def _make_attention(C_in, C_out):
    return SelfAttentionBlock(C_in, C_out)

def _make_mlp_mixer(C_in, C_out):
    return MLPMixerBlock(C_in, C_out)

def _make_skip(C_in, C_out):
    return SkipConnection(C_in, C_out)

def _make_zero(C_in, C_out):
    return ZeroOp(C_in, C_out)


OPS_REGISTRY: dict = {
    'conv3':      _make_conv3,
    'conv5':      _make_conv5,
    'attention':  _make_attention,
    'mlp_mixer':  _make_mlp_mixer,
    'skip':       _make_skip,
    'zero':       _make_zero,
}

OP_NAMES = list(OPS_REGISTRY.keys())
NUM_OPS = len(OP_NAMES)
