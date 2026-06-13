"""
NAFNet-style lightweight block for MoE bypass.

Extracted from: https://github.com/megvii-research/NAFNet
Paper: Simple Baselines for Image Restoration (ECCV 2022)

Used in MoCE-IR as a per-AdapterLayer bypass:
  out = MoE_out + blur_prob * NAF_out
"""

import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    """Replace GELU/ReLU with element-wise multiplication of two halves."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D features."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return self.weight.view(1, -1, 1, 1) * (x - mean) / (std + self.eps) + self.bias.view(1, -1, 1, 1)


class NAFBypass(nn.Module):
    """
    Lightweight NAF-style conv block for MoE bypass.

    Designed as a drop-in complement to MoE expert layer:
      - No gating/routing
      - Single conv path with SimpleGate activation
      - 2x DW_Expand: conv1x1 → DW conv3x3 → SimpleGate → SCA → conv1x1
      - No FFN (unlike full NAFBlock) to keep it truly lightweight

    Params per instance:
      dim (int): input channel dim (e.g. 48 for dec.0, 96 for dec.1, etc.)

    Params count: ~ 2 * dim * (dim*2) + 9*(dim*2) ≈ 4*dim² + 18*dim
      e.g. dim=48 → ~10K, dim=96 → ~38K
    """
    def __init__(self, dim, DW_Expand=2):
        super().__init__()
        dw_ch = dim * DW_Expand

        self.norm = LayerNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dw_ch, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1, groups=dw_ch, bias=True)  # depthwise
        self.conv3 = nn.Conv2d(dw_ch // 2, dim, kernel_size=1, padding=0, bias=True)

        # Simplified Channel Attention (channel-wise global context)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, kernel_size=1, bias=True),
        )

        self.sg = SimpleGate()
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x):
        residual = x
        x = self.norm(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        return residual + x * self.gamma
