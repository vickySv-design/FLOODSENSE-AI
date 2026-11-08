"""Lightweight Dual-Stream Attention U-Net for FloodSense-AI.

Designed for B.E. Computer Science & Engineering Final-Year Capstone projects
and efficient local training/inference on an NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM)
or modern CPU fallback.

Architecture:
- Stream 1: Lightweight Optical Encoder (4-channel: Red, Green, Blue, NIR)
- Stream 2: Lightweight SAR Encoder (2-channel: VV, VH)
- Cross-Modal Attention Fusion (CMAF) with Squeeze-and-Excitation + Spatial Attention
- Additive Soft Attention Gates (Oktay et al.) on Skip Connections
- Filter Scales: [32, 64, 128, 256], Bottleneck: 466 channels (Target ~8.24M Parameters)
- 3 Semantic Classes: 0: Dry Land, 1: Permanent Water, 2: Inundated Land
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightConvBlock(nn.Module):
    """Double 3x3 Convolution with Batch Normalization and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightSEChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention module for sensor feature recalibration."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.avg_pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class LightweightSpatialAttention(nn.Module):
    """Spatial Attention module computing spatial saliency across optical and SAR features."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        spatial_map = self.sigmoid(self.conv(pooled))
        return x * spatial_map, spatial_map


class LightweightCrossModalAttentionFusion(nn.Module):
    """
    Lightweight Cross-Modal Attention Fusion (CMAF) Module.
    Concatenates Optical and SAR feature maps at scale i, applies Channel + Spatial
    attention to suppress optical cloud artifacts and SAR speckle noise, and projects
    to a unified fused representation.
    """

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        concat_channels = feature_channels * 2
        self.se_channel_attn = LightweightSEChannelAttention(concat_channels, reduction=8)
        self.spatial_attn = LightweightSpatialAttention(kernel_size=7)
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(concat_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, opt_feat: torch.Tensor, sar_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([opt_feat, sar_feat], dim=1)
        ch_refined = self.se_channel_attn(concat)
        sp_refined, spatial_map = self.spatial_attn(ch_refined)
        fused = self.fusion_proj(sp_refined)
        return fused, spatial_map


class LightweightAttentionGate(nn.Module):
    """
    Additive Soft Attention Gate (Oktay et al.) for skip connections:
    alpha = sigmoid(psi(ReLU(W_g(g) + W_x(x))))
    """

    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, g: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha, alpha


class LightweightUpBlock(nn.Module):
    """Upsampling stage with bilinear interpolation + 1x1 projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class LightweightDualStreamAttentionUNet(nn.Module):
    """
    Lightweight Dual-Stream Siamese Attention U-Net for Multi-Modal Flood Mapping.

    Inputs:
        optical: [B, 4, H, W] (Sentinel-2 Red, Green, Blue, NIR)
        sar:     [B, 2, H, W] (Sentinel-1 VV, VH backscatter in dB / normalized)
    Outputs:
        logits:        [B, num_classes, H, W]
        attention_map: [B, 1, H, W] upsampled multi-scale spatial attention saliency map
    """

    def __init__(
        self,
        optical_channels: int = 4,
        sar_channels: int = 2,
        num_classes: int = 3,
        filters: Tuple[int, int, int, int] = (32, 64, 128, 256),
        bottleneck_channels: int = 466,
    ) -> None:
        super().__init__()
        f1, f2, f3, f4 = filters
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.num_classes = num_classes
        self.filters = filters
        self.bottleneck_channels = bottleneck_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stream 1: Lightweight Optical Encoder
        self.opt_enc1 = LightweightConvBlock(optical_channels, f1)
        self.opt_enc2 = LightweightConvBlock(f1, f2)
        self.opt_enc3 = LightweightConvBlock(f2, f3)
        self.opt_enc4 = LightweightConvBlock(f3, f4)

        # Stream 2: Lightweight SAR Encoder
        self.sar_enc1 = LightweightConvBlock(sar_channels, f1)
        self.sar_enc2 = LightweightConvBlock(f1, f2)
        self.sar_enc3 = LightweightConvBlock(f2, f3)
        self.sar_enc4 = LightweightConvBlock(f3, f4)

        # Cross-Modal Attention Fusion (CMAF) at scales 1..4
        self.cmaf1 = LightweightCrossModalAttentionFusion(f1)
        self.cmaf2 = LightweightCrossModalAttentionFusion(f2)
        self.cmaf3 = LightweightCrossModalAttentionFusion(f3)
        self.cmaf4 = LightweightCrossModalAttentionFusion(f4)

        # Bottleneck Bridge
        self.bottleneck = LightweightConvBlock(f4, bottleneck_channels)

        # Decoder Stage 4
        self.up4 = LightweightUpBlock(bottleneck_channels, f4)
        self.att4 = LightweightAttentionGate(f_g=f4, f_l=f4, f_int=f3)
        self.dec4 = LightweightConvBlock(f4 * 2, f4)

        # Decoder Stage 3
        self.up3 = LightweightUpBlock(f4, f3)
        self.att3 = LightweightAttentionGate(f_g=f3, f_l=f3, f_int=f2)
        self.dec3 = LightweightConvBlock(f3 * 2, f3)

        # Decoder Stage 2
        self.up2 = LightweightUpBlock(f3, f2)
        self.att2 = LightweightAttentionGate(f_g=f2, f_l=f2, f_int=f1)
        self.dec2 = LightweightConvBlock(f2 * 2, f2)

        # Decoder Stage 1
        self.up1 = LightweightUpBlock(f2, f1)
        self.att1 = LightweightAttentionGate(f_g=f1, f_l=f1, f_int=max(f1 // 2, 4))
        self.dec1 = LightweightConvBlock(f1 * 2, f1)

        # 1x1 Semantic Segmentation Classification Head
        self.head = nn.Conv2d(f1, num_classes, kernel_size=1)

    def forward(
        self, optical: torch.Tensor, sar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = optical.shape

        # Stage 1
        o1 = self.opt_enc1(optical)
        s1 = self.sar_enc1(sar)
        skip1, sp_map1 = self.cmaf1(o1, s1)

        # Stage 2
        o2 = self.opt_enc2(self.pool(o1))
        s2 = self.sar_enc2(self.pool(s1))
        skip2, _ = self.cmaf2(o2, s2)

        # Stage 3
        o3 = self.opt_enc3(self.pool(o2))
        s3 = self.sar_enc3(self.pool(s2))
        skip3, _ = self.cmaf3(o3, s3)

        # Stage 4
        o4 = self.opt_enc4(self.pool(o3))
        s4 = self.sar_enc4(self.pool(s3))
        skip4, sp_map4 = self.cmaf4(o4, s4)

        # Bottleneck
        b = self.bottleneck(self.pool(skip4))

        # Decoder 4
        d4 = self.up4(b)
        g4, alpha4 = self.att4(g=d4, x=skip4)
        d4 = self.dec4(torch.cat([g4, d4], dim=1))

        # Decoder 3
        d3 = self.up3(d4)
        g3, alpha3 = self.att3(g=d3, x=skip3)
        d3 = self.dec3(torch.cat([g3, d3], dim=1))

        # Decoder 2
        d2 = self.up2(d3)
        g2, alpha2 = self.att2(g=d2, x=skip2)
        d2 = self.dec2(torch.cat([g2, d2], dim=1))

        # Decoder 1
        d1 = self.up1(d2)
        g1, alpha1 = self.att1(g=d1, x=skip1)
        d1 = self.dec1(torch.cat([g1, d1], dim=1))

        logits = self.head(d1)

        # Multi-scale Explainable AI (XAI) Saliency Map upsampled to (h, w)
        up_sp4 = F.interpolate(sp_map4, size=(h, w), mode="bilinear", align_corners=False)
        up_alpha3 = F.interpolate(alpha3, size=(h, w), mode="bilinear", align_corners=False)
        up_alpha2 = F.interpolate(alpha2, size=(h, w), mode="bilinear", align_corners=False)
        attention_map = (
            0.35 * up_sp4
            + 0.25 * up_alpha3
            + 0.20 * up_alpha2
            + 0.20 * (0.5 * alpha1 + 0.5 * sp_map1)
        )

        return logits, attention_map

    def get_parameter_count(self) -> Dict[str, Any]:
        """Calculates detailed parameter counts and memory requirements."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = round((total_params * 4) / (1024 * 1024), 2)

        return {
            "model_name": "LightweightDualStreamAttentionUNet",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_float32_mb": model_size_mb,
            "filters": list(self.filters),
            "bottleneck_channels": self.bottleneck_channels,
            "optical_channels": self.optical_channels,
            "sar_channels": self.sar_channels,
            "num_classes": self.num_classes,
        }

    def save_checkpoint(
        self,
        checkpoint_path: Path | str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Saves model state and metadata to a distinct lightweight checkpoint file."""
        p = Path(checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "architecture": "LightweightDualStreamAttentionUNet",
            "optical_channels": self.optical_channels,
            "sar_channels": self.sar_channels,
            "num_classes": self.num_classes,
            "filters": self.filters,
            "bottleneck_channels": self.bottleneck_channels,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "epoch": epoch,
            "metrics": metrics or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, p)
        return str(p)

    def load_checkpoint(
        self, checkpoint_path: Path | str, device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Loads weights from checkpoint file."""
        p = Path(checkpoint_path)
        map_loc = device or next(self.parameters()).device
        state = torch.load(p, map_location=map_loc, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"])
            metadata = {k: v for k, v in state.items() if k != "model_state_dict"}
        else:
            self.load_state_dict(state)
            metadata = {}
        self.eval()
        return metadata
