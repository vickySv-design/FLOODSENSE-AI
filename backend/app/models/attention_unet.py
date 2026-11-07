from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double 3x3 Convolution with BatchNorm and ReLU activation."""

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


class SqueezeExcitationChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) Channel Attention module for multi-modal sensor weighting."""

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


class SpatialAttentionBlock(nn.Module):
    """Spatial Attention block computing channel-pooled spatial saliency map."""

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


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion (CMAF) Module.
    Concatenates Optical and SAR feature maps at scale i, applies Squeeze-and-Excitation
    Channel Attention + Spatial Attention to suppress optical cloud artifacts and SAR speckle,
    and projects to fused skip connection representation.
    """

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        concat_channels = feature_channels * 2
        self.se_channel_attn = SqueezeExcitationChannelAttention(concat_channels, reduction=8)
        self.spatial_attn = SpatialAttentionBlock(kernel_size=7)
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


class AttentionGate(nn.Module):
    """
    Additive Soft Attention Gate (Oktay et al.) for skip connections:
    alpha = sigma(psi(ReLU(W_g(g) + W_x(x) + b)))
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


class UpBlock(nn.Module):
    """Upsampling stage with bilinear interpolation + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class DualStreamAttentionUNet(nn.Module):
    """
    Dual-Stream Siamese Attention U-Net for Multi-Modal Optical + SAR Change Detection.

    Inputs:
        optical: [B, 4, H, W] (Pre-Flood Sentinel-2 Red, Green, Blue, NIR)
        sar:     [B, 2, H, W] (Post-Flood Sentinel-1 VV, VH backscatter)
    Outputs:
        logits:        [B, num_classes, H, W]
        attention_map: [B, 1, H, W] upsampled bottleneck + gate spatial activation map
    """

    def __init__(
        self,
        optical_channels: int = 4,
        sar_channels: int = 2,
        num_classes: int = 3,
        filters: Tuple[int, int, int, int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        f1, f2, f3, f4 = filters

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stream 1: Pre-Event Optical Encoder (4-channel RGB+NIR)
        self.opt_enc1 = ConvBlock(optical_channels, f1)
        self.opt_enc2 = ConvBlock(f1, f2)
        self.opt_enc3 = ConvBlock(f2, f3)
        self.opt_enc4 = ConvBlock(f3, f4)

        # Stream 2: Post-Event SAR Encoder (2-channel VV+VH)
        self.sar_enc1 = ConvBlock(sar_channels, f1)
        self.sar_enc2 = ConvBlock(f1, f2)
        self.sar_enc3 = ConvBlock(f2, f3)
        self.sar_enc4 = ConvBlock(f3, f4)

        # Cross-Modal Attention Fusion (CMAF) at scales 1..4
        self.cmaf1 = CrossModalAttentionFusion(f1)
        self.cmaf2 = CrossModalAttentionFusion(f2)
        self.cmaf3 = CrossModalAttentionFusion(f3)
        self.cmaf4 = CrossModalAttentionFusion(f4)

        # Bottleneck Bridge
        self.bottleneck = ConvBlock(f4, f4 * 2)  # 1024 channels at 16x16

        # Decoder Stage 4 (16x16 -> 32x32)
        self.up4 = UpBlock(f4 * 2, f4)
        self.att4 = AttentionGate(f_g=f4, f_l=f4, f_int=f3)
        self.dec4 = ConvBlock(f4 * 2, f4)

        # Decoder Stage 3 (32x32 -> 64x64)
        self.up3 = UpBlock(f4, f3)
        self.att3 = AttentionGate(f_g=f3, f_l=f3, f_int=f2)
        self.dec3 = ConvBlock(f3 * 2, f3)

        # Decoder Stage 2 (64x64 -> 128x128)
        self.up2 = UpBlock(f3, f2)
        self.att2 = AttentionGate(f_g=f2, f_l=f2, f_int=f1)
        self.dec2 = ConvBlock(f2 * 2, f2)

        # Decoder Stage 1 (128x128 -> 256x256)
        self.up1 = UpBlock(f2, f1)
        self.att1 = AttentionGate(f_g=f1, f_l=f1, f_int=f1 // 2)
        self.dec1 = ConvBlock(f1 * 2, f1)

        # 1x1 Semantic Segmentation Classification Head
        self.head = nn.Conv2d(f1, num_classes, kernel_size=1)

    def forward(
        self, optical: torch.Tensor, sar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = optical.shape

        # Stage 1 (256x256)
        o1 = self.opt_enc1(optical)
        s1 = self.sar_enc1(sar)
        skip1, sp_map1 = self.cmaf1(o1, s1)

        # Stage 2 (128x128)
        o2 = self.opt_enc2(self.pool(o1))
        s2 = self.sar_enc2(self.pool(s1))
        skip2, _ = self.cmaf2(o2, s2)

        # Stage 3 (64x64)
        o3 = self.opt_enc3(self.pool(o2))
        s3 = self.sar_enc3(self.pool(s2))
        skip3, _ = self.cmaf3(o3, s3)

        # Stage 4 (32x32)
        o4 = self.opt_enc4(self.pool(o3))
        s4 = self.sar_enc4(self.pool(s3))
        skip4, sp_map4 = self.cmaf4(o4, s4)

        # Bottleneck (16x16)
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

        # Multi-scale Explainable AI (XAI) Attention Activation Map upsampled to (H, W)
        up_sp4 = F.interpolate(sp_map4, size=(h, w), mode="bilinear", align_corners=False)
        up_alpha3 = F.interpolate(alpha3, size=(h, w), mode="bilinear", align_corners=False)
        up_alpha2 = F.interpolate(alpha2, size=(h, w), mode="bilinear", align_corners=False)
        attention_map = 0.35 * up_sp4 + 0.25 * up_alpha3 + 0.20 * up_alpha2 + 0.20 * (0.5 * alpha1 + 0.5 * sp_map1)

        return logits, attention_map

