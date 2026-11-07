import base64
import io
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
import torch

from app.config import (
    CLASS_NEW_FLOOD,
    CLASS_PERMANENT_WATER,
    HIGH_RISK_BUFFER_PIXELS,
    IMAGE_SIZE,
)


def lee_sigma_speckle_filter(
    sar_band: np.ndarray, window_size: int = 5, looks: float = 4.0
) -> np.ndarray:
    """
    Lee-Sigma Adaptive Speckle Filter for Synthetic Aperture Radar (SAR) imagery.
    Suppresses multiplicative radar granular noise in homogeneous areas while
    preserving sharp backscatter edges along flood boundaries.

    Formula:
        I_hat = I_bar + W * (I - I_bar)
        W = var_x / (var_x + var_noise)
    """
    img = sar_band.astype(np.float64)
    mean_local = uniform_filter(img, size=window_size)
    sq_mean_local = uniform_filter(img ** 2, size=window_size)
    var_local = np.maximum(sq_mean_local - mean_local ** 2, 0.0)

    # Noise variance estimate under equivalent number of looks (ENL)
    sigma_v2 = 1.0 / max(looks, 1.0)
    var_noise = sigma_v2 * (mean_local ** 2 + 1e-6)

    weight = var_local / (var_local + var_noise + 1e-8)
    weight = np.clip(weight, 0.0, 1.0)
    filtered = mean_local + weight * (img - mean_local)
    return filtered.astype(np.float32)


def apply_clahe_channel(channel: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a [0, 1] channel."""
    ch_norm = np.clip(channel, 0.0, 1.0)
    ch_u8 = (ch_norm * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(ch_u8)
    return (enhanced.astype(np.float32)) / 255.0


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Computes Normalized Difference Water Index (McFeeters, 1996):
        NDWI = (Green - NIR) / (Green + NIR + 1e-6)
    Values > 0.15 strongly indicate surface water presence in pre-event baseline.
    """
    g = green.astype(np.float32)
    n = nir.astype(np.float32)
    ndwi = (g - n) / (g + n + 1e-6)
    return np.clip(ndwi, -1.0, 1.0)


def decode_uploaded_optical(file_bytes: bytes) -> np.ndarray:
    """
    Decodes an uploaded optical image (PNG/JPG/TIFF or .npy) into a [4, 256, 256] float32 array
    representing [Red, Green, Blue, NIR] normalized to [0, 1].
    """
    try:
        arr = np.load(io.BytesIO(file_bytes))
        if arr.ndim == 3 and arr.shape[0] == 4:
            return cv2.resize(arr.transpose(1, 2, 0), IMAGE_SIZE).transpose(2, 0, 1).astype(np.float32)
    except Exception:
        pass

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Synthesize realistic NIR channel from vegetation/water spectral response if 3-band RGB uploaded
    water_likelihood = np.clip((b + g) * 0.5 - r * 0.85, 0.0, 1.0)
    veg_likelihood = np.clip(g * 1.3 - (r + b) * 0.5, 0.0, 1.0)
    nir = np.clip(0.38 + 0.52 * veg_likelihood - 0.65 * water_likelihood, 0.02, 0.98)

    stacked = np.stack([r, g, b, nir], axis=0).astype(np.float32)
    return stacked


def decode_uploaded_sar(file_bytes: bytes) -> np.ndarray:
    """
    Decodes an uploaded SAR image (PNG/JPG/TIFF or .npy) into a [2, 256, 256] float32 array
    representing [VV, VH] normalized to [0, 1] (mapped from [-28 dB, 0 dB]).
    """
    try:
        arr = np.load(io.BytesIO(file_bytes))
        if arr.ndim == 3 and arr.shape[0] == 2:
            return cv2.resize(arr.transpose(1, 2, 0), IMAGE_SIZE).transpose(2, 0, 1).astype(np.float32)
    except Exception:
        pass

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0

    vv = np.clip(0.55 * rgb[:, :, 0] + 0.45 * rgb[:, :, 1], 0.0, 1.0)
    vh = np.clip(0.35 * rgb[:, :, 1] + 0.65 * rgb[:, :, 2] * 0.82, 0.0, 1.0)
    return np.stack([vv, vh], axis=0).astype(np.float32)


def preprocess_multimodal_pair(
    optical_4ch: np.ndarray, sar_2ch: np.ndarray, device: torch.device
) -> Dict[str, object]:
    """
    Executes full remote-sensing preprocessing pipeline:
    1. Resize/validate to 256x256
    2. CLAHE radiometric enhancement on optical bands
    3. Lee-Sigma speckle filtering on SAR VV and VH bands
    4. NDWI baseline calculation
    5. PyTorch tensor conversion
    """
    H, W = optical_4ch.shape[1], optical_4ch.shape[2]
    if sar_2ch.shape[1:] != (H, W):
        sar_2ch = (
            cv2.resize(sar_2ch.transpose(1, 2, 0), (W, H))
            .transpose(2, 0, 1)
            .astype(np.float32)
        )

    optical_clean = np.clip(optical_4ch, 0.0, 1.0).astype(np.float32)
    sar_raw = np.clip(sar_2ch, 0.0, 1.0).astype(np.float32)

    # Apply Lee-Sigma speckle filter on VV and VH polarizations
    vv_filtered = lee_sigma_speckle_filter(sar_raw[0], window_size=5, looks=4.0)
    vh_filtered = lee_sigma_speckle_filter(sar_raw[1], window_size=5, looks=4.0)
    sar_filtered = np.stack([vv_filtered, vh_filtered], axis=0)

    # Compute NDWI from Green (ch 1) and NIR (ch 3)
    ndwi = compute_ndwi(optical_clean[1], optical_clean[3])

    # Convert normalized SAR [0, 1] into physical decibels [-28 dB, -2 dB]
    vv_db = vv_filtered * 26.0 - 28.0
    vh_db = vh_filtered * 26.0 - 31.0

    # Prepare normalized PyTorch tensors [1, C, H, W]
    opt_tensor = torch.from_numpy(optical_clean).unsqueeze(0).to(device)
    sar_tensor = torch.from_numpy(sar_filtered).unsqueeze(0).to(device)

    return {
        "optical_clean": optical_clean,
        "sar_raw": sar_raw,
        "sar_filtered": sar_filtered,
        "ndwi": ndwi,
        "vv_db": vv_db,
        "vh_db": vh_db,
        "opt_tensor": opt_tensor,
        "sar_tensor": sar_tensor,
    }


def _array_to_base64_png(rgb_uint8: np.ndarray, upscale_to: Tuple[int, int] = (512, 512)) -> str:
    """Encodes an RGB uint8 numpy array as a crisp data:image/png;base64 string."""
    if rgb_uint8.shape[:2] != upscale_to:
        rgb_uint8 = cv2.resize(rgb_uint8, upscale_to, interpolation=cv2.INTER_CUBIC)
    pil_img = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def compute_high_risk_buffer_mask(
    seg_mask: np.ndarray, buffer_pixels: int = HIGH_RISK_BUFFER_PIXELS
) -> np.ndarray:
    """Computes a 100m (10-pixel) morphological dilation buffer zone around active flood inundation."""
    flood_binary = (seg_mask == CLASS_NEW_FLOOD).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (buffer_pixels * 2 + 1, buffer_pixels * 2 + 1)
    )
    dilated = cv2.dilate(flood_binary, kernel, iterations=1)
    buffer_zone = (dilated == 1) & (seg_mask == 0)
    return buffer_zone


def render_visual_layers(
    optical_4ch: np.ndarray,
    sar_filtered: np.ndarray,
    ndwi: np.ndarray,
    attention_map: np.ndarray,
    seg_mask: np.ndarray,
) -> Dict[str, str]:
    """
    Renders 5 high-contrast, publication-grade Base64 PNG visual layers (512x512):
    1. optical_rgb: Pre-Disaster Sentinel-2 True-Color RGB (CLAHE enhanced)
    2. sar_composite: Post-Disaster Sentinel-1 SAR False-Color Composite
    3. ndwi_heatmap: Hydrological NDWI Colormap
    4. attention_heatmap: Explainable AI (XAI) Attention Gate Activation Heatmap
    5. flood_overlay: Post-Event SAR + Semantic Flood Mask + 100m High-Risk Buffer Overlay
    """
    # 1. Pre-Disaster Optical True-Color RGB
    r = apply_clahe_channel(optical_4ch[0], clip_limit=1.8)
    g = apply_clahe_channel(optical_4ch[1], clip_limit=1.8)
    b = apply_clahe_channel(optical_4ch[2], clip_limit=1.8)
    optical_rgb_u8 = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

    # 2. Post-Disaster SAR False-Color Composite (Authentic Sentinel-1 Dual-Pol VV/VH Radar Aesthetic)
    vv = np.clip(sar_filtered[0], 0.0, 1.0)
    vh = np.clip(sar_filtered[1], 0.0, 1.0)
    # Specular water is dark (-18..-24 dB); urban corner reflectors (>0.78) glow bright cyan-white
    urban_bounce = np.clip((vv - 0.72) * 3.2, 0.0, 1.0)
    ratio = np.clip(vv / (vh + 0.10), 0.0, 1.0)
    sar_r = np.clip(vv * 0.72 + urban_bounce * 0.45, 0.0, 1.0)
    sar_g = np.clip(vv * 0.68 + vh * 0.32 + urban_bounce * 0.35, 0.0, 1.0)
    sar_b = np.clip(vh * 0.58 + ratio * 0.38 + urban_bounce * 0.25 + 0.05, 0.0, 1.0)
    sar_rgb_u8 = (np.stack([sar_r, sar_g, sar_b], axis=-1) * 255.0).astype(np.uint8)

    # 3. NDWI Hydrological Heatmap
    ndwi_norm = np.clip((ndwi + 0.65) / 1.45, 0.0, 1.0)
    ndwi_u8 = (ndwi_norm * 255.0).astype(np.uint8)
    ndwi_color_bgr = cv2.applyColorMap(ndwi_u8, cv2.COLORMAP_VIRIDIS)
    ndwi_color_rgb = cv2.cvtColor(ndwi_color_bgr, cv2.COLOR_BGR2RGB)

    # 4. XAI Attention Gate Heatmap (Inferno / Jet blended over optical structure)
    att_norm = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-6
    )
    att_u8 = (np.clip(att_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    att_bgr = cv2.applyColorMap(att_u8, cv2.COLORMAP_JET)
    att_rgb = cv2.cvtColor(att_bgr, cv2.COLOR_BGR2RGB)
    att_blended = cv2.addWeighted(att_rgb, 0.72, optical_rgb_u8, 0.28, 0)

    # 5. Semantic Segmentation Mask Overlay on Post-Disaster SAR
    buffer_mask = compute_high_risk_buffer_mask(seg_mask)
    overlay = sar_rgb_u8.copy().astype(np.float32)

    # Permanent Water (Class 1) -> Crisp Tactical Cyan (#06b6d4 -> RGB: 6, 182, 212)
    perm_idx = seg_mask == CLASS_PERMANENT_WATER
    overlay[perm_idx] = 0.28 * overlay[perm_idx] + 0.72 * np.array([6, 182, 212], dtype=np.float32)

    # New Flood Inundation (Class 2) -> High-Alert Crimson/Rose (#f43f5e -> RGB: 244, 63, 94)
    flood_idx = seg_mask == CLASS_NEW_FLOOD
    overlay[flood_idx] = 0.22 * overlay[flood_idx] + 0.78 * np.array([244, 63, 94], dtype=np.float32)

    # 100m High-Risk Buffer -> Tactical Amber (#f59e0b -> RGB: 245, 158, 11)
    overlay[buffer_mask] = 0.55 * overlay[buffer_mask] + 0.45 * np.array(
        [245, 158, 11], dtype=np.float32
    )

    # Draw crisp flood contour boundaries for visual clarity
    flood_u8 = flood_idx.astype(np.uint8) * 255
    contours, _ = cv2.findContours(flood_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.drawContours(overlay_u8, contours, -1, (255, 220, 230), 1)

    return {
        "optical_rgb": _array_to_base64_png(optical_rgb_u8),
        "sar_composite": _array_to_base64_png(sar_rgb_u8),
        "ndwi_heatmap": _array_to_base64_png(ndwi_color_rgb),
        "attention_heatmap": _array_to_base64_png(att_blended),
        "flood_overlay": _array_to_base64_png(overlay_u8),
    }


def compute_sar_backscatter_histogram(
    vv_db: np.ndarray, vh_db: np.ndarray, bins: int = 24
) -> List[Dict[str, float]]:
    """
    Computes Sentinel-1 SAR VV and VH backscatter distribution curves across [-27 dB, -3 dB]
    for plotting the -16 dB specular water cutoff threshold in Recharts.
    """
    bin_edges = np.linspace(-27.0, -3.0, bins + 1)
    vv_hist, _ = np.histogram(vv_db.ravel(), bins=bin_edges, density=True)
    vh_hist, _ = np.histogram(vh_db.ravel(), bins=bin_edges, density=True)

    series: List[Dict[str, float]] = []
    for i in range(bins):
        db_center = round(float(0.5 * (bin_edges[i] + bin_edges[i + 1])), 1)
        series.append(
            {
                "db": db_center,
                "vv_density": round(float(vv_hist[i] * 100.0), 2),
                "vh_density": round(float(vh_hist[i] * 100.0), 2),
                "water_threshold": -16.0,
            }
        )
    return series

