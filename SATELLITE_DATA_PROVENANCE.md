# FloodSense-AI: Satellite Data Provenance Framework
## Verifiable Scientific Pedigree & Audit Trail for Multi-Modal Remote Sensing

[![Standard](https://img.shields.io/badge/Standard-OGC%20Data%20Provenance-2563eb.svg)](#)
[![Format](https://img.shields.io/badge/Format-JSON%20Manifest%20v2.1-10b981.svg)](#)
[![Integrity](https://img.shields.io/badge/Verification-Immutable%20Audit%20Trail-06b6d4.svg)](#)

---

## 1. Provenance Philosophy & Academic Compliance

In disaster response and remote sensing research, black-box data substitution undermines operational confidence. FloodSense-AI enforces **strict, end-to-end provenance recording** for every satellite acquisition, preprocessing transformation, neural network inference run, and tactical evacuation plan.

Under no circumstances is a synthetic dataset described as a genuine satellite observation, nor are unauthenticated mock responses masked as live data.

---

## 2. Immutable Provenance Schema

Every analysis session generates a unique, timestamped JSON manifest stored in [`backend/data/metadata/`](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/data/metadata/) adhering to the following schema:

```json
{
  "analysis_id": "024d3597",
  "recorded_at": "2026-11-08T05:24:50.123456Z",
  "provenance_tag": "GENUINE_CACHED_ACQUISITION",
  "aoi": {
    "name": "Chennai Test Sector",
    "bbox": [13.01, 80.22, 13.06, 80.27],
    "area_km2": 25.0,
    "utm_crs": "EPSG:32644",
    "terrain_type": "coastal_urban"
  },
  "provider": {
    "name": "copernicus",
    "endpoint": "Copernicus OData / STAC",
    "zero_cost_academic": true
  },
  "satellite_acquisitions": {
    "pre_disaster_optical": {
      "platform": "SENTINEL-2 MSI",
      "product_level": "Level-2A (Bottom-of-Atmosphere)",
      "bands": ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"],
      "resolution_m": 10.0,
      "scene_details": {
        "scene_id": "CDSE_SENTINEL-2_20251101_13.01_80.22",
        "cloud_cover_pct": 4.8,
        "acquisition_date": "2025-11-01T10:30:00Z"
      }
    },
    "post_disaster_sar": {
      "platform": "SENTINEL-1 C-SAR",
      "product_level": "Level-1 GRD (Ground Range Detected)",
      "polarizations": ["VV", "VH"],
      "instrument_mode": "IW (Interferometric Wide Swath)",
      "frequency_ghz": 5.405,
      "resolution_m": 10.0,
      "scene_details": {
        "scene_id": "CDSE_SENTINEL-1_20251201_13.01_80.22",
        "acquisition_date": "2025-12-01T10:30:00Z"
      }
    }
  },
  "preprocessing_pipeline": [
    "Optical CLAHE contrast adjustment (clip=2.0)",
    "Optical NDWI McFeeters 1996 baseline calculation",
    "SAR Sigma-0 backscatter decibel calibration [-30, 0] dB",
    "SAR Lee-Sigma 5x5 adaptive speckle suppression",
    "Bicubic resampling to common UTM metric grid"
  ],
  "deep_learning_pipeline": {
    "architecture": "Dual-Stream Siamese Attention U-Net with CMAF",
    "checkpoint": "DualStreamAttentionUNet_v2.1",
    "input_tensors": { "optical": "[B, 4, H, W]", "sar": "[B, 2, H, W]" },
    "semantic_classes": {
      "0": "Dry Land",
      "1": "Permanent Baseline Water (NDWI/Pre-SAR)",
      "2": "Newly Inundated Floodwater (Post-SAR Inundation)"
    },
    "inference_engine": "PyTorch Sliding-Window with 2D Hann Cosine Overlap Blending",
    "execution_hardware": "CPU Engine (PyTorch Optimized)"
  },
  "raster_validation": {
    "optical": {
      "B04": { "format": "GeoTIFF", "integrity_status": "VERIFIED", "crs": "EPSG:4326", "non_zero_pct": 100.0 }
    },
    "sar": {
      "VV": { "format": "GeoTIFF", "integrity_status": "VERIFIED", "crs": "EPSG:4326", "non_zero_pct": 100.0 }
    }
  },
  "output_products": {
    "optical_bands": { "B02": "...", "B03": "...", "B04": "...", "B08": "..." },
    "sar_bands": { "VV": "...", "VH": "..." }
  },
  "scientific_notes": "Full geospatial change detection pipeline executed without artificial metric clamping."
}
```

---

## 3. Provenance Classification Tags

| Provenance Tag | Criteria | Operational Meaning |
|---|---|---|
| `GENUINE_LIVE_ACQUISITION` | Products downloaded directly from live CDSE / GEE endpoints with validated auth tokens. | Live disaster observation from orbiting satellites. |
| `GENUINE_CACHED_ACQUISITION` | Products loaded from local verified GeoTIFF cache downloaded during prior mission runs. | 100% genuine satellite data, replayed offline. |
| `PRESET_CALIBRATED_BENCHMARK` | Pre-calibrated demonstration scenarios (e.g. Chennai 2026, Assam 2024, Wayanad 2024). | Calibrated case studies with ground-truth validation. |

---

## 4. Frontend & SitRep Integration

1. **Frontend Telemetry:** The active analysis session exposes provenance details directly via `GET /api/data/provenance/{analysis_id}`.
2. **NDRF Situation Report:** The automated AI SitRep generator embeds the sensor pedigree, product identifiers, and processing flags into the formal tactical flash bulletin header.
3. **GeoJSON Metadata:** Exported GeoJSON FeatureCollections (`roads_geojson`, `blocked_roads_geojson`, `rescue_routes_geojson`) include the complete provenance manifest within their root `metadata` block.
