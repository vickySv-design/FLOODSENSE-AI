# FloodSense-AI: Genuine Test-Set Inference & Visualization Report
**Author:** Senior Geospatial AI Engineer & Deep Learning Researcher  
**Project:** FloodSense-AI (Final-Year B.E. CSE Capstone)  
**Execution Date:** November 08, 2026  
**Model Architecture:** `LightweightDualStreamAttentionUNet` (8,242,931 parameters)  
**Model Checkpoint:** `best_checkpoint.pth` (Trained for 30 epochs, Best Epoch: 28, Val mIoU: 0.7217)  
**Execution Environment:** Isolated CUDA Environment (`backend/.venv_cuda`) on CUDA (NVIDIA GeForce RTX 4070 Laptop GPU)  

---

## 1. Executive Summary

This report presents genuine multi-modal satellite flood segmentation inference on representative scenes from the official **Sen1Floods11 v1.1 untouched test partition** (90 chips). 

The trained `LightweightDualStreamAttentionUNet` was evaluated in pure inference mode (`model.eval()`, `torch.no_grad()`) without retraining or weight modifications. Preprocessing and radiometric normalizations strictly mirror the verified training pipeline:
- **Sentinel-2 Optical Stream:** 4 bands `[B4, B3, B2, B8]` via indices `[3, 2, 1, 7]`, scaled $\text{DN}/10000.0 \to [0.0, 1.0]$.
- **Sentinel-1 SAR Stream:** 2 polarizations `[VV, VH]` in dB, standardized via robust z-score clipping $[-3.0, 3.0]$.
- **Target Delineation:** 3 semantic classes (Class 0: Dry Land, Class 1: Permanent Water, Class 2: Water outside JRC reference) + Ignore Index 255.

---

## 2. Hardware Telemetry & Inference Throughput

| Benchmark Parameter | Measured Value | Operational Assessment |
|---|---|---|
| **Inference Hardware** | NVIDIA GeForce RTX 4070 Laptop GPU | 8 GB GDDR6 Dedicated VRAM |
| **Input Spatial Dimensions** | $512 \times 512$ Full Satellite Chip | Fully convolutional single-pass processing |
| **Input Tensor Shapes** | Optical: `[1, 4, 512, 512]` \| SAR: `[1, 2, 512, 512]` | `torch.float32` |
| **Output Logits Shape** | `[1, 3, 512, 512]` | 3-Class Probability Distributions |
| **Average Inference Latency** | **68.48 ms per 512x512 tile** | **~14.6 tiles per second** |
| **Peak GPU VRAM Allocated** | **~380 MB** | Extremely lightweight; fits easily in local memory |

---

## 3. Evaluated Test Chips Performance Breakdown

The inference pipeline evaluated 5 diverse test chips spanning Africa, Asia, and Europe:

| Chip Identifier | Country | Latency | Valid Pixels | Pixel Accuracy | Class 0 IoU (Dry) | Class 1 IoU (Perm) | Class 2 IoU (Flood Proxy) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `Mekong_1443339` | **Mekong** | 231.5 ms | 262,135 | **89.41%** | 0.8667 | 0.1869 | **0.8577** |
| `Spain_7387658` | **Spain** | 29.5 ms | 262,057 | **84.89%** | 0.7676 | 0.5367 | **0.7350** |
| `India_399883` | **India** | 27.7 ms | 262,119 | **78.97%** | 0.8307 | 0.0121 | **0.6365** |
| `Nigeria_812045` | **Nigeria** | 27.0 ms | 236,584 | **90.43%** | 0.9476 | 0.0014 | **0.8143** |
| `Ghana_866994` | **Ghana** | 26.6 ms | 117,511 | **43.90%** | 0.0209 | 0.5871 | **0.1462** |

---

## 4. Class Distribution Comparison: Ground Truth vs. Model Predictions

| Chip Identifier | Metric | Class 0 (Dry Land) | Class 1 (Permanent Water) | Class 2 (Water outside JRC) |
|---|---|:---:|:---:|:---:|
| `Mekong_1443339` | **Ground Truth** | 69,352 px | 22,065 px | 170,718 px |
| | **Prediction** | 69,596 px | 4,379 px | 188,160 px |
| `Spain_7387658` | **Ground Truth** | 109,051 px | 17,787 px | 135,219 px |
| | **Prediction** | 128,584 px | 10,231 px | 123,242 px |
| `India_399883` | **Ground Truth** | 125,827 px | 32,350 px | 103,942 px |
| | **Prediction** | 116,809 px | 1,183 px | 144,127 px |
| `Nigeria_812045` | **Ground Truth** | 117,001 px | 16,305 px | 103,278 px |
| | **Prediction** | 118,733 px | 31 px | 117,820 px |
| `Ghana_866994` | **Ground Truth** | 11,540 px | 58,979 px | 46,992 px |
| | **Prediction** | 40,502 px | 53,181 px | 23,828 px |

---

## 5. Visual Demonstrations & Asset Structure

For every test chip, a 5-panel visualization was exported alongside raw georeferenced GeoTIFFs:

```
backend/outputs/sen1floods11_inference/
├── visualizations/
│   ├── Mekong_1443339_inference.png
│   ├── Spain_7387658_inference.png
│   ├── India_399883_inference.png
│   ├── Nigeria_812045_inference.png
│   ├── Ghana_866994_inference.png
├── masks/
│   ├── Mekong_1443339_pred_mask.tif (EPSG:4326 GeoTIFF)
│   ├── Mekong_1443339_pred_mask.npy (Raw NumPy uint8)
│   ├── Mekong_1443339_confidence.tif (Float32 Softmax Probability)
│   └── Mekong_1443339_confidence.npy (Raw NumPy float32)
│   ├── Spain_7387658_pred_mask.tif (EPSG:4326 GeoTIFF)
│   ├── Spain_7387658_pred_mask.npy (Raw NumPy uint8)
│   ├── Spain_7387658_confidence.tif (Float32 Softmax Probability)
│   └── Spain_7387658_confidence.npy (Raw NumPy float32)
│   ├── India_399883_pred_mask.tif (EPSG:4326 GeoTIFF)
│   ├── India_399883_pred_mask.npy (Raw NumPy uint8)
│   ├── India_399883_confidence.tif (Float32 Softmax Probability)
│   └── India_399883_confidence.npy (Raw NumPy float32)
│   ├── Nigeria_812045_pred_mask.tif (EPSG:4326 GeoTIFF)
│   ├── Nigeria_812045_pred_mask.npy (Raw NumPy uint8)
│   ├── Nigeria_812045_confidence.tif (Float32 Softmax Probability)
│   └── Nigeria_812045_confidence.npy (Raw NumPy float32)
│   ├── Ghana_866994_pred_mask.tif (EPSG:4326 GeoTIFF)
│   ├── Ghana_866994_pred_mask.npy (Raw NumPy uint8)
│   ├── Ghana_866994_confidence.tif (Float32 Softmax Probability)
│   └── Ghana_866994_confidence.npy (Raw NumPy float32)
└── INFERENCE_REPORT.md
```

### Color Palette Legend
- **Beige / Khaki (`#d4b886`):** Class 0 (Dry Land)
- **Deep Ocean Blue (`#1f77b4`):** Class 1 (Permanent Water)
- **Crimson Red (`#d62728`):** Class 2 (Water outside JRC permanent reference)
- **Dark Charcoal (`#1a1a1a`):** Missing Satellite Observation / Orbit Swath Boundary (255)

---

## 6. Scientific Limitations & Operational Boundaries

1. **Class 2 Definition:** In Sen1Floods11, Class 2 denotes *water observed during the satellite acquisition pass that was absent from the JRC 1984–2018 permanent water reference*. While this functions as a dependable proxy for flood disaster inundation, it must not be claimed as an absolute hydrological proof of flash flood water, as seasonal agricultural rice paddies and seasonal wetlands can also produce Class 2 signatures.
2. **Atmospheric Correction Notice:** Sentinel-2 optical rasters in Sen1Floods11 provide Level-1C Top-Of-Atmosphere (TOA) Digital Numbers. Atmospheric haze and cloud shadows can induce slight spectral distortions that are mitigated by fusing with Sentinel-1 SAR.
3. **Local Georeferencing Preservation:** All exported `.tif` files preserve the native `EPSG:4326` geographic coordinates and pixel transforms, allowing direct drag-and-drop ingestion into QGIS, ArcGIS, or Leaflet web GIS viewers.
