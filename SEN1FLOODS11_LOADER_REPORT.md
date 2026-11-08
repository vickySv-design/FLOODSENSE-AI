# FloodSense-AI: Official Sen1Floods11 PyTorch Dataset Loader Report
**Author:** Senior Geospatial AI Engineer & Deep Learning Researcher  
**Project:** FloodSense-AI (Final-Year B.E. CSE Capstone)  
**Dataset:** Official Sen1Floods11 v1.1 Hand-Labeled Ground Truth (Bonafilia et al., IEEE CVPRW 2020)  
**Execution Environment:** Isolated CUDA Python 3.10.11 (`backend/.venv_cuda`) on NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)  
**Verification Status:** 10 / 10 Unit Tests Passed (100%)  
**Implementation Files:**
- [`backend/app/data/sen1floods11_dataset.py`](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/data/sen1floods11_dataset.py)
- [`backend/app/data/__init__.py`](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/data/__init__.py)
- [`test_sen1floods11_dataset.py`](file:///c:/Users/vicky/Desktop/Disaster%20Response/test_sen1floods11_dataset.py)
- [`backend/scripts/test_sen1floods11_dataset.py`](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/scripts/test_sen1floods11_dataset.py)

---

## 1. Executive Summary & Verification Scorecard

The official PyTorch dataset pipeline for Sen1Floods11 v1.1 has been implemented and verified. The loader enforces strict chip pairing across all four sensor and label modalities, Level-1C Top-Of-Atmosphere (TOA) Digital Number (DN) radiometric scaling, Sentinel-1 SAR dB normalization with NaN swath-margin mask propagation, and a hydrologically reconciled 3-class semantic target mapping.

### Loader Verification Scorecard

| Test ID | Verified Domain | Target Specification | Empirical Outcome | Status |
|---|---|---|---|:---:|
| **Test 1** | **Chip Pairing & Existence** | 1-to-1-to-1-to-1 pairing across all 4 rasters | All 252 train, 89 val, 90 test, 15 bolivia chips exist | **PASS** |
| **Test 2** | **Split Isolation & Disjointness** | Strict pairwise disjoint partitions | 0 overlapping chips across any pair of splits | **PASS** |
| **Test 3** | **Optical Selection & Scaling** | 4 bands `[B4, B3, B2, B8]` via `[3, 2, 1, 7]`, DN/10000 | Shape $[4, 256, 256]$, `float32`, bounded in $[0.0, 1.0]$ | **PASS** |
| **Test 4** | **SAR Channels & Normalization** | 2 bands `[VV, VH]` in dB, z-score standard clip | Shape $[2, 256, 256]$, `float32`, bounded in $[-3.0, 3.0]$ | **PASS** |
| **Test 5** | **SAR NaN Swath Margin Handling** | Swath margin NaNs $\to 0.0$ and masked to 255 | Zero NaNs/Infs in tensor, edge pixels marked 255 | **PASS** |
| **Test 6** | **3-Class Target Label Mapping** | Classes 0, 1, 2 + Ignore 255 | Target shape $[256, 256]$, `int64`, values $\subseteq \{0, 1, 2, 255\}$ | **PASS** |
| **Test 7** | **Patch & Full-Chip Generation** | Quadrants $256 \times 256$ & Full-chip $512 \times 512$ | 4 non-overlapping quadrants, exact spatial alignment | **PASS** |
| **Test 8** | **Pure NoData Patch Filtering** | Discard patches with $\ge 99\%$ NoData pixels | 1,008 raw training patches filtered down to 980 valid | **PASS** |
| **Test 9** | **DataLoader Batching** | Batch size 4, pinned memory, tensor stacking | Batch shapes: Opt $[4, 4, 256, 256]$, SAR $[4, 2, 256, 256]$ | **PASS** |
| **Test 10** | **CPU & CUDA Model Forward Pass** | End-to-end forward pass through Lightweight U-Net | Output shape $[2, 3, 256, 256]$, executed on RTX 4070 GPU | **PASS** |

---

## 2. Corrected Full-Dataset Class Accounting (116,916,224 Pixels)

In compliance with scientific audit standards, the 113,301 disagreement pixels ($L=0, J=1$) are assigned to Class 0 (Dry Land), following the empirical ground truth observation that human annotators confirmed dry conditions during the satellite acquisition pass. Class percentages are reported against both the **complete 116,916,224-pixel denominator** and the **100,983,314-pixel valid denominator**:

### Corrected 3-Class Census

| Target Class | Defining Condition | Total Pixels | % of Total (116.9M) | % of Valid (100.98M) | Scientific Semantic Definition |
|---|---|---|:---:|:---:|---|
| **Class 0: Dry Land** | $L=0$ (includes $L=0, J=0$ and $L=0, J=1$) | **90,277,709** | **77.2157%** | **89.3986%** | Non-water land surface confirmed dry by human ground truth (including seasonal dry lakebeds and reservoir drawdown). |
| **Class 1: Permanent Water** | $L=1 \land J=1$ | **3,062,228** | **2.6192%** | **3.0324%** | Water observed by LabelHand that aligns with historical permanent water bodies in JRC multidecadal reference. |
| **Class 2: Water Outside JRC Reference** | $L=1 \land J=0$ | **7,643,377** | **6.5375%** | **7.5689%** | **Water identified by LabelHand but not present in the JRC permanent-water reference.** (Used as flood proxy; not claimed as independently confirmed new inundation). |
| **Subtotal (Valid Targets)** | $L \in \{0, 1\}$ | **100,983,314** | **86.3724%** | **100.0000%** | **Valid target pixels available for loss computation.** |
| **NoData Mask (Ignore 255)** | $L = -1$ | **15,932,910** | **13.6276%** | — | Sensor swath margins, thick cloud obscuration, or unannotated pixels. Dynamically masked with zero gradient. |
| **TOTAL DATASET** | All pixels on disk | **116,916,224** | **100.0000%** | — | **100% of dataset accounted for.** |

---

## 3. Loader Architecture & Preprocessing Pipeline

### 3.1 Sentinel-2 Optical Stream
- **Source Bands:** 13-band Sentinel-2 Level-1C Top-Of-Atmosphere (TOA) Digital Numbers (DN).
- **Channel Selection:** Exact 0-based band indices `[3, 2, 1, 7]` corresponding to `[B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)]`.
- **Scaling Formulation:**
  $$\text{Optical}_{c} = \text{clip}\left(\frac{\text{DN}_{c}}{10000.0}, 0.0, 1.0\right) \in \mathbb{R}^{4 \times H \times W}$$
- **Data Type:** `torch.float32`.

### 3.2 Sentinel-1 SAR Stream
- **Source Bands:** Band 1 = `VV`, Band 2 = `VH` in decibels ($\text{dB} = 10 \cdot \log_{10} \sigma^0$).
- **Radiometric Normalization:** Robust z-score standardization with hard clipping to $[-3.0, 3.0]$:
  $$\text{VV}_{\text{norm}} = \text{clip}\left(\frac{\text{VV} - (-12.0)}{6.0}, -3.0, 3.0\right)$$
  $$\text{VH}_{\text{norm}} = \text{clip}\left(\frac{\text{VH} - (-19.0)}{5.0}, -3.0, 3.0\right)$$
- **Swath Margin NaN Handling:**
  Satellite orbit boundaries contain NaN values where the sensor scan does not cover the full $512 \times 512$ tile. The loader:
  1. Computes binary missing mask: $\text{Mask}_{\text{NaN}} = \text{isnan}(\text{VV}) \lor \text{isnan}(\text{VH})$.
  2. Replaces NaNs with $0.0$ in the SAR input tensor.
  3. Sets $\text{Target}[\text{Mask}_{\text{NaN}}] = 255$ (ignore index).
  > [!IMPORTANT]
  > This guarantees missing radar observations are **never silently interpreted as zero-backscatter water bodies**, and will never compute false loss gradients.

### 3.3 Patch Generation & Deterministic Tiling
- **Training Mode (`patch_size=256`):**
  Each $512 \times 512$ chip is partitioned into 4 non-overlapping quadrants of $256 \times 256$:
  - Quadrant 0: $(0:256, 0:256)$ [Top-Left]
  - Quadrant 1: $(0:256, 256:512)$ [Top-Right]
  - Quadrant 2: $(256:512, 0:256)$ [Bottom-Left]
  - Quadrant 3: $(256:512, 256:512)$ [Bottom-Right]
- **Pure NoData Filtering:** Patches containing $\ge 99\%$ NoData pixels are discarded during index construction. In the training split, 1,008 potential patches are filtered to **980 valid training patches**, preventing wasted GPU compute on empty margins.
- **Evaluation Mode (`patch_size=None` or `512`):**
  Yields complete $512 \times 512$ original rasters without tiling, preserving geospatial coordinates for direct inference evaluation.

---

## 4. Hardware Telemetry & Resource Consumption

Measurements conducted on physical hardware: Intel Core i7-14700HX + NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM):

| Parameter | Measured Value | Operational Assessment |
|---|---|---|
| **Batch Size** | 4 patches ($256 \times 256$) | Optimal for 8 GB VRAM |
| **Optical Batch Tensor Shape** | `torch.Size([4, 4, 256, 256])` | `torch.float32` |
| **SAR Batch Tensor Shape** | `torch.Size([4, 2, 256, 256])` | `torch.float32` |
| **Target Batch Tensor Shape** | `torch.Size([4, 256, 256])` | `torch.int64` |
| **Single Batch Disk Loading Latency** | **317.63 ms** (unpinned) / **~45 ms** (cached) | High-speed local SSD reading |
| **GPU Forward Pass Latency** | **17.90 ms** (FP16 AMP) | 223 patches/sec throughput |
| **Peak GPU VRAM Allocated** | **1,872.25 MB (1.83 GB)** | **22.8% of 8 GB VRAM** |
| **Remaining VRAM Headroom** | **6,319.75 MB (6.17 GB)** | **77.2% Headroom available** |

---

## 5. Scientific Limitations & Boundary Conditions

1. **Class 2 Epistemic Boundary:** Class 2 is strictly defined as *water detected by LabelHand that was absent from the JRC 1984–2018 permanent water reference*. While this serves as a reliable surrogate for inundation during a flood disaster, it cannot distinguish between flash flood water and seasonal ephemeral wetlands that fill during normal seasonal rains.
2. **Atmospheric Correction (Level-1C vs Level-2A):** The Sen1Floods11 optical rasters are provided as Level-1C TOA DNs rather than Level-2A Bottom-Of-Atmosphere (BOA) surface reflectance. Applying $\text{DN}/10000$ provides nominal TOA reflectance $[0.0, 1.0]$, but Rayleigh scattering and aerosol attenuation in the blue band ($B2$) remain uncorrected.
3. **Double-Bounce in Vegetated Floods:** Dense submerged canopy conditions can produce high backscatter return rather than specular drop-off. Ground truth validation in forested wetlands requires optical NIR confirmation.

---

## 6. Conclusion & Next Steps

The official Sen1Floods11 PyTorch Dataset Loader [`backend/app/data/sen1floods11_dataset.py`](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/data/sen1floods11_dataset.py) is verified and production-ready. All 10 unit tests have passed cleanly on both CPU and native CUDA GPU. The project is fully equipped to proceed to model training whenever instructed.

