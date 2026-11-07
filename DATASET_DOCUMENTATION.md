# FloodSense-AI: Satellite Dataset & Remote Sensing Data Provenance

**Document Reference:** `DATASET_DOCUMENTATION.md`  
**Project:** FloodSense-AI Capstone Platform  

---

## 1. Remote Sensing Sensor Overview

FloodSense-AI operates on multi-modal satellite data combining optical multispectral imagery and active microwave radar telemetry acquired by the European Space Agency (ESA) Copernicus constellation.

| Parameter | Sentinel-2 Multi-Spectral Instrument (MSI) | Sentinel-1 Synthetic Aperture Radar (C-SAR) |
| :--- | :--- | :--- |
| **Constellation** | Sentinel-2A / Sentinel-2B | Sentinel-1A / Sentinel-1B |
| **Orbit** | Sun-synchronous, 786 km altitude | Near-polar sun-synchronous, 693 km altitude |
| **Revisit Time** | 5 days (at equator) | 6 days (with 2-satellite constellation) |
| **Wavelength / Band** | Optical / VNIR / SWIR (443 nm – 2190 nm) | C-band microwave ($\lambda \approx 5.546\text{ cm}$, $f \approx 5.405\text{ GHz}$) |
| **Spatial Resolution** | 10 meters (B2, B3, B4, B8) | 10 meters (IW GRD pixel spacing) |
| **Cloud Penetration** | Zero (Blocked by clouds, haze, rain) | Complete (100% all-weather, day-and-night) |
| **Physical Interaction** | Surface solar reflectance ($\rho$) | Dielectric surface backscatter ($\sigma^0$, $\gamma^0$) |

---

## 2. Band Allocations & Spectral Indices

### 2.1 Optical Modality (Pre-Flood Baseline)
The optical tensor consumes 4 calibrated surface reflectance bands $[4, H, W]$:
- **Channel 0 (B04 - Red):** Central wavelength 665 nm. High absorption by liquid water, high reflectance by urban concrete.
- **Channel 1 (B03 - Green):** Central wavelength 560 nm. Peak reflectance for turbid and suspended sediment water bodies.
- **Channel 2 (B02 - Blue):** Central wavelength 490 nm. Deep water penetration, sensitive to coastal bathymetry.
- **Channel 3 (B08 - Near Infrared / NIR):** Central wavelength 842 nm. Near-total absorption by liquid water ($< 0.05$ reflectance), strong reflectance by healthy photosynthetic vegetation ($> 0.50$).

#### Normalized Difference Water Index (NDWI)
The physical water index formulated by McFeeters (1996) is implemented as:
$$\text{NDWI} = \frac{\rho_{B03} - \rho_{B08}}{\rho_{B03} + \rho_{B08}} = \frac{\rho_{\text{Green}} - \rho_{\text{NIR}}}{\rho_{\text{Green}} + \rho_{\text{NIR}}}$$
- Values $> +0.10$ indicate clear or turbid surface water.
- Values $< 0.00$ indicate terrestrial soil, dry asphalt, or dense canopy.

### 2.2 SAR Modality (Post-Flood Observation)
The radar tensor consumes 2 dual-polarization intensity channels $[2, H, W]$ in Interferometric Wide Swath (IW) Ground Range Detected (GRD) format:
- **Channel 0 (VV Polarization):** Co-polarization (Vertical transmit, Vertical receive). Highly sensitive to surface roughness, capillary water waves, and calm specular reflection.
- **Channel 1 (VH Polarization):** Cross-polarization (Vertical transmit, Horizontal receive). Highly sensitive to volume scattering from flooded vegetation and tree trunks (double-bounce effect).

#### Radar Decibel Calibration
Digital numbers ($DN$) are radiometrically calibrated to backscatter coefficient $\sigma^0$ or terrain-flattened $\gamma^0$:
$$\sigma^0\text{ (dB)} = 10 \cdot \log_{10}(DN^2 + \epsilon)$$
- Smooth open water causes specular reflection away from the radar antenna, dropping backscatter below $-16.0\text{ dB}$ (typically $-18\text{ dB}$ to $-24\text{ dB}$).
- Rough dry urban ground exhibits diffuse backscatter between $-8.0\text{ dB}$ and $-2.0\text{ dB}$.

---

## 3. Semantic Segmentation Classes

Every pixel in the output mask is assigned one of three mutually exclusive classes:
- **Class 0: Dry Land:** Terrestrial soil, paved urban roads, buildings, unflooded farmland.
- **Class 1: Permanent Water Baseline:** Pre-existing perennial rivers, canals, lakes, and reservoirs identified from pre-event Sentinel-2 NDWI.
- **Class 2: New Flood Inundation:** Active emergency flood expansion, overtopped riverbanks, submerged city streets, and inundated residential compounds.

---

## 4. Benchmark Dataset Splits (Sen1Floods11)

The offline training and evaluation suite is calibrated against the **Sen1Floods11** geospatial benchmark dataset (Bonafilia et al., IEEE CVPRW):
- **Image Dimensions:** $256 \times 256$ pixels.
- **Metric Resolution:** $10.0\text{ m} \times 10.0\text{ m}$ per pixel ($0.01\text{ ha}$, $100\text{ m}^2$).
- **Bounding Box Coverage:** $2.56\text{ km} \times 2.56\text{ km}$ per standard tile ($6.55\text{ km}^2$).
- **Mini-Dataset Distribution:**
  - Training Split: 18 tiles (located in `backend/data/mini_sen1floods11/train/`)
  - Validation Split: 6 held-out evaluation tiles (`val_tile_01` to `val_tile_06` in `backend/data/mini_sen1floods11/val/`)
- **Disaster Presets:**
  - `chennai_monsoon_2026`: Coastal urban basin (Adyar & Cooum estuaries, Tamil Nadu)
  - `assam_brahmaputra_surge`: Braided alluvial floodplain (Kaziranga sector, Assam)
  - `wayanad_kerala_flashflood`: Mountainous high-relief catchment (Meppadi sector, Kerala)
