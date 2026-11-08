# FloodSense-AI: Pre-Event Sentinel-1 Data Acquisition & Verification Report

**Author:** Geospatial AI & Remote Sensing Specialist  
**Project:** FloodSense-AI (Final-Year B.E. Computer Science and Engineering Capstone)  
**Execution Date:** November 8, 2026  
**Primary Candidate Event:** Bolivia February 2018 Mamore/Beni River Flooding (Amazon Floodplain)  
**Reference Chip:** `Bolivia_103757`  
**Pre-Event Acquisition Timestamp:** **2018-02-03 09:57:45 UTC** (12 days prior repeat cycle)  
**Event-Time Acquisition Timestamp:** **2018-02-15 09:57:45 UTC**  
**Sensor & Mode:** Sentinel-1B C-SAR, IW Mode, GRD, Descending, Relative Orbit Track 156  

---

## 1. Scene Verification & Provenance

- **Earth Engine Asset ID:** `COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20180203T095745_20180203T095810_009457_01101A_C4EF`
- **Collection:** `COPERNICUS/S1_GRD`
- **GCP Project:** `floodsense-ai-514105`
- **Polarizations Ingested:** `VV` (Band 1) and `VH` (Band 2)
- **Local Download Path:** `backend/data/flood_event/before\Bolivia_103757_S1_PreEvent_20180203.tif`
- **Clipped File Size:** **2052.0 KB**
- **Full-Scene Size Avoided:** **~1,727.8 MB** (99.88% bandwidth reduction via direct tile extraction)

---

## 2. Spatial Compatibility & Geometric Alignment

| Spatial Attribute | Event-Time Observation (`2018-02-15`) | Pre-Event Observation (`2018-02-03`) | Match Status |
|---|---|---|:---:|
| **Coordinate Reference System** | `EPSG:4326` (WGS 84) | `EPSG:4326` (WGS 84) | **EXACT MATCH** |
| **Spatial Dimensions** | 512 x 512 pixels | 512 x 512 pixels | **EXACT MATCH** |
| **Pixel Resolution** | ~10m Ground Sample Distance | ~10m Ground Sample Distance | **EXACT MATCH** |
| **Affine Transform** | `Affine(8.983e-5, 0, -65.265, 0, -8.983e-5, -14.212)` | `Affine(8.983e-5, 0, -65.265, 0, -8.983e-5, -14.212)` | **EXACT MATCH** |
| **Geographic Bounds** | `[-65.265121, -14.258060, -65.219127, -14.212066]` | `[-65.265121, -14.258060, -65.219127, -14.212066]` | **EXACT MATCH** |
| **Band Count & Order** | 2 Bands (`[VV, VH]`) | 2 Bands (`[VV, VH]`) | **EXACT MATCH** |
| **NoData Encoding** | `NaN` (Float32) | `NaN` (Float32) | **EXACT MATCH** |

---

## 3. Radiometric Compatibility & Physical Verification

Both rasters represent log-scaled radar backscatter ($\sigma^0$) in **decibels (dB)** ($10 \log_10 \sigma^0$):

| Radiometric Metric | Pre-Flood Baseline (`2018-02-03`) | Event-Time Flood (`2018-02-15`) | Inundation Delta ($\Delta = \text{Post} - \text{Pre}$) |
|---|:---:|:---:|:---:|
| **VV Minimum** | -25.45 dB | -40.82 dB | — |
| **VV Maximum** | 0.58 dB | -0.69 dB | — |
| **VV Mean Backscatter** | **-8.37 dB** | **-13.01 dB** | **-4.64 dB** |
| **VH Mean Backscatter** | **-14.15 dB** | **-19.29 dB** | **-5.14 dB** |

### Physical Interpretation:
The **4.64 dB drop** in mean VV backscatter between February 3 and February 15 represents the classic physical radar signature of open water inundation: smooth water surfaces induce specular forward reflection away from the radar antenna, causing a dramatic drop in received backscatter power.

---

## 4. Deliverables Staged

- Pre-Event GeoTIFF: `backend/data/flood_event/before\Bolivia_103757_S1_PreEvent_20180203.tif`
- Comparative Panel Plot: `backend/data/flood_event/before\Bolivia_103757_pre_vs_post_comparison.png`
- Provenance Metadata: `backend/outputs/flood_event_analysis/pre_event_metadata.json` and `pre_event_metadata.json`

**Conclusion:** **Phase 4B Pre-Event Data Acquisition and Compatibility Verification PASSED.**
