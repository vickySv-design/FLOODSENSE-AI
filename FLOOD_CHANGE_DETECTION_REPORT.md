# FloodSense-AI: Forensic Audit & Reconciliation of Sentinel-1 Temporal Change Detection

**Author:** Geospatial AI & Microwave Remote Sensing Specialist  
**Project:** FloodSense-AI (Final-Year B.E. Computer Science and Engineering Capstone)  
**Execution Date:** November 8, 2026  
**Primary Event:** Bolivia February 2018 Mamore and Beni River Catastrophic Flood  
**Target Chip:** `Bolivia_103757` (Department of Beni, Bolivian Amazon Basin)  
**Pre-Event Pass:** 2018-02-03 09:57:45 UTC (Sentinel-1B, Relative Orbit 156, Descending)  
**Event-Time Pass:** 2018-02-15 09:57:45 UTC (Sentinel-1B, Relative Orbit 156, Descending)  
**Repeat Cycle Interval:** Exactly 12 Days (Identical Orbital Geometry & Look Angles)  

---

## 1. Forensic Audit: Pixel Count Discrepancy & Evaluation Domain Reconciliation

### Root Cause of Previous Discrepancy
In Phase 4C, the reported changed-pixel count differed between the confusion matrix and the area section:
- Confusion Matrix: **37,525 changed pixels** (TP + FP = 29,249 + 8,276).
- Area Quantification: **45,788 changed pixels** (441.20 ha / 4.41 km²).

This discrepancy is entirely resolved by decomposing the raster into its explicit evaluation domains:

| Evaluation Domain | Description | Total Pixels | Changed Pixels ($\Delta \text{VV} < -5.0\text{ dB}$) | Geodesic Area |
|---|---|:---:|:---:|:---:|
| **Domain A: Full SAR Swath** (`valid_sar`) | All mutual non-NaN Sentinel-1 pixels across both orbital passes | **101,623** | **45,788** | **441.20 ha (4.4120 km²)** |
| **Domain C: Evaluation Domain** (`eval_mask`) | Intersection of SAR swath and human-annotated ground truth (`LabelHand != -1`) | **88,077** | **37,525** | **361.58 ha (3.6158 km²)** |
| **Region D: Annotator Excluded Region** | Valid SAR observations where annotators marked NoData (`LabelHand == -1`) | **13,546** | **8,263** | **79.62 ha (0.7962 km²)** |
| **Out-of-Swath Tile Boundary** | Border regions outside the radar swath footprint | **160,521** | 0 | 1,546.76 ha |
| **Total Grid Footprint** | Complete 512 x 512 tile extent | **262,144** | — | 2,525.98 ha (25.26 km²) |

#### Mathematical Reconciliation Check:
$$\text{Domain A Total Pixels} = \text{Domain C} + \text{Region D} = 88,077 + 13,546 = 101,623 \quad \text{(Exact)}$$
$$\text{Domain A Changed Pixels} = \text{Domain C (TP + FP)} + \text{Region D Changed} = 37,525 + 8,263 = 45,788 \quad \text{(Exact)}$$
$$\text{Domain A Changed Area} = 361.58\text{ ha} + 79.62\text{ ha} = 441.20\text{ ha} \quad \text{(Exact)}$$

The confusion matrix correctly evaluated only the **88,077 pixels** where ground truth annotations exist, whereas the physical raster mask covers the full **101,623-pixel SAR observation swath**.

---

## 2. Spatial Co-Registration & Feature Alignment Audit

The previous report claimed that a cross-correlation peak near zero proved "sub-pixel co-registration". A scientific audit refines this finding:

- **Zero-Shift Cross-Correlation ($dx=0, dy=0$):** **$r = 0.2577$**
- **Optimal Correlation Peak:** $(dy=0, dx=-1)$ with $r = 0.2583$
- **Correlation Difference:** $|0.2583 - 0.2577| = 0.0006$ (< 0.25% relative difference).
- **Scientific Limitation:**
  1. The linear cross-correlation surface over temporal SAR imagery is inherently flat ($r \sim 0.26$) because single-look speckle noise and 12 days of temporal decorrelation (monsoonal soil wetting from $-1.98\text{ dB}$, vegetation phenology) reduce phase and amplitude coherence.
  2. While both rasters share an exact geospatial coordinate grid (identical CRS `EPSG:4326`, identical affine transform, and identical WGS84 corners), **cross-correlation on temporal SAR imagery does not prove sub-pixel co-registration**.
  3. Standard Sentinel-1 GRD terrain-correction residual uncertainty remains approximately 0.5 to 1.0 pixel (~5 to 10 meters). The data support **grid alignment with residual sub-pixel/1-pixel registration uncertainty**, not verified sub-pixel feature lock.

---

## 3. Threshold Calibration & Validation Independence Status

### Critical Methodological Disclosure:
The $\Delta \text{VV}$ threshold ($T = -5.0\text{ dB}$) and the composite multi-condition rule were **calibrated using the reference labels of chip `Bolivia_103757`**.

Therefore:
1. **Exploratory Status:** These metrics represent **exploratory, chip-specific calibration results**. They must NOT be cited as independent out-of-sample validation metrics.
2. **No Universal Threshold Claim:** A threshold of $-5.0\text{ dB}$ is not established as a universally valid flood threshold for unseen global basins.
3. **Proposed Independent Validation Protocol:**
   To establish genuine out-of-sample validation, the calibrated $-5.0\text{ dB}$ threshold must be evaluated on the remaining 14 Bolivia holdout chips (e.g. `Bolivia_195474`, `Bolivia_242570`, `Bolivia_290290`, `Bolivia_314919`) **without prior threshold re-tuning**.

---

## 4. Mathematically Verified Metrics on Evaluation Domain (`eval_mask`, $N = 88,077$)

Binary Classification Target: Positive = Potential Water outside JRC (Class 2), Negative = Dry Land (Class 0).

$$\text{Ground Truth Positives (Class 2)} = 35,362 \quad \text{(40.15\%)}\\
\text{Ground Truth Negatives (Class 0)} = 52,715 \quad \text{(59.85\%)}\\
\text{Permanent Water (Class 1)} = 0 \quad \text{(0.00\%)}$$

### Primary Baseline ($\Delta \text{VV} < -5.0\text{ dB}$):
- **True Positives (TP):** **29,249**
- **False Positives (FP):** **8,276**
- **False Negatives (FN):** **6,113**
- **True Negatives (TN):** **44,439**
- **Predicted Positives (TP + FP):** **37,525**
- **Predicted Negatives (TN + FN):** **50,552**
- **Total Evaluated (TP + FP + FN + TN):** **88,077**

#### Derived Verification Scores:
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{29,249}{37,525} = \mathbf{0.7795}$
- **Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{29,249}{35,362} = \mathbf{0.8271}$
- **Intersection-over-Union (IoU):** $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{29,249}{43,638} = \mathbf{0.6703}$
- **F1-Score:** $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{0.8026}$
- **Overall Accuracy:** $\frac{\text{TP} + \text{TN}}{88,077} = \frac{73,688}{88,077} = \mathbf{83.66\%}$

---

## 5. Geodesic Area Calculation & Pixel Area Variation

Calculated using per-pixel geodesic integration across the WGS 84 ellipsoid ($a = 6,378,137\text{ m}, f = 1/298.257$):
- **Northern Boundary Latitude ($-14.2121^\circ$):** Pixel area = **96.3682 m²** ($dx=9.6959\text{m}, dy=9.9391\text{m}$).
- **Southern Boundary Latitude ($-14.2580^\circ$):** Pixel area = **96.3491 m²** ($dx=9.6939\text{m}, dy=9.9391\text{m}$).
- **Latitudinal Area Variation:** Only **$-0.0191\text{ m²}$** ($-0.0198\%$) across the 5.12 km tile.
- **Mean Pixel Area:** **96.3586 m²**.

### Exact Integrated Surface Areas:
- **Total 512 x 512 Tile Area:** **25.2598 km²**
- **Domain A [Full SAR Swath Area]:** **9.7922 km²** (101,623 pixels)
  - **Estimated Changed Water Area ($\Delta \text{VV} < -5.0\text{ dB}$):** **441.20 hectares (4.4120 km²)** (45,788 pixels)
- **Domain C [Evaluation Domain Area]:** **8.4868 km²** (88,077 pixels)
  - **Evaluated Changed Area:** **361.58 hectares (3.6158 km²)** (37,525 pixels)
- **Region D [Annotator Excluded Area]:** **79.62 hectares (0.7962 km²)** (8,263 pixels)

*(Label: Experimental estimated changed-water area, not a confirmed flood-damaged area).*

---

## 6. Discrepancy & Correction Summary Table

| Issue / Metric | Previous Phase 4C Report | Reconciled Phase 4D Audit | Forensic Resolution |
|---|---|---|---|
| **Changed Pixels ($\Delta \text{VV} < -5\text{ dB}$)** | Ambiguous: reported 37,525 in confusion matrix, 45,788 in area | Reconciled: **37,525** in Domain C (Eval Domain), **45,788** in Domain A (Full SAR Swath) | Explicit separation of Evaluation Domain (88,077 px) vs Full SAR Swath (101,623 px). Region D holds 8,263 changed pixels. |
| **Excluded Pixels (13,546)** | Unexplained gap between 101,623 and 88,077 | Documented: 13,546 pixels have valid SAR data but were marked NoData (`LabelHand = -1`) by annotators | Evaluated only where reference ground truth exists. |
| **Spatial Alignment** | Claimed "sub-pixel alignment proved" | Refined: **Grid-aligned with residual sub-pixel/1-pixel registration uncertainty** | Linear correlation surface is flat ($r=0.258$); temporal speckle decorrelation prevents claiming mathematical proof of sub-pixel feature lock. |
| **Threshold Independence** | Presented $-5.0\text{ dB}$ without clear calibration caveat | Declared: **Exploratory, chip-specific calibration result** | Clearly distinguished from independent test metrics. Proposed cross-validation on remaining 14 Bolivia chips. |
| **Geodesic Pixel Area** | Scalar center-latitude approximation | **Per-pixel row-wise WGS84 integration** | North-South variation ($0.02\%$) accounted for in exact integrated surface area. |

---

## 7. Deliverables Staged

```
backend/outputs/flood_event_analysis/
├── delta_VV.tif (Georeferenced Float32 GeoTIFF, EPSG:4326)
├── delta_VH.tif (Georeferenced Float32 GeoTIFF, EPSG:4326)
├── potential_inundation_mask.tif (Binary GeoTIFF: 0=Dry, 1=Potential Water, 255=NoData)
├── flood_change_visualization.png (8-Panel Scientific Assessment Figure)
├── threshold_sensitivity.csv (Reconciled Evaluation Domain Metrics)
├── change_detection_metrics.json (Fully Reconciled Machine-Readable Output)
└── FLOOD_CHANGE_DETECTION_REPORT.md (Comprehensive Reconciled Report)
```
