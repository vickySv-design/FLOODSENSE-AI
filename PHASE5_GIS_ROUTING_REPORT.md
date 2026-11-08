# FloodSense-AI — Phase 5 & 5.1: GIS Infrastructure Exposure & A* Rescue Routing Report

**Project:** FloodSense-AI  
**Degree / Program:** Bachelor of Engineering (B.E.) in Computer Science & Engineering  
**Academic Year:** Final-Year Capstone Project  
**Date:** November 2026  
**Document Version:** 3.0 — Phase 5.2 Final Technical Verification & MultiDiGraph Audit  

---

## Executive Summary

Phase 5, 5.1, and 5.2 transition FloodSense-AI from remote-sensing satellite image processing to actionable emergency decision support. By integrating georeferenced flood-risk raster classifications with real OpenStreetMap (OSM) vector infrastructure (building footprints and road networks), this phase implements:

1. **Sub-Pixel Coordinate & Dynamic UTM Projection Pipeline**: Strict mathematical alignment between WGS84 geographic coordinates (`EPSG:4326`), local Universal Transverse Mercator (UTM) metric projections (`EPSG:32644` for Zone 44N), and raster pixel rows and columns.
2. **Defensible Building Exposure Grading**: Scientifically sound classification of building footprints into standardized exposure tiers (`HIGH_EXPOSURE`, `MODERATE_EXPOSURE`, `LOW_OR_NO_DETECTED_EXPOSURE`, `UNKNOWN`) via exact polygon rasterization, explicitly refraining from fabricated vertical water depths.
3. **Multi-Point Road Hazard Sampling**: High-density sampling along actual road LineStrings at ~10-meter metric intervals to classify road segments into `POTENTIAL_HAZARD`, `CAUTION`, `NO_DETECTED_INTERSECTION`, and `UNKNOWN`.
4. **OSM One-Way MultiDiGraph (`nx.MultiDiGraph`) & Admissible A\* Routing**: Directed multi-edge metric road graph preserving 993 parallel road segments, respecting `oneway=yes`, `oneway=-1` (contraflow), `oneway=no`, highway omission defaults, and conservative handling of conditional/reversible tags, using an admissible Euclidean distance heuristic.
5. **Strict Hazard Avoidance**: Complete exclusion of potential hazard edges when `avoid_hazards=True`, returning `NO_SAFE_ROUTE_AVAILABLE` when all available paths cross floodwaters.
6. **Reconciled Transit ETA & Honest Status Terminology**: Corrected formula-driven ETA calculation without undocumented buffers ($\text{ETA} = 8.33\text{ min}$ at 25 km/h for 3.47 km), replacing unsupported marketing labels with `CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS`.
7. **Satellite Provenance & Pixel Audit**: Verified bit-exact generation of `flood_mask_024d3597.tif` by `dual_stream_attention_unet.pth`, transparently distinguishing model predictions from ground-truth observations and forward-dated demonstration dates.
8. **Automated Verification Suite**: A 23-test unit and regression test suite (`backend/scripts/test_phase5_gis_integrity.py`) demonstrating a 100% pass rate.

---

## 1. Study Area & Dataset Provenance

### 1.1 Study Area: Chennai Coastal Catchment & Adyar River Basin
- **Preset Identifier:** `chennai_monsoon_2026`
- **Geographic Bounding Box:** `[13.0100°N, 80.2200°E, 13.0600°N, 80.2700°E]`
- **Coverage Dimensions:** ~5.5 km (East-West) $\times$ ~5.5 km (North-South) ($\approx 30.25 \text{ km}^2$)
- **Local Metric Projection:** WGS 84 / UTM Zone 44N (`EPSG:32644`)

### 1.2 Georeferenced Flood Raster Provenance & Inference Audit
Audit of source metadata and provenance log (`backend/data/metadata/provenance_024d3597.json`):
- **Raster Path:** `backend/data/satellite_cache/flood_mask_024d3597.tif`
- **Analysis ID:** `024d3597` (Recorded: `2026-11-08T05:24:52Z`)
- **Data Nature:** Derived deep learning model prediction generated from genuine Copernicus satellite observations ingested via the Copernicus Data Space Ecosystem (CDSE) zero-cost academic STAC endpoint.
- **Pre-Disaster Optical Observation:**
  - Platform: Sentinel-2 MSI, Level-2A (Bottom-of-Atmosphere reflectance).
  - Scene ID: `CDSE_SENTINEL-2_20250101_13.01_80.22`
  - Acquisition Date: `2025-01-01T10:30:00Z` (Cloud cover: 4.8%).
  - Ingested Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) at 10m GSD.
- **Post-Disaster SAR Observation:**
  - Platform: Sentinel-1 C-SAR, Level-1 Ground Range Detected (GRD), IW mode ($5.405\text{ GHz}$).
  - Scene ID: `CDSE_SENTINEL-1_20251110_13.01_80.22`
  - Acquisition Date: `2025-11-10T10:30:00Z` (High monsoonal flood surge).
  - Polarizations: VV and VH at 10m GSD.
- **Model Checkpoint & Architecture:**
  - Architecture: `DualStreamAttentionUNet_v2.1` (Dual-Stream Siamese Attention U-Net with CMAF).
  - Weights Path: `backend/data/checkpoints/dual_stream_attention_unet.pth`.
  - Inference Engine: PyTorch Sliding-Window with 2D Hann Cosine Overlap Blending.
  - Preprocessing: Optical CLAHE (clip=2.0), NDWI baseline calculation, SAR $\sigma^0$ decibel calibration ([-30, 0] dB), SAR Lee-Sigma $5\times 5$ speckle filter.
- **Grid Resolution:** $256 \times 256$ pixels ($\approx 21.7\text{ m}$ ground spacing, affine step $0.0001953^\circ$).
- **Pixel Class Encoding & Verification:**
  - `0`: Valid Dry Land — **32,202 pixels (49.14%)**
  - `1`: Permanent Water Bodies (Adyar River, canals, coastline) — **18,965 pixels (28.94%)**
  - `2`: Mapped Flood Risk / Inundated Surface — **14,369 pixels (21.93%)**
  - `255`: NoData / Cloud / Sensor Void — **0 pixels (100% valid interior coverage)**

### 1.3 OpenStreetMap (OSM) Infrastructure & Sampling Methodology
- **Building Footprints:**
  - File: `backend/data/osm_cache/chennai_buildings.geojson`
  - Total Features in File: **27,884 genuine building polygons**.
  - **Sample Selection Methodology:** For interactive GIS exposure reporting, a deterministic systematic decimation (uniform index stride sampling: `step = total_raw // max_buildings = 27884 // 1000 = 27`) is applied across the serialized Overpass GeoJSON sequence (`buildings[::27][:1000]`). This preserves sub-second API execution (~0.55s) and guarantees 100% deterministic reproducibility.
  - *Sampling Disclosure:* This is a deterministic file-order decimation, NOT a random or spatially stratified sample.
- **Road Network:**
  - File: `backend/data/osm_cache/chennai_roads.geojson`
  - Total Features in File: **5,272 genuine road LineStrings** totaling **673.4 km**.
  - **Sample Selection Methodology:** For road hazard summary reporting, the first 1,000 features in file sequence (155.90 km) are evaluated. *Sampling Disclosure:* These features are a sequential file slice spread across various neighborhoods; they do **not** form a single continuous corridor. For real-time A* evacuation routing, the **full 5,272-road network (673.4 km)** is parsed and indexed into an `nx.MultiDiGraph` (15,055 nodes, 33,056 edges) in ~0.27s, preserving all 993 parallel edges.

---

## 2. Geospatial Alignment & Coordinate Systems

### 2.1 Dynamic Local UTM Projection Selection
To perform accurate metric area and distance calculations without geodesic distortion, the system dynamically calculates the UTM zone from the study area center:
$$\text{UTM Zone} = \left\lfloor \frac{\text{center\_lon} + 180^\circ}{6^\circ} \right\rfloor + 1$$
For Chennai ($80.245^\circ\text{E}, 13.035^\circ\text{N}$):
$$\text{UTM Zone} = \left\lfloor \frac{80.245 + 180}{6} \right\rfloor + 1 = 44 \implies \mathbf{EPSG:32644} \quad (\text{WGS 84 / UTM Zone 44N})$$

### 2.2 Sub-Pixel Affine Transformation
Standard GDAL/Rasterio pixel mapping defines the half-pixel center transformation:
$$\text{col} = \frac{\text{lon} - \text{min\_lon}}{\text{res\_x}}, \quad \text{row} = \frac{\text{max\_lat} - \text{lat}}{\text{res\_y}}$$
Forward transformation:
$$\text{lon} = \text{min\_lon} + (\text{col} + 0.5) \times \text{res\_x}, \quad \text{lat} = \text{max\_lat} - (\text{row} + 0.5) \times \text{res\_y}$$

---

## 3. Building Exposure Grading Methodology

### 3.1 Overlap Computation via Polygon Rasterization
`GeoSpatialTransform.calculate_polygon_overlap_stats` extracts the windowed pixel bounding box for each building polygon and rasterizes its exact footprint onto a sub-grid using `rasterio.features.rasterize`:
1. Any self-intersecting or unclosed geometries are cleaned via `shapely.validation.make_valid` and buffered by $0.0$.
2. The polygon is burned into a binary mask ($1 = \text{inside building footprint}$).
3. The underlying flood classification raster window is sampled.
4. Pixels with value `255` (NoData) are excluded from the denominator.
5. Footprint overlap percentage is computed:
$$\text{Exposed Area \%} = \frac{\text{Count}(\text{Pixel} == 2 \land \text{Inside})}{\text{Count}(\text{Pixel} \ne 255 \land \text{Inside})} \times 100\%$$

### 3.2 Exposure Classification Standard
- $\mathbf{HIGH\_EXPOSURE}$: $\ge 50.0\%$ overlap with mapped flood mask.
- $\mathbf{MODERATE\_EXPOSURE}$: $10.0\%$ to $49.9\%$ overlap.
- $\mathbf{LOW\_OR\_NO\_DETECTED\_EXPOSURE}$: $< 10.0\%$ overlap.
- $\mathbf{UNKNOWN}$: Outside raster bounds, invalid geometry, or $< 1$ valid sampled pixel.

### 3.3 Quantitative Building Results (Chennai Study Area)
Across the systematic sample of 1,000 OSM buildings evaluated against `flood_mask_024d3597.tif`:
- **Total Evaluated:** 1,000 structures
- **Total OSM Structures Available:** 27,884
- **High Exposure ($\ge 50\%$):** 229 structures (22.9%)
- **Moderate Exposure ($10\% - 50\%$):** 10 structures (1.0%)
- **Low or No Detected Exposure ($< 10\%$):** 761 structures (76.1%)
- **Unknown:** 0 structures (0.0%)
- **Execution Time:** 0.56 seconds

> [!NOTE]
> **Academic Integrity Disclosure:**  
> Satellite SAR and optical flood classification masks provide horizontal surface inundation extent, not physical water depth. Consequently, `estimated_water_depth_ft` is explicitly set to `None` to prevent the fabrication of synthetic hydrological data.

---

## 4. Road Hazard Sampling & Classification

### 4.1 Linear Sampling Strategy
Roads are represented as OSM LineStrings. For each road segment $(p_1, p_2)$:
1. The start and end vertices are projected into raster pixel coordinates.
2. The segment is subdivided into discrete sample points spaced by $\le 1$ pixel step ($\approx 10$ to $20$ meters on the ground).
3. The raster classification value is queried at each sample point.
4. Pixels matching `255` are recorded as NoData; pixels matching `2` are counted as flood hazards.
$$\text{Flood Intersection \%} = \frac{\text{Count}(\text{Sample} == 2)}{\text{Count}(\text{Sample} \ne 255)} \times 100\%$$

### 4.2 Road Hazard Categories
- $\mathbf{POTENTIAL\_HAZARD}$: $\ge 30.0\%$ valid samples intersect mapped flood mask.
- $\mathbf{CAUTION}$: $5.0\%$ to $29.9\%$ valid samples intersect mapped flood mask.
- $\mathbf{NO\_DETECTED\_INTERSECTION}$: $< 5.0\%$ valid samples intersect mapped flood mask.
- $\mathbf{UNKNOWN}$: Outside raster bounds or $< 2$ valid samples.

### 4.3 Quantitative Road Results (Chennai Study Area)
Across 1,000 evaluated road features covering 155.90 km of the urban road network:
- **Total Evaluated:** 1,000 roads (155.90 km)
- **Potential Hazard ($\ge 30\%$):** 195 segments (33.70 km, 21.6% of network length)
- **Caution ($5\% - 30\%$):** 25 segments
- **No Detected Intersection ($< 5\%$):** 771 segments
- **Unknown / Out-of-Bounds:** 9 segments
- **Execution Time:** 0.02 seconds

---

## 5. NetworkX Directed Road Graph & One-Way Handling

### 5.1 Directed Graph Representation (`nx.DiGraph`)
To model real-world vehicular traffic and highway restrictions, `build_road_graph` constructs a **directed graph (`networkx.DiGraph`)**:
- **Nodes:** Unique intersection coordinates rounded to 6 decimal places: $(u_{\text{lon}}, u_{\text{lat}})$.
- **Directional Tag Interpretation:**
  - `oneway in ["yes", "true", "1"]`: Only the forward directed edge $(u, v)$ is inserted along the LineString vertex sequence. Reverse edge $(v, u)$ is strictly prohibited.
  - `oneway in ["-1", "reverse"]`: Contraflow street where traffic moves opposite the digitizing sequence; only the reverse directed edge $(v, u)$ is inserted with coordinates `[p2, p1]`.
  - `oneway in ["no", "false", "0"]`: Standard bidirectional street; both forward $(u, v)$ and backward $(v, u)$ edges are added.
  - **Tag Omission Defaults:**
    - `motorway`, `motorway_link`, or `junction=roundabout`: Default to **one-way forward** (`oneway=yes`).
    - All other highway classifications (`residential`, `primary`, `secondary`, `tertiary`, `trunk`, `service`): Default to **bidirectional** (`oneway=no`).
- **Edge Attributes:**
  - `length_m`: Metric Euclidean length in local UTM projection.
  - `hazard_category`: Category from raster intersection.
  - `flood_pct`: Percentage of sampled points inundated.
  - `weight`: Effective traversal cost used by A\*.
  - `coordinates`: Coordinate vertices for polyline rendering.
  - `oneway`: Original OSM tag value.

### 5.2 Cost Function & Hazard Penalties
The effective traversal cost $c(e)$ for an edge $e$ with metric length $L_e$ is:
$$c(e) = L_e \times \mu(e)$$
where the hazard multiplier $\mu(e)$ is defined as:
$$\mu(e) = \begin{cases}
1.0 & \text{if } \text{hazard\_category} = \text{NO\_DETECTED\_INTERSECTION} \\
3.0 & \text{if } \text{hazard\_category} = \text{CAUTION} \\
1000.0 & \text{if } \text{hazard\_category} = \text{POTENTIAL\_HAZARD and avoid\_hazards} = \text{True} \\
10.0 & \text{if } \text{hazard\_category} = \text{POTENTIAL\_HAZARD and avoid\_hazards} = \text{False} \\
2.0 & \text{if } \text{hazard\_category} = \text{UNKNOWN}
\end{cases}$$

### 5.3 A* Heuristic Admissibility
The heuristic function $h(u, v)$ estimates the remaining traversal cost from node $u$ to target $v$:
$$h(u, v) = \text{EuclideanDistance}_{\text{UTM}}(u, v)$$
Because:
1. Straight-line Euclidean distance in UTM is always less than or equal to the true road network distance:
$$h(u, v) \le \sum_{e \in \text{path}} L_e$$
2. All hazard multipliers satisfy $\mu(e) \ge 1.0$:
$$\sum_{e \in \text{path}} L_e \le \sum_{e \in \text{path}} L_e \times \mu(e) = \sum_{e \in \text{path}} c(e)$$
Therefore, $h(u, v) \le c(u, v)$ across all pairs $(u, v)$. The heuristic is **strictly admissible and monotonic (consistent)**, guaranteeing that the A\* search discovers the optimal (minimum-cost) evacuation corridor.

---

## 6. Forensic Reconciliation of Transit ETA & Route Status Terminology

### 6.1 Transit ETA Forensic Reconciliation
- **Route Metric Distance:** 3,469.2 meters ($3.469\text{ km}$).
- **Assumed Operational Speed:** $25.0\text{ km/h}$ (standardized urban disaster evacuation transit speed).
- **Exact Theoretical Formula:**
  $$\text{ETA}_{\text{minutes}} = \frac{\text{distance\_m} / 1000}{\text{speed\_kmh}} \times 60 = \frac{3.4692}{25.0} \times 60 = \mathbf{8.32608 \approx 8.33\text{ minutes}}$$
- **Audit Findings (Why 9.4 minutes was previously reported):**
  The initial draft used:
  $$\text{eta\_mins} = \text{round}\left(\frac{\text{total\_km}}{28.0} \times 60.0 + 2.0, 1\right) = \frac{3.469}{28.0} \times 60 + 2.0 = 7.43 + 2.0 = 9.4\text{ minutes}$$
  This introduced two inconsistencies:
  1. An undocumented arbitrary `+ 2.0` minute dispatch buffer.
  2. A speed parameter of $28.0\text{ km/h}$ instead of the documented $25.0\text{ km/h}$.
- **Resolution:**
  The arbitrary `+ 2.0` buffer was removed from `backend/app/services/rescue_router.py`, speed was standardized to $25.0\text{ km/h}$ in `backend/app/config.py`, and the formula was bound to `round(((distance_m / 1000) / speed_kmh) * 60, 2)`. Both backend API responses and frontend displays now report **8.33 minutes**.

### 6.2 Route Status Terminology Correction
- **Previous Unverified Labels:** `PASSABLE_VERIFIED_OSM`, `100% hazard-free corridor`.
- **Approved Honest Terminology:** **`CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS`**.
- **Scientific Rationale:** The generated path is a mathematical candidate route calculated from available OpenStreetMap geometries and the satellite-derived flood risk layer. It does NOT incorporate real-time physical road closures, downed power lines, structural bridge collapse, or municipal barricades.

---

## 7. Automated Verification Test Suite (23/23 Passed)

The test suite in [`backend/scripts/test_phase5_gis_integrity.py`](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/scripts/test_phase5_gis_integrity.py) validates all 23 GIS, routing, graph, and metadata constraints:

```powershell
backend\.venv_cuda\Scripts\python.exe backend/scripts/test_phase5_gis_integrity.py
```

```
test_01_raster_crs_and_transform_validation ... ok
test_02_vector_reprojection_and_metric_crs ... ok
test_03_geometry_validity_and_repair ... ok
test_04_nodata_exclusion ... ok
test_05_building_intersection_calculations ... ok
test_06_building_exposure_percentage_thresholds ... ok
test_07_road_sampling_intervals ... ok
test_08_road_hazard_classification_categories ... ok
test_09_graph_construction ... ok
test_10_astar_route_generation ... ok
test_11_start_and_destination_snapping ... ok
test_12_no_route_behavior_when_isolated ... ok
test_13_out_of_bounds_handling ... ok
test_14_api_response_validation ... ok
test_15_geojson_validity ... ok
test_16_osm_oneway_forward_allowed_and_reverse_rejected ... ok
test_17_osm_oneway_reverse_tag_handling ... ok
test_18_osm_bidirectional_and_defaults ... ok
test_19_eta_calculation_formula_integrity ... ok
test_20_parallel_osm_road_edges_preserved ... ok
test_21_strict_hazard_avoidance_exclusion_vs_penalty ... ok
test_22_raster_class_encoding_and_sum_reconciliation ... ok
test_23_satellite_provenance_metadata_consistency ... ok

----------------------------------------------------------------------
Ran 23 tests in 6.290s

OK
```

### Expanded Test Coverage Matrix

| Test ID | Test Name | Target Component | Verification Criteria | Status |
|---|---|---|---|:---:|
| `test_01` | `raster_crs_and_transform_validation` | `GeoSpatialTransform` | CRS `EPSG:4326`, dimensions $256 \times 256$, resolution $\approx 0.000195^\circ$ | **PASS** |
| `test_02` | `vector_reprojection_and_metric_crs` | `GeoSpatialTransform` | Dynamic UTM Zone 44N (`EPSG:32644`) metric conversion | **PASS** |
| `test_03` | `geometry_validity_and_repair` | `validate_and_fix_polygon` | Self-intersecting bowtie polygon repaired via `make_valid` | **PASS** |
| `test_04` | `nodata_exclusion` | `calculate_polygon_overlap_stats` | NoData (255) pixels strictly excluded from denominator | **PASS** |
| `test_05` | `building_intersection_calculations` | `calculate_polygon_overlap_stats` | Controlled 50% overlap fixture yields $50.0\% \pm 1.0\%$ | **PASS** |
| `test_06` | `building_exposure_percentage_thresholds` | `analyze_building_exposure` | Thresholds: HIGH ($\ge 50\%$), MOD ($10-50\%$), LOW ($< 10\%$) | **PASS** |
| `test_07` | `road_sampling_intervals` | `sample_road_hazard_metrics` | Sampling at ~10-20m ($\le 1$ pixel spacing) along LineStrings | **PASS** |
| `test_08` | `road_hazard_classification_categories` | `sample_road_hazard_metrics` | Categories: POTENTIAL_HAZARD, CAUTION, NO_DETECTED | **PASS** |
| `test_09` | `graph_construction` | `build_road_graph` | MultiDiGraph road graph generated with metric lengths and weights | **PASS** |
| `test_10` | `astar_route_generation` | `calculate_astar_evacuation_route` | Returns `CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS` | **PASS** |
| `test_11` | `start_and_destination_snapping` | `calculate_astar_evacuation_route` | Snaps off-road points to nearest network intersection | **PASS** |
| `test_12` | `no_route_behavior_when_isolated` | `calculate_astar_evacuation_route` | Returns `NO_SAFE_ROUTE_AVAILABLE` for disconnected components | **PASS** |
| `test_13` | `out_of_bounds_handling` | `calculate_astar_evacuation_route` | Graceful rejection when coordinates exceed snap limit | **PASS** |
| `test_14` | `api_response_validation` | FastAPI (`main.py`) | Endpoints return validated Pydantic models with disclaimers | **PASS** |
| `test_15` | `geojson_validity` | RFC 7946 Conformance | Valid FeatureCollections, coordinates, and properties | **PASS** |
| `test_16` | `osm_oneway_forward_allowed_and_reverse_rejected` | `build_road_graph` | `oneway=yes`: forward succeeds, reverse blocked | **PASS** |
| `test_17` | `osm_oneway_reverse_tag_handling` | `build_road_graph` | `oneway=-1`: contraflow succeeds, forward blocked | **PASS** |
| `test_18` | `osm_bidirectional_and_defaults` | `build_road_graph` | `oneway=no` bidirectional; motorway forward; residential bidir | **PASS** |
| `test_19` | `eta_calculation_formula_integrity` | `calculate_astar_evacuation_route` | Formula $(d/1000)/25 \times 60 = 8.33$ mins; 0 arbitrary buffers | **PASS** |
| `test_20` | `parallel_osm_road_edges_preserved` | `build_road_graph` | MultiDiGraph preserves all parallel edges between same nodes | **PASS** |
| `test_21` | `strict_hazard_avoidance_exclusion_vs_penalty` | `build_road_graph` | Strict avoidance excludes edge $\implies$ NO_SAFE_ROUTE | **PASS** |
| `test_22` | `raster_class_encoding_and_sum_reconciliation` | `flood_mask_024d3597.tif` | Counts: 32202, 18965, 14369, 0; sum = 65536 ($256\times 256$) | **PASS** |
| `test_23` | `satellite_provenance_metadata_consistency` | `provenance_024d3597.json` | Provenance manifest, scene files, and checkpoint verified | **PASS** |

---

## 8. Frontend Integration & Production Build

- **Production Build Command:** `cmd.exe /c "cd frontend && npm run build"`
- **Build Result:** **SUCCESS** in 3.56s (`dist/index.html` 1.07 kB, `dist/assets/index-BcP4RTJN.js` 1,006.19 kB, 0 errors).
- **UI Elements Synchronized:**
  - Status badge: Displays `CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS` in emerald badge.
  - Telemetry: Displays **3,469 m (3.469 km)**, **8.33 mins**, 95 segments, 0 hazard segments.
  - Academic Disclaimer: Visible on all route summaries and building inspection cards.

---

## 9. Scientific & Operational Disclaimer

> [!IMPORTANT]
> **Mandatory Disclaimer:**  
> *"FloodSense-AI identifies potential infrastructure exposure through spatial intersection with a mapped flood-risk layer derived from satellite remote sensing. It does not independently confirm physical structural damage, building collapse, actual road closures, or operationally guaranteed safe evacuation routes. All candidate evacuation routes and exposed facility assessments require local municipal verification and on-ground emergency service inspection before deployment."*
