# FloodSense-AI: Comprehensive System Architecture & Engineering Audit

**Audit Date:** November 7, 2026  
**Project Title:** FloodSense-AI: Multi-Modal Flood Detection, Dynamic Flood Simulation, Infrastructure Damage Mapping and Emergency Rescue Routing System  
**Lead Engineer / Auditor:** Antigravity Principal Geospatial AI & Systems Architect  
**Repository Working Directory:** `c:\Users\vicky\Desktop\Disaster Response`  
**Git Checkpoint Reference:** `7ef22b6`  

---

## 1. Executive Summary

FloodSense-AI is a full-stack geospatial deep learning and emergency routing platform designed for rapid disaster response during catastrophic flood events. The application integrates multi-modal satellite data (Sentinel-2 Optical + Sentinel-1 C-band SAR), a PyTorch Dual-Stream Siamese Attention U-Net architecture, OpenStreetMap vector infrastructure, dynamic hydrological hazard propagation, and A* shortest safe rescue routing.

While previous development phases established a working React-FastAPI pipeline, Leaflet geospatial visualization, and OSM vector retrieval, this audit identifies critical divergence between live observational capabilities and simulated heuristics:
1. **Satellite Ingestion Provenance:** The system relies primarily on offline pre-rendered presets (`chennai_monsoon_2026`, `assam_brahmaputra_surge`, `wayanad_kerala_flashflood`) with ESRI World Imagery RGB tiles and procedurally generated SAR channels rather than live programmatic acquisition from the European Space Agency (Copernicus Data Space Ecosystem) or Google Earth Engine.
2. **Hydrological Simulation Physics:** The current `/api/simulate-timeline` endpoint in `hydro_forecaster.py` utilizes mathematical distance transforms (`cv2.distanceTransform`) and morphological dilation/erosion rather than a 2D shallow-water / diffusive wave mass-conservative fluid model ($\Delta h / \Delta t + \nabla \cdot q = R - I$).
3. **OSM Vector Integration:** Roads and buildings for presets are acquired from OpenStreetMap (Overpass & OSM API 0.6) and cached locally, but live AOI queries require robust fallback strategies when rate-limited.
4. **Model Metrics:** Model evaluation reported benchmark metrics derived from procedural testing; a rigorous evaluation suite against held-out validation tiles with reproducible confusion matrices is required.

This document serves as the technical ground truth and roadmap for upgrading FloodSense-AI into an academically defensible, real-world operational disaster-response platform.

---

## 2. Repository & Architectural File Structure

```
Disaster Response/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                 # System configurations, preset bounds, model hyperparameters
│   │   ├── main.py                   # FastAPI server entry point, routers, CORS, startup handlers
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── attention_unet.py     # DualStreamAttentionUNet PyTorch model definition
│   │   │   └── losses.py             # Combined BCE + Dice loss implementation
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── benchmark_suite.py    # Multi-model ablation benchmark runner
│   │       ├── damage_analyzer.py    # Building polygon damage extraction and severity index
│   │       ├── gee_connector.py      # Google Earth Engine interface
│   │       ├── georef.py             # Affine transforms, pixel-to-WGS84 mapping, raster bounds
│   │       ├── hydro_forecaster.py   # 48-hour flood progression engine (morphological)
│   │       ├── inference_engine.py   # PyTorch model loader, pre/post processing, NDWI integration
│   │       ├── osm_service.py        # Overpass/OSM 0.6 API client for roads and buildings
│   │       ├── preprocessor.py       # Cloud masking, SAR calibration (dB), Normalization
│   │       ├── rescue_router.py      # NetworkX graph builder, water penalty weights, A* pathfinder
│   │       └── sitrep_generator.py   # NDRF tactical situation report generator (Gemini / rule-based)
│   ├── data/
│   │   ├── checkpoints/              # PyTorch model weights (.pth)
│   │   ├── mini_sen1floods11/        # 24-tile multi-band validation dataset
│   │   ├── osm_cache/                # Cached GeoJSON roads and building footprints
│   │   └── presets/                  # Precomputed .npy multi-band tiles for offline viva demo
│   ├── scripts/
│   │   ├── fetch_osm_data.py         # Batch OSM vector fetcher for presets
│   │   ├── init_demo_data.py         # Procedural multi-band generator & weight initializer
│   │   ├── test_osm_api.py           # OSM API test runner
│   │   ├── test_overpass.py          # Overpass query testing script
│   │   ├── test_phase4.py            # Phase 4 integration and validation test suite
│   │   └── train_sen1floods11.py     # Training loop for DualStreamAttentionUNet
│   ├── requirements.txt              # Backend dependencies
│   └── run.py                        # Uvicorn entrypoint script
├── frontend/
│   ├── dist/                         # Production build artifacts
│   ├── src/
│   │   ├── components/
│   │   │   ├── AnalyticsDock.jsx         # Telemetry panels, land cover breakdown, damage counts
│   │   │   ├── ControlSidebar.jsx        # AOI selector, layer visibility, threshold sliders
│   │   │   ├── HeaderBar.jsx             # Status badges, disaster scenario selector, SitRep trigger
│   │   │   ├── MapViewport.jsx           # Leaflet interactive map, vector overlays, route visualization
│   │   │   ├── ModelBenchmarkTab.jsx     # F1-Score, IoU, inference latency ablation charts
│   │   │   ├── SitRepModal.jsx           # Military/NDRF tactical bulletin modal with export
│   │   │   ├── SplitCompareSlider.jsx    # Pre vs. Post optical/SAR curtain comparator
│   │   │   └── TimelineSimulatorPanel.jsx# T+0 to T+48 flood progression timeline playback
│   │   ├── services/
│   │   │   └── api.js                # Axios client connecting frontend to FastAPI backend
│   │   ├── App.jsx                   # Central state management, disaster presets, layer dispatch
│   │   ├── index.css                 # Tailwind CSS styling and geospatial map markers
│   │   └── main.jsx                  # React DOM mount point
│   ├── package.json                  # NPM packages & build scripts
│   ├── tailwind.config.js
│   └── vite.config.js
└── PROJECT_AUDIT.md                  # This document
```

---

## 3. Feature Classification (Categories A – F)

| Feature Component | Category | Current Status & Assessment |
| :--- | :---: | :--- |
| **PyTorch DualStreamAttentionUNet** | **Category A** | Genuine PyTorch model with dual encoders, spatial cross-attention gates, and multi-channel input (RGB+NIR pre-event, VV+VH post-event). Successfully compiles and runs on CUDA/CPU. |
| **BCE + Dice Loss Implementation** | **Category A** | Numerically stable combined loss function with epsilon smoothing, suitable for highly imbalanced water segmentation. |
| **Georeferencing & Affine Transforms** | **Category B** | Basic affine coordinate transformations via `pyproj` and forward/inverse matrix mapping. Lacks automatic UTM reprojection, sub-pixel raster resampling, and robust NoData nodata mask handling. |
| **OpenStreetMap Vector Roads & Buildings** | **Category A** | Successfully extracts real OSM geometries via Overpass and direct OSM API 0.6 fallback; caches GeoJSON locally for Chennai (5.2k roads, 27k buildings), Assam, and Wayanad. |
| **NetworkX A* Rescue Routing** | **Category B** | Builds geometric road graph with real node coordinates. Computes path cost dynamically penalized by water intersection. Needs dynamic recalculation against time-stepped simulation depth grids. |
| **Building Damage Assessment** | **Category B** | Calculates intersection of flood raster mask with building polygons, categorizing into Minor, Moderate, and Severe. Needs water depth distribution ($h > 0.3m, h > 1.0m$) rather than binary flood flags. |
| **Hydrological Simulation** | **Category C** | Currently uses Euclidean distance transforms and morphological dilation/erosion with synthetic speed. Must be upgraded to a 2D mass-conservative shallow-water / diffusive wave raster model. |
| **Satellite Imagery Ingestion** | **Category B** | Preset tiles are generated with ESRI World Imagery RGB stitched offline. Live dynamic Sentinel-1/2 fetching via Copernicus Data Space or GEE exists in rudimentary form (`gee_connector.py`) without full auth flow. |
| **Automated SitRep Generation** | **Category B** | Generates tactical NDRF bulletins using dynamic metrics; supports Gemini API and offline rule-based fallback. Needs multi-format export (PDF/Markdown/GeoJSON) and clear distinction between observed vs. simulated statistics. |
| **Model Benchmark Suite** | **Category B** | Ablation suite runs on synthetic mini-dataset. Requires real validation against held-out Sen1Floods11 splits with genuine confusion matrices and zero hardcoded metrics. |
| **User Interface & Visualization** | **Category A** | Professional React/Tailwind/Leaflet dashboard with SplitCompareSlider, timeline controls, benchmark charts, and modal reports. |
| **Walkthrough & Project Defense Mode**| **Category F** | Missing explicit interactive project review mode for academic/examiner evaluation. |

---

## 4. Coordinate Systems & Georeferencing Audit

1. **Current Coordinate Conventions:**
   - Map Viewport: Leaflet uses WGS 84 (`EPSG:4326`) latitude/longitude coordinate pairs `[lat, lon]`.
   - Backend GeoJSON: GeoJSON standards dictate `[lon, lat]` coordinate order.
   - Raster Matrix: Pixel grid `[y, x]` corresponding to row (northing/latitude) and column (easting/longitude).
2. **Current Alignment Implementation:**
   - In `backend/app/services/georef.py`, the system maps normalized image coordinates `[0, 1]` linearly between `min_lat`, `max_lat` and `min_lon`, `max_lon`.
   - Affine matrix: $x_{lon} = \min_{lon} + \frac{x}{W}(\max_{lon} - \min_{lon})$, $y_{lat} = \max_{lat} - \frac{y}{H}(\max_{lat} - \min_{lat})$.
3. **Identified Deficiencies:**
   - Linear unprojected interpolation across wide geographic bounds induces longitudinal distortion away from the equator.
   - Lack of automated local UTM zone detection (`EPSG:326XX` / `EPSG:327XX`) causes distance and area calculations to slightly misestimate real metric surface areas ($m^2$ and $km^2$).
   - Solution: Implement explicit ellipsoidal geodesic calculations (`pyproj.Geod` or UTM projection) for area and distance measurements in `georef.py`.

---

## 5. Model & ML Pipeline Audit

1. **Architecture:**
   - `DualStreamAttentionUNet`: Stream 1 consumes Pre-Flood Optical (4 channels: Red, Green, Blue, NIR), Stream 2 consumes Post-Flood SAR (2 channels: VV, VH).
   - Fusion: Cross-attention gating modules at encoder levels (64, 128, 256, 512 channels) fuse optical contextual features with SAR radar backscatter changes.
   - Decoder: Symmetrical transposed-convolution decoder with skip connections, outputting a single-channel logits map for water probability.
2. **Parameter Count:**
   - Total trainable parameters: ~17.4 Million parameters.
3. **Weight Provenance:**
   - Checkpoint: `backend/data/checkpoints/dual_stream_attention_unet.pth`.
   - Checkpoint is initialized with He-normal / Kaiming weights and fine-tuned on the procedural mini-dataset.
4. **Physical Prior Integration:**
   - The inference engine (`inference_engine.py`) blends model probability $P_{dl}$ with physical indices:
     - Normalized Difference Water Index: $NDWI = \frac{\rho_{Green} - \rho_{NIR}}{\rho_{Green} + \rho_{NIR}}$
     - SAR Backscatter Thresholding: $SAR_{diff} = \gamma_{VV}^{post} - \gamma_{VV}^{pre} < -3.5\text{ dB}$
     - Final Mask: $M = (P_{dl} > 0.5) \lor (NDWI > 0.1 \land SAR_{diff} < -2.0\text{ dB})$.
5. **Remediation Needed:**
   - Execute an honest validation run against held-out ground truth tiles in `mini_sen1floods11/val` and record genuine precision, recall, IoU, and F1 scores in `evaluation_results.json`.

---

## 6. Flood Simulation Engine Audit

1. **Existing Mechanism (`hydro_forecaster.py`):**
   - The current engine computes distance transforms from the initial water mask: $D(x, y) = \text{distance to nearest water}$.
   - Water expansion at time $t$ is calculated via:
     $$\text{Mask}_t = D(x, y) \le \text{rate} \cdot t \cdot (1 - \text{slope}) + \text{synthetic noise}$$
   - **Critical Limitation:** This is purely morphological computer vision. It does not conserve water mass ($\sum h_{i,j} \Delta A$), does not account for continuous rainfall flux $R(t)$, soil infiltration $I(t)$, drainage outfalls $D(t)$, or physical gravity-driven flow into lower DEM cells.
2. **Upgraded Engine Specification:**
   - Implement a 2D time-stepped raster flow equation:
     $$\frac{\partial h}{\partial t} + \frac{\partial q_x}{\partial x} + \frac{\partial q_y}{\partial y} = R(t) - I(t) - D(t)$$
   - Discharge $q$ between neighboring cells determined by water surface elevation difference ($H = z + h$) via Manning's approximation:
     $$q = \frac{1}{n} h^{5/3} S_0^{1/2}$$
   - Strict Mass Balance Verification:
     $$\text{Mass Error} = \left| V_t - (V_{t-1} + V_{rain} - V_{infil} - V_{drain}) \right| < 10^{-4}$$

---

## 7. OpenStreetMap & Vector Data Audit

1. **Road Network:**
   - Source: OpenStreetMap via Overpass API and direct OSM 0.6 API (`https://api.openstreetmap.org/api/0.6/map.json`).
   - Query Filter: `way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|track"]`.
   - Node count (Chennai): 5,272 real road segments cached in `backend/data/osm_cache/chennai_roads.geojson`.
   - Road classification: Segment hazards classified based on intersected water depth $h$:
     - $h < 0.05\text{ m}$: `OPEN` (Passable for all vehicles)
     - $0.05\text{ m} \le h < 0.30\text{ m}$: `CAUTION` (Passable for high-clearance / 4x4)
     - $0.30\text{ m} \le h < 0.80\text{ m}$: `HIGH_RISK` (Specialist rescue trucks / unimogs only)
     - $h \ge 0.80\text{ m}$: `BLOCKED` (Impassable; requires rescue boats)
2. **Building Footprints:**
   - Source: OpenStreetMap `way["building"]` and `relation["building"]`.
   - Node count (Chennai): 27,884 building polygons cached in `backend/data/osm_cache/chennai_buildings.geojson`.
   - Damage scoring evaluates average and peak flood depth across building polygon boundaries.

---

## 8. API Endpoints Audit

| Endpoint | Method | Parameters | Provenance | Current Behavior & Deficiencies |
| :--- | :---: | :--- | :---: | :--- |
| `/api/health` | GET | None | Real | Returns status `healthy`, device (`cuda`/`cpu`), timestamp. |
| `/api/presets` | GET | None | Real | Returns list of available presets (`chennai_monsoon_2026`, etc.). |
| `/api/analyze` | POST | `{preset_id, aoi_bounds}` | Real DL | Runs PyTorch model forward pass, applies NDWI/SAR priors, intersects OSM vectors. |
| `/api/route` | POST | `{preset_id, start_lat, start_lon, end_lat, end_lon, vehicle_type}` | Real Graph | Computes A* shortest path through OSM road graph avoiding flood polygons. |
| `/api/simulate-timeline` | POST | `{preset_id, rainfall_scenario, duration_hours}` | Morphological | Generates T+0 to T+48 snapshots using distance transform. Needs 2D physical hydraulic solver. |
| `/api/generate-sitrep` | POST | `{analysis_id, format}` | Rule/Gemini | Synthesizes disaster report from actual metrics. Needs PDF/Markdown/GeoJSON export endpoints. |
| `/api/benchmarks` | GET | `{metric}` | Empirical | Runs model comparison on validation set. Needs reproducible evaluation JSON. |
| `/api/osm-data` | GET | `{preset_id, type}` | Real OSM | Returns cached or live OSM GeoJSON features for roads and buildings. |
| `/api/acquire/copernicus` | POST | `{aoi, date_range}` | Missing | New endpoint needed for Copernicus Data Space Sentinel search/fetch. |
| `/api/acquire/local` | POST | `{file_upload}` | Missing | New endpoint needed for local multi-band GeoTIFF ingestion. |

---

## 9. Identified Bugs, Fragilities & Technical Debt

1. **DEM Synthetic Uniformity:** For offline presets without internet access to SRTM/Copernicus 30m DEM servers, the DEM was generated with low-frequency spatial noise. A realistic hydraulic gradient must be computed with drainage toward coastal or river basin outlets.
2. **Simulation Blocking I/O:** The 48-hour simulation is computationally intensive. When executed synchronously on a single thread, it blocks the FastAPI event loop for several seconds. An asynchronous background task runner with a polling `/api/simulation-status/{task_id}` endpoint is required.
3. **NoData Value Handling:** Satellite GeoTIFFs containing black margins (`NoData = 0` or `-9999`) can distort min-max normalization if not explicitly masked out.
4. **Demonstration Labelling:** Prior to this audit, UI elements did not clearly demarcate whether imagery and metrics originated from live satellite ingest or offline calibrated demonstration presets. Clear UI badges (`LIVE SATELLITE TELEMETRY` vs. `OFFLINE BENCHMARK PRESET`) are required for academic integrity.

---

## 10. Upgrade & Remediation Plan

The upgrade will be systematically executed across the following phases:
1. **Module 1: Real Satellite Data Acquisition Providers (`backend/app/services/data_providers/`):**
   - Create `base_provider.py` abstract interface.
   - Implement `copernicus_provider.py` with OAuth token handling and catalog search.
   - Implement `gee_provider.py` with Google Earth Engine authentication.
   - Implement `local_raster_provider.py` for arbitrary user GeoTIFF ingestion.
2. **Module 2: Geospatial Georeferencing & Projection Engine (`georef.py`):**
   - Add automated UTM zone calculation and geodesic area/length measurement.
   - Add multi-band raster alignment, clipping, and NoData masking.
3. **Module 3: Real Model Evaluation Suite (`backend/scripts/evaluate_model.py`):**
   - Compute honest IoU, Precision, Recall, F1, and confusion matrix over validation tiles.
   - Save verified metrics to `backend/data/checkpoints/evaluation_results.json`.
4. **Module 4: OSM Building & Multi-Tier Road Hazard Assessment:**
   - Update `damage_analyzer.py` and `rescue_router.py` with depth thresholds: `OPEN`, `CAUTION`, `HIGH_RISK`, `BLOCKED`.
5. **Module 5: Physically Motivated 2D Dynamic Time-Stepped Flood Engine (`simulation_engine.py`):**
   - Implement 2D raster flow solver ($h_{t+1} = h_t + \Delta t [R - I - D - \nabla \cdot q]$) with Manning's equation.
   - Real-time mass balance calculation with $< 0.1\%$ error constraint.
   - Asynchronous job execution and snapshot generator (T+0, T+1, T+3, T+6, T+12, T+18, T+24, T+36, T+48).
6. **Module 6: Dynamic Rescue Routing Recalculation:**
   - Connect simulation time-steps directly to `rescue_router.py` to recalculate safe rescue corridors as floodwaters advance.
7. **Module 7: Frontend Dashboards, Depth Heatmaps & Walkthrough Mode:**
   - Add water depth visualization (meters), mass-balance display, and Project Review Walkthrough mode for final viva defense.
8. **Module 8: SitRep Multi-Format Export:**
   - Support Markdown, JSON, GeoJSON, and printable report export with clear data provenance labels.
9. **Module 9: Academic Documentation Suite & End-to-End Verification:**
   - Author all 8 required system documentation files and verify the entire platform with end-to-end tests.
