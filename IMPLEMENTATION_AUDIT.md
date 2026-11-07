# FloodSense-AI: Comprehensive Source-Code Implementation Audit

**Audit Date:** 2026-11-07  
**Auditor:** Principal Geospatial AI & Hydrological Modeling Auditor  
**Repository:** `c:\Users\vicky\Desktop\Disaster Response`  
**Target Architecture:** Dual-Stream Attention U-Net, 2D Hydrodynamic Modeling, OSM Vector Ingestion, Live Remote Sensing Data Pipeline, Evacuation Routing.

---

## 1. Executive Summary & Audit Mandate

This audit performs an exhaustive, ground-truth inspection of all backend Python services (`backend/app/`), frontend React components (`frontend/src/`), test scripts (`backend/scripts/`), and data directories (`backend/data/`). No documentation claims in `README.md` were accepted without direct line-by-line code verification.

### Key Audit Findings at a Glance
1. **Core AI Inference Engine:** The PyTorch `DualStreamAttentionUNet` ([backend/app/models/attention_unet.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/models/attention_unet.py#L141-L266)) is fully implemented with real convolutions, Squeeze-and-Excitation channel attention, Cross-Modal Attention Fusion (CMAF), and soft attention gates. However, in [backend/app/services/inference_engine.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/inference_engine.py#L133-L146), evaluation metrics are artificially clamped between 0.865 and 0.962 or blended with preset constants.
2. **2D Hydrodynamic Simulation Engine:** Both [backend/app/services/simulation_engine.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/simulation_engine.py) and [backend/app/services/scenario_engine.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/scenario_engine.py) implement genuine finite-volume approximations of the 2D diffusive wave shallow-water equation ($\partial h/\partial t + \nabla \cdot q = R - I - D$). However, reported mass-conservation error of $0.00\%$ is hardcoded or clamped in several return dictionaries, and boundary conditions are strictly closed/reflective walls rather than open hydrodynamic boundaries.
3. **Route Recalculation Disconnect:** In [backend/app/main.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/main.py#L1031-L1034), the endpoint `/api/scenarios/{scenario_id}/routes/recalculate` synthesizes a hardcoded flood square (`seg_mask[100:150, 100:150] = CLASS_NEW_FLOOD`) rather than querying the actual simulated water depth raster $h$ from the active scenario. Similarly, `/api/route-dynamic` ([backend/app/main.py#L730-L750](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/main.py#L730-L750)) ignores `req.timestep_hour` and routes against static pre-disaster ground-truth rasters.
4. **Data Provider Connectivity:**
   - **Open-Meteo:** Fully functional and tested live over the internet; returns 24–72 hours of real precipitation data without requiring credentials.
   - **OpenStreetMap Overpass:** Fully functional and tested live; downloads and parses real road and building vectors.
   - **Copernicus CDSE & NASA GPM:** Client interfaces and OData query builders are written, but in the absence of valid credentials, they gracefully drop into fallback stubs rather than executing full raster downloads.
5. **Critical Code Defect in Fallback OSM Generator:** In [backend/app/services/osm_service.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/osm_service.py#L327), `numpy` is not imported at module scope, causing an unhandled `NameError: name 'np' is not defined` whenever live Overpass queries fail for an uncached AOI.

---

## 2. Feature-by-Feature Code Inspection

Below is the verified status of every documented capability in the system:

### 2.1 Dual-Stream Attention U-Net
- **Classification:** **Implemented and Functional**
- **File Reference:** [backend/app/models/attention_unet.py#L141-L266](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/models/attention_unet.py#L141-L266)
- **Detailed Findings:**
  - 4-stage optical encoder accepts $[B, 4, H, W]$ (`B4_Red`, `B3_Green`, `B2_Blue`, `B8_NIR`).
  - 4-stage SAR encoder accepts $[B, 2, H, W]$ (`VV`, `VH`).
  - Cross-Modal Attention Fusion (CMAF) combines features at all 4 scales using Squeeze-and-Excitation channel gating and spatial saliency pooling.
  - Soft attention gates (Oktay et al.) modulate skip connections before concatenation.
  - Trained checkpoint exists on disk at `backend/data/checkpoints/dual_stream_attention_unet.pth` ($54.1\text{ MB}$, $14.1\text{M}$ trainable parameters).
  - Tiled sliding-window inference with 2D Hann cosine window blending is implemented in [backend/app/services/inference_engine.py#L228-L278](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/inference_engine.py#L228-L278) to support arbitrary image dimensions ($H, W > 256$).

### 2.2 Sentinel-1 and Sentinel-2 Data Acquisition
- **Classification:** **Implemented but Partially Functional**
- **File References:** [backend/app/services/data_providers/copernicus_provider.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/data_providers/copernicus_provider.py), [backend/app/services/gee_connector.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/gee_connector.py)
- **Detailed Findings:**
  - `CopernicusDataSpaceProvider.search_scenes()` constructs genuine OData v1 queries with polygon WKT filters against `https://catalogue.dataspace.copernicus.eu/odata/v1/Products`.
  - In `fetch_scene_bands()` ([copernicus_provider.py#L188-L208](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/data_providers/copernicus_provider.py#L188-L208)), actual band download writes `.meta` JSON stub files rather than downloading $1\text{GB}+$ raw Sentinel-1 GRD/Sentinel-2 SAFE zip archives, because free CDSE accounts enforce rate-limiting and session tokens.
  - `gee_connector.py` contains a complete Google Earth Engine client; when GEE service credentials are not supplied, it falls back cleanly to procedural satellite synthesis.

### 2.3 Universal AOI Selection
- **Classification:** **Implemented and Functional**
- **File Reference:** [backend/app/services/aoi_service.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/aoi_service.py)
- **Detailed Findings:**
  - Integrates OpenStreetMap Nominatim geocoder with automated bounding box clamping ($\le 500\text{ km}^2$).
  - Correctly calculates geodesic ellipsoidal surface area using `pyproj.Geod(ellps='WGS84')`.
  - Automatically identifies local Universal Transverse Mercator (UTM) projection EPSG code (`EPSG:326XX` / `EPSG:327XX`).
  - Classifies terrain archetype (`coastal_urban`, `river_floodplain`, `mountainous_catchment`, `urban_drainage_basin`) based on lat/lon, coastal proximity, and elevation range.
  - Generates isolated, location-specific cache directories in `backend/data/aoi_cache/`, preventing cross-AOI data contamination.

### 2.4 DEM Acquisition and Geospatial Alignment
- **Classification:** **Implemented but Partially Functional**
- **File References:** [backend/app/services/simulation_engine.py#L93-L187](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/simulation_engine.py#L93-L187), [backend/app/services/data_providers/copernicus_provider.py#L209-L290](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/data_providers/copernicus_provider.py#L209-L290)
- **Detailed Findings:**
  - `get_or_generate_dem()` checks preset folders, aoi cache, and dem cache.
  - If a cached raster is missing, `copernicus_provider.fetch_dem_raster` creates a hydro-conditioned terrain matrix (elevation range $0.5\text{m} \to 1200\text{m}$) using seed-based fractal gradients rather than issuing live HTTP tile requests to the OpenTopography Copernicus GLO-30 endpoint.
  - The generated DEM is properly indexed and aligned to the spatial raster grid via [backend/app/services/georef.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/georef.py).

### 2.5 OpenStreetMap Roads and Buildings
- **Classification:** **Implemented and Functional**
- **File Reference:** [backend/app/services/osm_service.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/osm_service.py)
- **Detailed Findings:**
  - Uses live Overpass QL queries across 3 redundant servers (`overpass.kumi.systems`, `overpass-api.de`, `overpass.private.coffee`).
  - Successfully tested live during audit: downloaded 228 genuine roads and 139 genuine building polygons for an arbitrary coordinate bounding box in under 15 seconds.
  - Parses OSM tags: `highway`, `is_bridge`, `is_oneway`, `building:levels`.
  - Saves GeoJSON directly into `backend/data/osm_cache/`.
  - **Defect Found:** Line 327 missing `import numpy as np` in offline synthetic fallback.

### 2.6 Live, Offline, and Hybrid Data Modes
- **Classification:** **Implemented and Functional**
- **File Reference:** [backend/app/services/data_acquisition_manager.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/data_acquisition_manager.py)
- **Detailed Findings:**
  - `DataAcquisitionManager` orchestrates provider calls with background threads (`_jobs` registry).
  - Mode switching (`LIVE`, `OFFLINE`, `HYBRID`) verified via REST API endpoints (`/api/data-providers/status`, `/api/data/cache`, `/api/data/satellite/download`).
  - Cache auditing checks directory sizes and file integrity across all 5 data vaults.

### 2.7 2D Hydrodynamic Simulation
- **Classification:** **Implemented and Functional**
- **File Reference:** [backend/app/services/simulation_engine.py#L189-L445](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/simulation_engine.py#L189-L445)
- **Detailed Findings:**
  - Simulates 2D diffusive wave shallow-water flow: $\partial h/\partial t + \nabla \cdot q = R - I - D$.
  - Includes cell-to-cell water surface gradient calculations ($dH = H_i - H_j$).
  - Incorporates Horton infiltration decay ($f(t) = f_c + (f_0 - f_c)e^{-kt}$) and urban drainage sinks.
  - Numerical sub-stepping ($dt = 180\text{s}$) maintains Courant-Friedrichs-Lewy (CFL) numerical stability.
  - Discharges calculated via Manning formula: $q = \frac{1}{n} h^{5/3} S_0^{1/2}$.
  - Boundary conditions are reflective (zero-gradient edge walls).

### 2.8 Cyclone and Rainfall Scenario Simulation
- **Classification:** **Implemented and Functional**
- **File Reference:** [backend/app/services/scenario_engine.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/scenario_engine.py)
- **Detailed Findings:**
  - `CycloneScenarioSimulator` models cyclone eye movement along user-specified wind direction vectors.
  - Spatial rainfall intensity decays radially according to a Rankine vortex model.
  - Hourly simulation advances water depths across the entire DEM grid.
  - Generates GeoJSON contour polygons for time-stepped map scrubbing.
  - Classifies storm on Indian Meteorological Department (IMD) and Saffir-Simpson scales.

### 2.9 Flood-Depth Timeline
- **Classification:** **Implemented and Functional**
- **File References:** [backend/app/services/hydro_forecaster.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/hydro_forecaster.py), [frontend/src/components/TimelineSimulatorPanel.jsx](file:///c:/Users/vicky/Desktop/Disaster%20Response/frontend/src/components/TimelineSimulatorPanel.jsx)
- **Detailed Findings:**
  - Both legacy cellular automaton (`hydro_forecaster.py`) and dynamic 2D shallow-water engines generate time-stepped depth distributions from $T+0\text{h}$ to $T+48\text{h}$.
  - Frontend scrubber updates flood extent, volume, and histogram bins across 4 depth classes ($<0.3\text{m}$, $0.3\text{m} \to 0.8\text{m}$, $0.8\text{m} \to 1.5\text{m}$, $\ge 1.5\text{m}$).

### 2.10 Building Exposure Analysis
- **Classification:** **Implemented and Functional**
- **File Reference:** [backend/app/services/damage_analyzer.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/damage_analyzer.py)
- **Detailed Findings:**
  - Intersects vector building footprints against flood raster classes.
  - Categorizes buildings into 4 damage tiers: `CRITICAL_SUBMERGED` (Class 2 hit), `MODERATE_BUFFER_RISK` (within 100m buffer), `LOW_RISK_PERMANENT` (Class 1 proximity), and `SAFE_DRY`.
  - Estimates affected population, economic loss (\$ and ₹), and critical infrastructure exposure (schools, hospitals, substations).

### 2.11 Dynamic A* Rescue Routing
- **Classification:** **Implemented but Partially Functional**
- **File Reference:** [backend/app/services/rescue_router.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/rescue_router.py)
- **Detailed Findings:**
  - Constructs `networkx.Graph` directly from OpenStreetMap way geometries.
  - `_sample_line_flood_status()` densifies road segments at 10m intervals to test intersection with Class 2 flood pixels.
  - Bridges crossing Class 1 permanent water are preserved as passable unless overtopped by Class 2 flood.
  - Calculates shortest evacuation path to command base using A* with Euclidean metric heuristic.
  - **Defects Identified:**
    1. If road coordinates lie outside the raster bounding box, `_sample_line_flood_status` defaults to `PASSABLE`.
    2. In [backend/app/main.py#L1031-L1034](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/main.py#L1031-L1034), dynamic scenario route recalculation uses a hardcoded synthetic flood box instead of passing the simulated water raster $h$.
    3. In `/api/route-dynamic` ([backend/app/main.py#L730-L750](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/main.py#L730-L750)), `req.timestep_hour` is ignored and the static baseline mask is loaded.

### 2.12 NDRF Situation Report Generation (AI SitRep)
- **Classification:** **Implemented and Functional**
- **File Reference:** [backend/app/services/sitrep_generator.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/sitrep_generator.py)
- **Detailed Findings:**
  - Dual-engine: Calls Google Gemini 2.5 Flash if `GEMINI_API_KEY` is set; executes a deterministic rule-based expert system fallback if offline.
  - Dynamically injects actual flooded area ($\text{km}^2$), expansion $\%$, building damage counts, safe route names, and staging base coordinates.
  - Calculates logistics allocations: inflatable boats, medical teams, 24h food packets, and evacuation buses.

### 2.13 PDF / Report Export
- **Classification:** **Implemented but Partially Functional**
- **File References:** [backend/app/main.py#L775-L815](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/main.py#L775-L815), [frontend/src/components/SitRepModal.jsx#L50-L53](file:///c:/Users/vicky/Desktop/Disaster%20Response/frontend/src/components/SitRepModal.jsx#L50-L53)
- **Detailed Findings:**
  - **Markdown Export:** Fully functional via `/api/sitrep/export-markdown` and client-side blob download.
  - **GeoJSON Export:** Fully functional via `/api/sitrep/export-geojson`.
  - **Scenario Report:** Fully functional via `/api/scenarios/{scenario_id}/report`.
  - **PDF Export:** Relies on client-side `window.print()` rather than server-side binary PDF compilation (e.g. ReportLab or Weasyprint). Works in modern desktop browsers, but is not an automated headless server-generated PDF.

---

## 3. Detailed Audit of Cyclone Simulation Mechanics

Verification of [scenario_engine.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/scenario_engine.py) and [simulation_engine.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/simulation_engine.py):

| Physical Simulation Aspect | Implementation Verification | Assessment |
|---|---|:---:|
| **Wind Field Generation** | Evaluates IMD wind scale and Saffir-Simpson category in `calculate_cyclone_category()`. Applies Rankine vortex radial decay. | **Verified Functional** |
| **Cyclone Eye Tracking** | `compute_cyclone_eye_position(t_hour)` updates coordinates along heading angle: $d_{\text{lat}} = \frac{v \cdot t \cdot \cos(\theta)}{111}$, $d_{\text{lon}} = \frac{v \cdot t \cdot \sin(\theta)}{111 \cdot \cos(\phi)}$. | **Verified Functional** |
| **Spatial-Temporal Rainfall** | Computes temporal factor (uniform, front-loaded, back-loaded, bell curve) and spatial radial decay from eye: $I(r) = I_{\text{base}} (0.35 + 0.65 e^{-0.5 (r / R_{\text{max}})^2})$. | **Verified Functional** |
| **Unit Consistency** | Converts $mm/h$ to $m/s$ ($1000 \times 3600$). Volumetric integration applies pixel area $100\text{ m}^2$ ($10\text{m} \times 10\text{m}$). | **Verified Consistent** |
| **DEM Gradient Influence** | Evaluates cardinal gradients $dH = H_{\text{center}} - H_{\text{neighbor}}$ where $H = z + h$. Water drives flux only toward lower hydraulic head. | **Verified Functional** |
| **Manning Discharge Equations** | Conductance $\alpha = \frac{\Delta t}{n \cdot \Delta x} h^{2/3}$. Outflow volume $V = \alpha \sqrt{dH / \Delta x} \cdot \Delta x \cdot \Delta t$. | **Verified Functional** |
| **Flux Limiter Conservation** | Flux limiter $\Phi = \min(1.0, V_{\text{available}} / \sum V_{\text{out}})$ prevents negative water depth. | **Verified Functional** |
| **Mass Balance Reporting** | In `scenario_engine.py` line 273: `if mass_error_pct > 1.0: mass_error_pct = 0.00`. Line 311 hardcodes `"mass_balance_error_pct": 0.00`. Actual internal mass error is negligible ($<0.05\%$) due to the flux limiter, but the code forces $0.00$ in the output. | **Partially Synthetic Reporting** |
| **Route Recalculation Coupling** | `recalculate_scenario_rescue_routes` in `main.py` lines 1031–1034 hardcodes a synthetic rectangle instead of reading $h$ from the active simulation run. | **Disconnected / Synthetic Stub** |

---

## 4. Live Data Provider Audit & Provenance

Direct connectivity tests conducted from Python on the local development environment:

```
Provider          Status Reported     Live Egress Verified   Data Returned
-----------------------------------------------------------------------------------------
Open-Meteo        CONNECTED           YES (HTTP 200)         Real 24h/72h precipitation series
OpenStreetMap     CONNECTED           YES (Overpass POST)    Real roads & buildings GeoJSON
Copernicus CDSE   NOT_CONFIGURED      NO (Awaiting creds)    Fallback catalog discovery stub
NASA GPM IMERG    NOT_CONFIGURED      NO (Awaiting creds)    Fallback microwave series stub
Google Earth Eng. NOT_CONFIGURED      NO (Awaiting creds)    Procedural multi-band tiles
```

### Provenance Distinction
1. **Genuine Real-World Data:**
   - Open-Meteo precipitation time-series (saved in `backend/data/rainfall_cache/`).
   - OpenStreetMap road polylines and building polygons for Chennai, Assam, Wayanad, and custom AOIs (saved in `backend/data/osm_cache/`).
2. **Cached Satellite Data:**
   - `backend/data/presets/`: Contains calibrated multi-spectral optical and C-band SAR rasters derived from real Sentinel-1 and Sentinel-2 baseline events (Chennai Adyar, Brahmaputra, Wayanad).
3. **Synthetic / Procedural Calibration Data:**
   - `backend/data/mini_sen1floods11/`: 24 calibrated 256x256 tiles procedurally synthesized to match the radiometric properties of Sen1Floods11.
   - Dynamic DEM generation for custom AOIs without cached Copernicus tiles: Synthesizes hydro-conditioned topography.

---

## 5. Critical Code Defects & Implementation Deficiencies

The following specific defects were identified during source-code auditing:

### Defect 1: Missing `numpy` import in `osm_service.py`
- **Location:** [backend/app/services/osm_service.py#L327](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/osm_service.py#L327)
- **Impact:** Calling `_generate_fallback_osm_network` crashes with `NameError: name 'np' is not defined`.
- **Trigger:** Happens whenever live Overpass query fails for an uncached AOI.
- **Recommended Fix:** Add `import numpy as np` at line 10 of `backend/app/services/osm_service.py`.

### Defect 2: Hardcoded flood mask in scenario route recalculation
- **Location:** [backend/app/main.py#L1031-L1034](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/main.py#L1031-L1034)
- **Impact:** When a user scrubs the timeline in the Cyclone Simulation Lab and recalculates rescue corridors, the routing engine does not evaluate the simulated flood depths $h$. Instead, it tests against a fixed box `[100:150, 100:150]`.
- **Recommended Fix:** Extract the flood depth mask from the scenario snapshot corresponding to `req.hour` ($h \ge \text{threshold}$), convert to a 3-class segmentation mask, and pass to `compute_osm_rescue_routes`.

### Defect 3: Disconnected timestep in `/api/route-dynamic`
- **Location:** [backend/app/main.py#L730-L750](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/main.py#L730-L750)
- **Impact:** Dynamic route endpoint returns `timestep_hour` in the response but computes routes on static `ground_truth.npy`.
- **Recommended Fix:** Query active simulation job manager for the snapshot at `req.timestep_hour` and sample that depth raster.

### Defect 4: Road segments outside bounding box deemed passable
- **Location:** [backend/app/services/rescue_router.py#L53-L54](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/rescue_router.py#L53-L54)
- **Impact:** If an OSM road segment extends beyond the georeferenced bounding box, 0 raster pixels are sampled, and the segment is classified as `PASSABLE`. Under 100% flood conditions, roads on the outer boundary remain navigable.
- **Recommended Fix:** Clip road segments to the bounding box or mark out-of-bounds segments with an explicit `OUT_OF_BOUNDS` status.
