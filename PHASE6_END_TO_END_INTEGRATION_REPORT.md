# FloodSense-AI — Phase 6: End-to-End System Integration Report

**Project Title:** FloodSense-AI: Dual-Stream Multimodal Deep Learning & GIS-Integrated Flood Emergency Response System  
**Academic Context:** Final-Year Bachelor of Engineering (B.E.) in Computer Science and Engineering Capstone Project  
**Target Hardware:** Intel Core i7-14700HX, NVIDIA RTX 4070 Laptop GPU (8GB VRAM), 16GB DDR5 RAM, Windows 11 Pro  
**Development Cost:** \$0.00 (Zero-Cost Local Architecture — No Paid Cloud Hosting, No Cloud GPUs, No Paid APIs)  
**Date of Audit & Integration:** November 8, 2026  
**Final Status:** **READY FOR END-TO-END DEMONSTRATION**

---

## 1. Executive Summary

Phase 6 unifies all independent modules developed across Phases 1 through 5.2 into a single, cohesive, end-to-end disaster response system. Prior to Phase 6, satellite data retrieval, physical tensor preprocessing, deep learning flood inundation inference, temporal SAR change analysis, OpenStreetMap infrastructure exposure grading, and A* candidate evacuation routing existed as discrete backend services requiring separate API calls or isolated scripts.

Phase 6 delivers:
1. **The Unified Backend Workflow Orchestrator (`FloodSensePipelineOrchestrator`):** A thread-safe, stage-aware orchestration engine executing an 8-stage automated pipeline with atomic JSON persistence, per-study-area execution locking, and graceful degradation on missing temporal data.
2. **Dedicated FastAPI Endpoints (`/api/pipeline/*`):** Exposing background pipeline trigger, polling status, full structured result retrieval, historic manifest listing, and orchestrator health telemetry.
3. **The Unified Frontend Command Dashboard (`FloodSensePipelinePage.jsx`):** An interactive, mission-control dashboard with study-area selection, module triggers, live stage execution progress cards, summary metrics dock, interactive multi-layer Leaflet GIS map, detailed telemetry tabs, execution history loader, and prominent academic disclaimers.
4. **Automated Test Suite (`test_phase6_pipeline.py`):** 18 comprehensive integration tests validating data validation, stage progression, error handling, atomic persistence, and API contract conformity. All 18 tests pass in 3.88s, alongside 43 regression tests across previous phases (61/61 total passing).

---

## 2. Integrated Pipeline Architecture

The end-to-end integration coordinates 8 sequential stages, ensuring strict physical, mathematical, and spatial consistency:

```
[Satellite Data Sources] (CDSE / GEE / Offline Presets)
            │
            ▼
[Stage 1: Data Acquisition] ──► Validates AOI Bounding Box & Retrieves Compatible Optical & SAR Rasters
            │
            ▼
[Stage 2: Physical Preprocessing] ──► CLAHE (Optical), NDWI Index, Decibel Transform, Lee-Sigma Speckle Filter
            │
            ▼
[Stage 3: Deep Learning Inference] ──► DualStreamAttentionUNet_v2.1 (40.6M params) -> 3-Class Flood Probability Mask
            │
            ▼
[Stage 4: Temporal Change Analysis] ──► Sentinel-1 SAR Differencing (ΔVV < -5 dB) [Graceful fallback if pre-event SAR absent]
            │
            ▼
[Stage 5: OSM Building Exposure] ──► Polygon Intersection: HIGH (≥50%), MODERATE (10–50%), LOW (<10%)
            │
            ▼
[Stage 6: Road Hazard Network] ──► Highway Hazard Sampling & MultiDiGraph (Preserving 993 parallel road segments)
            │
            ▼
[Stage 7: A* Rescue Routing] ──► Admissible Euclidean Heuristic, Hazard Avoidance, Exact Transit ETA: (d/1000)/25 * 60
            │
            ▼
[Stage 8: Finalization] ──► Atomic JSON Manifest Persistence in backend/data/pipeline_runs/exec_*.json
```

---

## 3. Orchestrator Specification (`floodsense_pipeline.py`)

The pipeline orchestrator is implemented in `backend/app/services/floodsense_pipeline.py` as `FloodSensePipelineOrchestrator`.

### 3.1 Concurrency & Safety Controls
- **Thread-Safe Memory Management:** Utilizes a standard Python `threading.Lock()` protecting internal dictionaries (`_active_runs`, `_study_area_locks`).
- **Per-Study-Area Execution Locking:** Prevents race conditions and duplicate disk writes by prohibiting simultaneous pipeline runs on the same geographic AOI. Concurrent requests receive an HTTP 409 Conflict.
- **Asynchronous Worker Thread:** Launches dedicated daemon threads (`PipelineWorker-exec_<uuid>`), immediately returning an execution ID to the client so UI polling never blocks FastAPI server threads.
- **Atomic Manifest Persistence:** Uses a write-to-temporary (`.tmp`) and atomic rename (`shutil.move`) strategy targeting `backend/data/pipeline_runs/exec_<uuid>.json` to ensure external processes never read partial or corrupted JSON files.

### 3.2 Stage-by-Stage Workflow & Error Handling

| Stage # | Stage Enum | Operations & Inputs | Outputs & Artifacts | Graceful Degradation / Fallback |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `INITIALIZATION` | Validate parameters, assign `exec_<uuid>`, acquire study-area lock. | Initial pending manifest. | Rejects conflicting active area runs. |
| **1** | `DATA_ACQUISITION` | Resolve bbox, retrieve Sentinel-2 (B4, B3, B2, B8) and Sentinel-1 (VV, VH) from local cache or central acquisition manager. | `opt_arr`, `sar_arr`, `pre_event_sar_arr`. | If remote island or unregistered study area, fails cleanly with `FileNotFoundError`. |
| **2** | `PREPROCESSING` | Validate channel dimensions (`[4, H, W]` and `[2, H, W]`), initialize `GeoSpatialTransform`, apply CLAHE, NDWI, dB conversion, and Lee-Sigma filter. | Normalized PyTorch tensors on GPU (`cuda`). | Validates dimensions; raises `ValueError` on corrupt tensor bands. |
| **3** | `FLOOD_INFERENCE` | Execute `DualStreamAttentionUNet_v2.1` with trained checkpoint `best_checkpoint.pth` (or baseline weights). | Georeferenced GeoTIFF (`flood_mask_<uuid>.tif`), pixel counts, inundated hectares/km². | Preserves full error log if PyTorch forward fails. |
| **4** | `TEMPORAL_ANALYSIS` | If pre-event SAR exists on disk, compute backscatter diff $\Delta\text{VV} = \text{VV}_{\text{post}} - \text{VV}_{\text{pre}}$ with fixed $-5.0\text{ dB}$ threshold. | Changed pixel counts, temporal inundation flags. | **Graceful Fallback:** If absent, flags `UNAVAILABLE` (`NO_PRE_EVENT_OBSERVATION`), logs warning, sets status `COMPLETED_WITH_WARNINGS` without crashing. |
| **5** | `BUILDING_EXPOSURE` | Query cached/live OSM building footprints, validate/repair geometries via Shapely, rasterize, and grade exposure. | `HIGH_EXPOSURE` ($\ge 50\%$), `MODERATE_EXPOSURE` ($10\text{–}50\%$), `LOW_OR_NO_DETECTED_EXPOSURE` ($<10\%$). | Empty building collections return zeroed summaries without error. |
| **6** | `ROAD_HAZARD_ANALYSIS` | Sample road vertices and interior points at 10m intervals against flood raster. Build `nx.MultiDiGraph`. | Classified roads GeoJSON (`POTENTIAL_HAZARD`, `CAUTION`, `NO_DETECTED_INTERSECTION`), MultiDiGraph. | Respects OSM `oneway` tags and preserves parallel edges. |
| **7** | `RESCUE_ROUTING` | Snap origin/destination to graph, execute A* shortest path with Euclidean metric heuristic, calculate transit ETA. | Candidate evacuation corridor GeoJSON, metric distance ($m$), ETA ($min$). | If isolated or submerged, returns `NO_SAFE_ROUTE_AVAILABLE` gracefully ($0.0\text{ m}$). |
| **8** | `FINALIZATION` | Aggregate run metadata, compute total wall-clock duration, serialize atomic JSON manifest to disk. | `backend/data/pipeline_runs/<exec_id>.json`. | Locks released in `finally:` block under all failure modes. |

---

## 4. FastAPI Endpoints Specification

Five dedicated REST endpoints are exposed in `backend/app/main.py`:

### 1. `POST /api/pipeline/run`
- **Request Body:**
  ```json
  {
    "study_area": "chennai_monsoon_2026",
    "bbox": [12.980, 80.190, 13.060, 80.290],
    "run_temporal_analysis": true,
    "run_infrastructure_exposure": true,
    "run_candidate_routing": true,
    "avoid_potential_hazards": true,
    "start_coords": [80.259097, 13.015806],
    "dest_coords": [80.266400, 13.039248],
    "max_buildings": 1000
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "success": true,
    "execution_id": "exec_40f0c0ae2f1c",
    "study_area": "chennai_monsoon_2026",
    "status": "PENDING",
    "message": "Pipeline execution 'exec_40f0c0ae2f1c' launched successfully.",
    "created_at": "2026-11-08T10:52:15.123456+00:00"
  }
  ```
- **Response (409 Conflict):** If study area already has an active running pipeline.

### 2. `GET /api/pipeline/status/{execution_id}`
- **Response (200 OK):** Returns live stage progression, stage durations, warnings list, and error messages.
- **Response (404 Not Found):** If execution ID is invalid.

### 3. `GET /api/pipeline/result/{execution_id}`
- **Response (200 OK):** Returns complete structured JSON payload including raster metadata, layer GeoJSONs, exposure tables, and routing geometry.
- **Response (404 Not Found):** If run is still pending/running or not found.

### 4. `GET /api/pipeline/history?limit=20`
- **Response (200 OK):** Returns sorted array of recent executions from memory and disk.

### 5. `GET /api/pipeline/health`
- **Response (200 OK):**
  ```json
  {
    "status": "HEALTHY",
    "active_executions": 0,
    "pending_executions": 0,
    "completed_in_memory": 2,
    "failed_in_memory": 0,
    "locked_study_areas": [],
    "runs_directory": "C:\\Users\\vicky\\Desktop\\Disaster Response\\backend\\data\\pipeline_runs",
    "timestamp": "2026-11-08T10:53:00.000000+00:00"
  }
  ```

---

## 5. Unified Frontend Command Dashboard

The frontend interface is implemented in `frontend/src/pages/FloodSensePipelinePage.jsx` and integrated into `App.jsx` and `Sidebar.jsx`.

### Key Dashboard Components:
1. **Mission Control Header:** Displays live backend engine connectivity, active execution count, breadcrumb navigation, and an academic disclaimer callout.
2. **Pipeline Execution Controller:**
   - Study area selector dropdown with pre-configured coordinates and corridor presets (e.g., Chennai Coastal Basin, Bolivia Mamoré Basin).
   - Module activation checkboxes (Temporal SAR, Building Exposure, A* Routing, Hazard Avoidance).
   - Footprint sample size input (100–5,000 structures).
   - One-click "Execute Full Pipeline" action with loading spinner.
3. **Live Stage Progress Tracker:** Visual 8-card grid showing real-time status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`), step indicators, and precise millisecond runtime for every stage.
4. **Summary Metrics Dock:** Displays flood inundation area (hectares, $km^2$, pixels), high-exposure structures, road hazard count, and candidate route distance and ETA.
5. **Unified Multi-Layer GIS Map (`SharedMap`):** Renders all layers simultaneously with independent toggles:
   - Deep-learning flood prediction mask.
   - OSM building polygons color-coded by exposure category (Red: High, Amber: Moderate, Green: Low).
   - OSM highway network color-coded by hazard status (Red: Hazard, Amber: Caution, Green: Clear).
   - A* candidate evacuation corridor polyline (Emerald).
6. **Detailed Findings Tabs:**
   - **Telemetry & Checkpoint Provenance:** Complete sensor bands, CRS, spatial grid dimensions, PyTorch architecture, and model weights origin.
   - **Building Exposure Table:** Paginated list of analyzed structures with OSM IDs, overlap areas ($m^2$), overlap percentages, and centroid coordinates.
   - **Road Hazard & Routing Corridor:** MultiDiGraph node/edge counts, snapping proximity ($m$), metric transit distance, and ETA calculations.
   - **Execution History:** Direct inspection and loading of past executions stored on disk.

---

## 6. Automated Test Suite Results

The Phase 6 test suite was executed in the local CUDA Python 3.10 environment using `backend/scripts/test_phase6_pipeline.py`.

```
======================================================================
FloodSense-AI Phase 6 Integration Test Suite
======================================================================
test_01_request_model_validation .................................... OK
test_02_atomic_persistence_and_naming ............................... OK
test_03_pipeline_status_constants ................................... OK
test_04_full_pipeline_execution_chennai .............................. OK
test_05_missing_satellite_input_handling ............................. OK
test_06_invalid_band_tensor_dimension_check .......................... OK
test_07_checkpoint_validation ....................................... OK
test_08_inference_engine_output_validity ............................. OK
test_09_temporal_analysis_graceful_fallback .......................... OK
test_10_missing_osm_data_handling .................................... OK
test_11_building_exposure_categories ................................. OK
test_12_road_hazard_multidigraph_preservation ........................ OK
test_13_astar_metric_length_and_eta_formula .......................... OK
test_14_no_route_handling_when_disconnected .......................... OK
test_15_partial_failure_output_preservation .......................... OK
test_16_provenance_metadata_completeness ............................. OK
test_17_fastapi_endpoints_integration ................................ OK
test_18_study_area_execution_lock .................................... OK
----------------------------------------------------------------------
Ran 18 tests in 3.883s

OK
```

### Full System Regression Verification:
- **Phase 6 Pipeline Integration:** 18/18 tests passed (`0 errors`, `0 failures`) in 3.883s.
- **Phase 5 GIS & MultiDiGraph Integrity (`test_phase5_gis_integrity.py`):** 23/23 tests passed in 10.489s.
- **Phase 4E Multi-Chip Holdout Integrity (`test_phase4e_integrity.py`):** 14/14 tests passed in 0.217s.
- **Phase 4C/4D Temporal Change Integrity (`test_flood_change_detection_integrity.py`):** 6/6 tests passed in 0.024s.
- **Frontend Production Build (`npm run build`):** Compiled cleanly in 3.51s with 0 errors.
- **Total Automated Test Coverage:** **61/61 tests passing across entire repository**.

---

## 7. Academic Integrity & Operational Disclaimers

In strict adherence to academic research ethics for B.E. Computer Science capstone evaluation:

1. **Non-Operational Safety Notice:** FloodSense-AI is a computer vision and computational GIS research demonstrator. Candidate evacuation routes produced by the A* algorithm are mathematical shortest paths avoiding mapped high-water probabilities. They do **not** confirm real-time ground passability, current water depth, road washouts, structural bridge failure, or operational safety. All evacuation actions require on-the-ground authorization by NDRF, SDRF, or district disaster management authorities.
2. **Model Predictions vs. Historical Observations:** Mapped inundation zones generated by `DualStreamAttentionUNet_v2.1` represent statistical deep learning inference and remote-sensing physical priors. They are clearly distinguished from ground-truth sensor observations and verified official flood extent maps.
3. **No Synthetic Depth Fabrication:** Satellite SAR and optical imagery provide surface reflectance and backscatter, not volumetric water depth. FloodSense-AI strictly refrains from fabricating synthetic or unverified millimeter/meter depth measurements on flat 2D satellite rasters.

---

## 8. Final Acceptance Verdict

All objectives defined in the Phase 6 specification have been completely implemented, verified, and audited:
- Complete end-to-end orchestration pipeline operational.
- Thread-safe background execution with atomic disk persistence.
- Unified frontend dashboard with full layer interactivity.
- 100% test pass rate across 61 regression and integration test cases.

**Final Verdict:** **PHASE 6 END-TO-END SYSTEM INTEGRATION IS COMPLETE AND VERIFIED READY FOR DEMONSTRATION.**

