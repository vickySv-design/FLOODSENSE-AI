# FloodSense-AI — Phase 6: End-to-End System Integration Architecture Audit

**Project:** FloodSense-AI (B.E. CSE Final-Year Capstone)  
**Date:** November 8, 2026  
**Document Version:** 1.0 — Pre-Integration System Audit  

---

## 1. System Overview & Audit Objective

The objective of Phase 6 is to unite all previously developed independent and semi-independent modules into a single, cohesive, locally executable flood emergency workflow. This audit systematically catalogs:
- Existing backend services, models, and scripts.
- Existing REST API endpoints.
- Existing frontend UI pages, components, and client methods.
- Module dependencies and inter-service data flows.
- Missing integration points and potential execution bottlenecks.
- Recommended orchestration sequence for the unified pipeline.

---

## 2. Existing Backend Modules Inventory

### 2.1 Satellite Ingestion & Providers (`backend/app/services/data_providers/`)
- `CopernicusDataSpaceProvider` (`copernicus_provider.py`): Programmatic interface to ESA CDSE STAC/OData API with local caching and offline fallback synthesis.
- `GoogleEarthEngineProvider` (`gee_provider.py`): Google Earth Engine interface querying `COPERNICUS/S2_SR` and `COPERNICUS/S1_GRD`.
- `LocalRasterProvider` (`local_raster_provider.py`): Ingestion of local GeoTIFF composites (e.g. Coimbatore 2025).
- `DataAcquisitionManager` (`data_acquisition_manager.py`): Central coordinator for multi-modal satellite pair acquisition.
- `RasterValidator` (`raster_validator.py`): Rigorous physical GeoTIFF verification (CRS, transform, valid bands, non-zero checks).

### 2.2 Preprocessing & Feature Engineering (`backend/app/services/`)
- `preprocessor.py`:
  - `preprocess_multimodal_pair(opt_arr, sar_arr, device)`: Normalization, optical CLAHE (clip=2.0), NDWI calculation, SAR $\sigma^0$ dB conversion ([-30, 0] dB), SAR Lee-Sigma $5\times 5$ speckle filter.
  - Generates 6-channel fused input or dual-stream tensor representations `[B, 4, H, W]` and `[B, 2, H, W]`.

### 2.3 Deep Learning Models & Inference (`backend/app/models/`, `backend/app/services/`)
- `LightweightDualStreamAttentionUNet` (`lightweight_attention_unet.py`): 8.24M parameters, dual-stream encoders (S2 optical 4ch, S1 SAR 2ch), Cross-Modal Attention Fusion (CMAF).
- Checkpoints on disk:
  - `backend/data/checkpoints/dual_stream_attention_unet.pth` (162.6 MB)
  - `backend/outputs/sen1floods11_training/best_checkpoint.pth` (99.2 MB, Epoch 28 best validation checkpoint)
- `InferenceEngine` (`inference_engine.py`):
  - Sliding-window inference with 2D Hann cosine overlap blending.
  - `export_geotiff_flood_mask()`: Exports predicted masks as georeferenced GeoTIFF with CRS and Affine transform.

### 2.4 Temporal SAR Change Detection (`backend/scripts/`)
- `analyze_flood_event.py` & `run_phase4e_multichip_validation.py`:
  - Performs temporal SAR difference calculation: $\Delta \text{VV} = \text{VV}_{\text{post}} - \text{VV}_{\text{pre}}$.
  - Fixed calibrated threshold: $\Delta \text{VV} < -5.0\text{ dB}$.
  - Geodesic WGS84 pixel area integration and domain decomposition.

### 2.5 GIS Infrastructure & Rescue Routing (`backend/app/services/`)
- `GeoSpatialTransform` (`georef.py`): Dynamic UTM zone detection (e.g. `EPSG:32644`), sub-pixel affine transformations, polygon windowed rasterization with `rasterio.features.rasterize`.
- `damage_analyzer.py`: `analyze_building_exposure()` classifies building footprints into `HIGH_EXPOSURE` ($\ge 50\%$), `MODERATE_EXPOSURE` ($10-50\%$), `LOW_OR_NO_DETECTED_EXPOSURE` ($< 10\%$), and `UNKNOWN`.
- `rescue_router.py`:
  - `sample_road_hazard_metrics()`: 10m interval sampling along LineStrings.
  - `classify_road_hazards()`: Categories: `POTENTIAL_HAZARD` ($\ge 30\%$), `CAUTION` ($5-30\%$), `NO_DETECTED_INTERSECTION` ($< 5\%$).
  - `build_road_graph()`: Constructs `networkx.MultiDiGraph` preserving parallel edges and respecting OSM `oneway` tags.
  - `calculate_astar_evacuation_route()`: A* pathfinding with Euclidean metric heuristic, strict hazard exclusion or cost penalties, and formula-driven ETA ($\text{ETA} = (d / 1000) / 25 \times 60$).

---

## 3. Existing REST API Endpoints

### 3.1 Core System & Presets
- `GET /api/health`: System health, CUDA status, active device.
- `GET /api/presets`: Metadata and bounding boxes for available disaster scenarios.
- `POST /api/analyze`: Multipart/preset flood analysis returning raw masks and scenario statistics.

### 3.2 Satellite & GEE Integration
- `POST /api/gee-fetch`: Query and fetch satellite tiles from Google Earth Engine.
- `POST /api/satellite/acquire`: Programmatic multi-modal acquisition via CDSE/GEE.
- `GET /api/satellite/local/status`: Local GeoTIFF ingestion status.
- `POST /api/satellite/local/infer`: Run inference on local GeoTIFF files.

### 3.3 GIS Infrastructure & Rescue Routing (Phase 5 Endpoints)
- `POST /api/gis/exposure-analysis`: Compute building exposure breakdown and road hazards on matching study area raster.
- `POST /api/gis/evacuation-route`: Compute A* evacuation corridor between start and destination coordinates.

### 3.4 Reporting & Simulation
- `POST /api/generate-sitrep`: Synthesize NDRF tactical situation reports.
- `POST /api/scenarios/run`: Trigger hydrodynamic 2D simulation scenarios.

---

## 4. Existing Frontend Pages & Client Architecture

### 4.1 Pages (`frontend/src/pages/`)
1. `DashboardPage.jsx`: Main telemetry overview, quick action buttons, sensor metadata.
2. `SatelliteAnalysisPage.jsx`: Dual-stream model architecture details, multi-band inspection, manual inference trigger.
3. `FloodSimulationPage.jsx`: 2D diffusive-wave shallow water hydrodynamic simulator.
4. `RescueInfrastructurePage.jsx`: Live A* route calculation, building exposure table, road hazard layers, corridor presets.
5. `DisasterReportsPage.jsx`: AI-synthesized tactical SitRep generation and export.
6. `DataSourcesPage.jsx`: Satellite provider health, OSM caches, cache flushing.
7. `SettingsPage.jsx`: Hardware diagnostics, default scenario configuration.

### 4.2 Shared Components
- `SharedMap.jsx` & `MapViewport.jsx`: Leaflet map with layer visibility toggles, tile layers, and GeoJSON overlays.
- `AppLayout.jsx` & `Sidebar.jsx`: Modern responsive sidebar navigation.
- `api.js`: Axios HTTP client with 45s timeout.

---

## 5. Module Dependencies & Inter-Service Data Flows

```
[Study Area Preset / AOI]
        │
        ▼
[DataAcquisitionManager] ─── (CDSE / GEE / Local Cache)
        │
        ▼
[RasterValidator] ────────── (Physical GeoTIFF Integrity)
        │
        ▼
[Preprocessor] ───────────── (CLAHE, NDWI, dB Calib, Speckle)
        │
        ▼
[InferenceEngine] ────────── (LightweightDualStreamAttentionUNet)
        │
        ├──► [Export GeoTIFF] ──► flood_mask_<id>.tif
        │
        ├──► [Temporal SAR Analysis] (if pre-event SAR exists)
        │
        ├──► [GeoSpatialTransform] ──► (UTM Zone 44N Metric Grid)
        │           │
        │           ├──► [damage_analyzer] ──► Building Exposure
        │           │
        │           └──► [rescue_router] ────► Road Hazards & MultiDiGraph A*
        ▼
[Pipeline Results & Provenance] ──► Unified Dashboard
```

---

## 6. Missing Integration Points & Identified Conflicts

1. **Fragmented Execution:** Previously, acquiring satellite imagery, running inference, evaluating building damage, and routing rescue vehicles required navigating through 3 different pages (`SatelliteAnalysisPage`, `RescueInfrastructurePage`, `DataSourcesPage`) or calling disconnected endpoints.
2. **Lack of a Unified Orchestrator:** No single backend service tracked the lifecycle of an end-to-end disaster response run from raw imagery to candidate evacuation corridors.
3. **Execution State Persistence:** Long-running inference and raster processing ran synchronously, risking HTTP timeouts on complex rasters. A lightweight asynchronous background execution manager with stage-level progress tracking is needed.
4. **Graceful Fallbacks:** If pre-event satellite data is unavailable, temporal change detection must report `UNAVAILABLE` rather than crashing the pipeline. Similarly, if OSM data is missing for a remote region, infrastructure grading must report `NO_OSM_DATA` while preserving flood prediction rasters.

---

## 7. Recommended Integration Sequence (Phase 6 Roadmap)

1. **Orchestration Service (`floodsense_pipeline.py`):**
   - Create `FloodSensePipelineOrchestrator` managing thread-safe background runs.
   - Implement stage transitions: `DATA_ACQUISITION` $\to$ `PREPROCESSING` $\to$ `FLOOD_INFERENCE` $\to$ `TEMPORAL_ANALYSIS` $\to$ `BUILDING_EXPOSURE` $\to$ `ROAD_HAZARD_ANALYSIS` $\to$ `RESCUE_ROUTING` $\to$ `FINALIZATION`.
   - Record start time, end time, duration, status, warnings, errors, and stage outputs.
2. **Pipeline REST API Endpoints (`main.py`):**
   - `POST /api/pipeline/run`: Launch execution thread, return `execution_id`.
   - `GET /api/pipeline/status/{execution_id}`: Poll status, stage progress, and elapsed time.
   - `GET /api/pipeline/result/{execution_id}`: Retrieve completed pipeline payload.
   - `GET /api/pipeline/history`: List recent pipeline runs.
   - `GET /api/pipeline/health`: Pipeline orchestrator diagnostics.
3. **Frontend Unified Dashboard (`FloodSensePipelinePage.jsx`):**
   - Add "End-to-End Pipeline" to sidebar and hash router.
   - Interactive study area selection, analysis stage toggles, live progress tracker.
   - Unified Leaflet map displaying satellite baseline, flood prediction, permanent water, exposed buildings, road hazards, and candidate evacuation routes on a single canvas.
   - Real-time telemetry cards and provenance banners.
4. **Automated Test Suite (`test_phase6_pipeline.py`):**
   - 15+ automated unit and integration tests covering end-to-end execution, stage skipping, error handling, partial failure output preservation, and regression suites.

