# FloodSense-AI: REST API Specification & Endpoint Documentation

**Document Reference:** `API_DOCUMENTATION.md`  
**Base URL:** `http://localhost:8000`  
**Specification Standard:** OpenAPI 3.0 / Swagger UI at `http://localhost:8000/docs`  

---

## 1. System Health & Infrastructure

### `GET /api/health`
Returns runtime status, PyTorch compute device, CUDA availability, model parameter count, and active presets.
- **Request:** None
- **Response `200 OK`:**
  ```json
  {
    "status": "OPERATIONAL",
    "service": "FloodSense-AI Dual-Stream SAR-Optical Engine",
    "device": "CPU Engine (PyTorch Optimized)",
    "cuda_available": false,
    "model_architecture": "DualStreamAttentionUNet (Siamese 4-Stage + CMAF + Attention Gates)",
    "model_parameters": 17452683,
    "weights_calibrated": true,
    "presets_count": 3
  }
  ```

---

## 2. Disaster Presets & Geospatial Analysis

### `GET /api/presets`
Lists all available pre-loaded disaster presets (`chennai_monsoon_2026`, `assam_brahmaputra_surge`, `wayanad_kerala_flashflood`).

### `POST /api/analyze`
Executes full multi-modal pipeline: tensor preprocessing, PyTorch segmentation, NDWI/SAR priors, OSM vector intersection, and A* rescue routing.
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `preset_id` *(string, optional)*: Preset identifier (e.g. `chennai_monsoon_2026`).
  - `pre_optical_file` *(file, optional)*: Pre-flood Sentinel-2 GeoTIFF.
  - `post_sar_file` *(file, optional)*: Post-flood Sentinel-1 SAR GeoTIFF.
  - `bbox_json` *(string, optional)*: JSON array `[min_lat, min_lon, max_lat, max_lon]`.
- **Response `200 OK`:**
  Returns complete payload containing `telemetry`, `visual_layers`, `buildings_summary`, `rescue`, `roads_geojson`, `blocked_roads_geojson`, `rescue_routes_geojson`, `buildings_geojson`, and `export_geojson`.

---

## 3. Physical 2D Flood Simulation Endpoints

### `POST /api/simulate-dynamic`
Initiates an asynchronous 2D shallow-water fluid simulation.
- **Request Body (JSON):**
  ```json
  {
    "preset_id": "chennai_monsoon_2026",
    "scenario_id": "severe_cyclonic"
  }
  ```
- **Response `200 OK`:**
  ```json
  {
    "job_id": "a1b2c3d4",
    "status": "RUNNING",
    "preset_id": "chennai_monsoon_2026",
    "scenario_id": "severe_cyclonic",
    "message": "Physical 2D shallow-water simulation initiated in background."
  }
  ```

### `GET /api/simulation-status/{job_id}`
Polls progress, status, and snapshots of the simulation.
- **Response `200 OK`:**
  ```json
  {
    "job_id": "a1b2c3d4",
    "status": "COMPLETED",
    "progress_pct": 100.0,
    "current_hour": 48,
    "snapshots_count": 9,
    "snapshots": [
      {
        "timestep_hours": 0,
        "label": "T+00 Hours",
        "flooded_area_km2": 0.285,
        "mean_water_depth_m": 0.85,
        "max_water_depth_m": 2.10,
        "water_volume_m3": 489500.0,
        "mass_balance_error_pct": 0.0,
        "depth_distribution_pixels": { "shallow_0_to_30cm": 1420, "moderate_30_to_80cm": 2100, "deep_80_to_150cm": 1800, "critical_over_150cm": 450 },
        "geojson": { "type": "FeatureCollection", "features": [...] }
      }
    ]
  }
  ```

---

## 4. Dynamic Emergency Routing Recalculation

### `POST /api/route-dynamic`
Recalculates NetworkX A* evacuation routes against water depth grids at a specific simulation timestep.
- **Request Body (JSON):**
  ```json
  {
    "preset_id": "chennai_monsoon_2026",
    "timestep_hour": 12,
    "start_coords": [13.035, 80.245],
    "end_coords": [13.055, 80.265],
    "vehicle_type": "amphibious"
  }
  ```
- **Response `200 OK`:**
  ```json
  {
    "preset_id": "chennai_monsoon_2026",
    "timestep_hour": 12,
    "vehicle_type": "amphibious",
    "rescue_summary": {
      "active_safe_routes": 2,
      "shortest_route_km": 3.42,
      "fastest_eta_mins": 9.2,
      "blocked_road_segments": 42
    },
    "safe_routes_geojson": { "type": "FeatureCollection", "features": [...] },
    "severed_roads_geojson": { "type": "FeatureCollection", "features": [...] }
  }
  ```

---

## 5. Situation Reports & Export Formats

### `POST /api/generate-sitrep`
Synthesizes structured NDRF Disaster Situation Report.
- **Request Body (JSON):** Analysis output dictionary.
- **Response `200 OK`:** Structured JSON containing `severity_level`, `executive_summary`, `ndrf_deployment_orders`, `resource_allocation`, and `markdown_report`.

### `POST /api/sitrep/export-markdown`
Returns raw formatted Markdown string for downloading or archiving.

### `POST /api/sitrep/export-geojson`
Returns unified GeoJSON FeatureCollection consolidating flood polygons, building footprints, road hazards, and evacuation routes.

---

## 6. Remote Sensing Data Providers

### `GET /api/data-providers/status`
Reports configuration state, credentials status, and instructions for Copernicus CDSE, Google Earth Engine, and Local Rasters.

### `POST /api/data-providers/search`
Searches available satellite imagery catalog across providers.
