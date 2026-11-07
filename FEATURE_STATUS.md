# FloodSense-AI: Feature Status Matrix & Implementation Roadmap

**Status Date:** 2026-11-07  
**Evaluation Scope:** Complete Full-Stack Geospatial Repository  
**Classification Categories:**
- **Functional:** Fully implemented with real processing and passing tests.
- **Partially Functional:** Implemented, but has specific edge-case defects, artificial constraints, or missing data links.
- **Synthetic / Stub:** Uses mathematical procedural synthesis, simulated stubs, or mock returns.
- **Missing:** Documented or planned, but has no underlying implementation.

---

## 1. Complete System Feature Status Matrix

| System Module / Feature | Status | Primary Code Reference | Verified Capabilities | Gaps, Limitations & Stubs |
|---|:---:|---|---|---|
| **Dual-Stream Attention U-Net** | **Functional** | [attention_unet.py#L141](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/models/attention_unet.py#L141) | Real PyTorch convolutions, CMAF fusion, soft attention gates, $14.1\text{M}$ weights. | Sliding-window inference works, but metrics are clamped in inference service. |
| **Multi-Tile Sliding-Window Inference** | **Functional** | [inference_engine.py#L228](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/inference_engine.py#L228) | Seamless 2D Hann cosine blending for arbitrary raster dimensions ($H, W > 256$). | Stride fixed at 192px with 256px window; no GPU batching across tiles. |
| **Universal Study Area (AOI) Engine** | **Functional** | [aoi_service.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/aoi_service.py) | Geodesic area calculation (`pyproj.Geod`), UTM projection detection, auto-clamping. | Direct call to `validate_and_normalize_aoi` throws on $>500\text{ km}^2$ rather than clamping. |
| **Coastal Ocean Discrimination** | **Functional** | [preprocessor.py#L90](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/preprocessor.py#L90) | Classifies coastal ocean (Bay of Bengal) as Class 1 permanent baseline. | Uses geographic longitude threshold combined with low NIR reflectance. |
| **Real OpenStreetMap Vector Ingestion** | **Functional** | [osm_service.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/osm_service.py) | Live Overpass querying, parsing of highways and buildings, local cache. | Missing `import numpy as np` in offline synthetic fallback at line 327. |
| **Open-Meteo Weather API Client** | **Functional** | [open_meteo_provider.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/data_providers/open_meteo_provider.py) | Live HTTP querying, returns genuine 24h/72h rainfall, wind, and temp time-series. | Keyless; rate-limited to 10,000 daily calls per IP. |
| **Copernicus CDSE Satellite Queries** | **Partially Functional** | [copernicus_provider.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/data_providers/copernicus_provider.py) | OData v1 catalogue search with polygon WKT and cloud-cover filters. | Band download writes `.meta` JSON stubs instead of streaming multi-gigabyte rasters. |
| **Copernicus DEM (GLO-30/90)** | **Synthetic / Stub** | [copernicus_provider.py#L248](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/data_providers/copernicus_provider.py#L248) | Caches elevation grid and generates hydro-conditioned slope matrices. | Generates seed-based procedural elevation rather than querying real OpenTopography tiles. |
| **NASA GPM IMERG Rainfall** | **Partially Functional** | [nasa_gpm_provider.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/data_providers/nasa_gpm_provider.py) | Earthdata authentication builder and HDF5/GeoTIFF ingestion routines. | Drops to fallback series unless `NASA_EARTHDATA_USERNAME` and password are supplied. |
| **2D Hydrodynamic Fluid Simulator** | **Functional** | [simulation_engine.py#L189](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/simulation_engine.py#L189) | Solves 2D diffusive wave equation ($\partial h/\partial t + \nabla \cdot q = R - I - D$) with Manning flow. | Closed/reflective wall boundaries; water cannot flow out of domain. |
| **Cyclone & Rainfall Scenario Engine** | **Functional** | [scenario_engine.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/scenario_engine.py) | Models cyclone eye motion, Rankine vortex wind, and spatial rainfall decay. | Mass balance error percentage output is forced to $0.00\%$ via code clamps. |
| **Building Damage Exposure Grading** | **Functional** | [damage_analyzer.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/damage_analyzer.py) | 4-tier structural submergence grading, affected population, economic loss. | Downsamples buildings to $\le 250$ for frontend Leaflet rendering performance. |
| **A* Rescue Evacuation Pathfinder** | **Partially Functional** | [rescue_router.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/rescue_router.py) | Navigates real OSM road graph; avoids Class 2 flood pixels; preserves bridges. | Deems road segments outside the raster bounding box passable; allows 0.25km single-node paths. |
| **Dynamic Route Recalculation** | **Synthetic / Stub** | [main.py#L1031](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/main.py#L1031) | Recalculates routes via REST API. | Endpoint `/api/scenarios/{id}/routes/recalculate` hardcodes a synthetic box `[100:150, 100:150]`. |
| **NDRF Action SitRep Generator** | **Functional** | [sitrep_generator.py](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/sitrep_generator.py) | Gemini 2.5 Flash + offline deterministic expert system fallback. | Fully operational; produces structured JSON, Markdown, and logistics requirements. |
| **Report Export (Markdown / GeoJSON)** | **Functional** | [main.py#L775](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/main.py#L775) | Exports formatted Markdown flash bulletins and GeoJSON FeatureCollections. | Fully operational and downloadable. |
| **PDF Export** | **Partially Functional** | [SitRepModal.jsx#L50](file:///c:/Users/vicky/Desktop/Disaster%20Response/frontend/src/components/SitRepModal.jsx#L50) | Prints via browser native print dialog (`window.print()`). | No headless server-side binary PDF generator (e.g. ReportLab, Weasyprint). |

---

## 2. Identified Deficiencies Requiring Remediation

### High Priority
1. **Scenario Route Recalculation Disconnect:** Connect `/api/scenarios/{scenario_id}/routes/recalculate` to the simulated water depth matrix $h$ of the requested time step, replacing the hardcoded `seg_mask[100:150, 100:150]` box.
2. **Dynamic Route Timestep Parameter:** In `/api/route-dynamic`, evaluate the simulation snapshot at `req.timestep_hour` instead of querying the static baseline `ground_truth.npy`.
3. **Missing `numpy` Import in `osm_service.py`:** Add `import numpy as np` at line 10 of `backend/app/services/osm_service.py` to prevent crashes when offline fallback executes.
4. **Out-of-Bounds Road Segment Handling:** In `rescue_router.py`, reject or flag road segments whose coordinates fall outside the raster bounding box instead of treating them as `PASSABLE`.

### Medium Priority
5. **Ablation Benchmark Realism:** Update documentation to explain that $100.00\%$ validation mIoU is achieved on synthetic calibration tiles, whereas real-world Sen1Floods11 achieves $74\% \to 81\%$ mIoU.
6. **Inference Metric Transparency:** Remove artificial metric clamping (`min(max(calc_miou, 0.865), 0.962)`) in `backend/app/services/inference_engine.py` and report raw calculated segmentation metrics.
7. **Mass Conservation Reporting:** Display true calculated mass balance error ($<0.05\%$) rather than clamping or hardcoding `0.00%`.

### Low Priority
8. **Server-Side PDF Generation:** Add an automated backend PDF generation endpoint (using ReportLab) to produce official downloadable PDF briefs without relying on browser `window.print()`.
9. **Open Hydrodynamic Boundaries:** Add open downstream drainage boundaries to the 2D fluid solver so flood waters can leave the domain in coastal or low-lying grid cells.

---

## 3. Prioritized Implementation Backlog

### Phase 1: Critical Bug Fixes & Wiring (Immediate)
- [ ] Fix `NameError: np not defined` in `backend/app/services/osm_service.py`.
- [ ] Wire simulated flood depths from active scenario runs into `/api/scenarios/{scenario_id}/routes/recalculate`.
- [ ] Wire dynamic time-step simulation snapshots into `/api/route-dynamic`.
- [ ] Fix out-of-bounds road segment vulnerability in `backend/app/services/rescue_router.py`.

### Phase 2: Scientific Accuracy & Metric Integrity (Next)
- [ ] Remove artificial metric clamping in `backend/app/services/inference_engine.py`.
- [ ] Report real calculated mass balance error in `scenario_engine.py`.
- [ ] Add 1-minute to 3-minute sustained wind conversion factor ($0.88$) in IMD vs. Saffir-Simpson classification.

### Phase 3: Live Data Provider Expansion (Future)
- [ ] Integrate OpenTopography free API key support for real 30m Copernicus GLO-30 GeoTIFF downloads.
- [ ] Implement headless server-side PDF generation via ReportLab for disaster situation reports.
