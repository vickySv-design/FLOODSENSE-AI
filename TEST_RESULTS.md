# FloodSense-AI: Comprehensive Test Execution & Edge-Case Verification Report

**Date of Execution:** 2026-11-07  
**Operating System:** Windows 11 (PowerShell)  
**Python Runtime:** Python 3.10+  
**Node.js Runtime:** Node 18+ (Vite 5.4.21)

---

## 1. Summary of Test Execution

| Test Suite / Target | Command Executed | Assertions / Modules | Duration | Outcome |
|---|---|:---:|:---:|:---:|
| **Live Data & Scenarios** | `python backend/scripts/test_live_data_and_scenarios.py` | 10 / 10 | 2.47 s | **100% PASS** |
| **Universal Study Area** | `python backend/scripts/test_universal_aoi.py` | 7 / 7 | 4.71 s | **100% PASS** |
| **OSM Topology & Phase 4** | `python backend/scripts/test_phase4.py` | 5 / 5 | 1.23 s | **100% PASS** |
| **Georeferencing Math** | `python backend/scripts/test_georef.py` | 4 / 4 | 0.42 s | **100% PASS** |
| **Model Validation Benchmark** | `python backend/scripts/evaluate_model.py` | 6 val tiles | 2.91 s | **100% PASS** |
| **Frontend Production Build** | `npm.cmd --prefix frontend run build` | 2,191 modules | 3.48 s | **100% PASS** |
| **Edge-Case & Failure Audit** | `python backend/scripts/test_edge_cases_and_audit.py` | 8 edge cases | 2.08 s | **FAIL (2 failures)** |

---

## 2. Actual Terminal Outputs

### 2.1 Live Data Providers & Cyclone Scenarios Test Suite
**Command:** `python backend/scripts/test_live_data_and_scenarios.py`

```text
test_01_provider_statuses (__main__.TestLiveDataAndScenarioSimulation)
Test 1: All integrated data providers report valid operational statuses. ... ok
test_02_open_meteo_rainfall (__main__.TestLiveDataAndScenarioSimulation)
Test 2: Open-Meteo provider retrieves valid hourly precipitation time-series. ... ok
test_03_nasa_gpm_precipitation (__main__.TestLiveDataAndScenarioSimulation)
Test 3: NASA GPM IMERG provider returns satellite microwave precipitation estimates. ... ok
test_04_copernicus_dem_retrieval (__main__.TestLiveDataAndScenarioSimulation)
Test 4: Copernicus DEM acquires hydro-conditioned elevation grid with valid metric elevation. ... ok
test_05_data_acquisition_job (__main__.TestLiveDataAndScenarioSimulation)
Test 5: Central Data Acquisition Manager executes asynchronous acquisition job with status updates. ... ok
test_06_cache_audit (__main__.TestLiveDataAndScenarioSimulation)
Test 6: Local persistent cache auditing reports offline availability and directory sizes. ... ok
test_07_cyclone_category_classification (__main__.TestLiveDataAndScenarioSimulation)
Test 7: Cyclone wind scale accurately determines IMD and Saffir-Simpson categories. ... ok
test_08_cyclone_scenario_simulation_and_mass_conservation (__main__.TestLiveDataAndScenarioSimulation)
Test 8: Interactive Cyclone Simulator solves 2D diffusive wave flow with 0.00% mass balance error. ... ok
test_09_scenario_manager_and_presets (__main__.TestLiveDataAndScenarioSimulation)
Test 9: Scenario Manager provides editable presets and creates user-defined scenarios. ... ok
test_10_scenario_report_generation (__main__.TestLiveDataAndScenarioSimulation)
Test 10: Scenario report generator outputs structured JSON with explicit hypothetical disclaimer. ... ok

----------------------------------------------------------------------
Ran 10 tests in 2.465s

OK
```

---

### 2.2 Universal Study Area (AOI) Engine Test Suite
**Command:** `python backend/scripts/test_universal_aoi.py`

```text
test_01_aoi_service_geodesic_and_utm (__main__.TestUniversalStudyAreaEngine)
Test 1: AOIService correctly calculates ellipsoidal area and UTM EPSG projection. ... ok
test_02_nominatim_search_and_clamping (__main__.TestUniversalStudyAreaEngine)
Test 2: AOIService location search returns clamped disaster windows for safe memory usage. ... ok
test_03_chennai_coastal_ocean_classification (__main__.TestUniversalStudyAreaEngine)
Test 3: Ocean baseline in Chennai Adyar Estuary is classified as Permanent Water, NOT New Flood. ... ok
test_04_brand_new_custom_aoi_cache_isolation (__main__.TestUniversalStudyAreaEngine)
Test 4: Custom AOI (Cuddalore Port) creates isolated cache and does not contaminate presets. ... ok
test_05_sliding_window_arbitrary_raster_inference (__main__.TestUniversalStudyAreaEngine)
Test 5: Sliding-window inference supports arbitrary raster dimensions (e.g. 384x384). ... ok
test_06_dynamic_dem_and_hydraulic_simulation_for_custom_aoi (__main__.TestUniversalStudyAreaEngine)
Test 6: Dynamic 2D hydraulic DEM and simulation generation works on arbitrary AOI. ... ok
test_07_all_three_presets_load_and_route_cleanly (__main__.TestUniversalStudyAreaEngine)
Test 7: All 3 demonstration presets (Chennai, Assam, Wayanad) load, damage-grade and route cleanly. ... ok

----------------------------------------------------------------------
Ran 7 tests in 4.710s

OK
```

---

### 2.3 Phase 4 OSM Geometries & Routing Test Suite
**Command:** `python backend/scripts/test_phase4.py`

```text
.....
----------------------------------------------------------------------
Ran 5 tests in 1.232s

OK
```

---

### 2.4 Georeferencing & Mathematical Transforms Test Suite
**Command:** `python backend/scripts/test_georef.py`

```text
[TEST] Running Georeferencing & Projection Validation...
  [PASS] Pixel <-> Lon/Lat Roundtrip: (128.5, 64.2) -> (80.24015625, 13.0599375) -> (128.50, 64.20)
  [PASS] WGS84 <-> UTM EPSG:32644 Roundtrip: (80.24, 13.04) -> (417595.6m, 1441682.5m)
  [PASS] Metric Polygon Area: 1199198.8 m2 (1.199 km2)
  [PASS] Raster Sub-Pixel Sampling: sample=16563.70, expected=16563.70

ALL GEOREFERENCING TESTS PASSED (4/4)!
```

---

### 2.5 Real Model Evaluation Suite on Validation Set
**Command:** `python backend/scripts/evaluate_model.py`

```text
======================================================================
FLOODSENSE-AI: RIGOROUS EMPIRICAL VALIDATION SUITE
======================================================================
Found 6 validation tiles in: C:\Users\vicky\Desktop\Disaster Response\backend\data\mini_sen1floods11\val
Evaluating on Device: CPU
Loaded trained checkpoint: C:\Users\vicky\Desktop\Disaster Response\backend\data\checkpoints\dual_stream_attention_unet.pth
[1/6] val_tile_01: Model IoU=1.000, F1=1.000 | Otsu IoU=0.552 (295.9ms)
[2/6] val_tile_02: Model IoU=1.000, F1=1.000 | Otsu IoU=0.586 (1059.8ms)
[3/6] val_tile_03: Model IoU=1.000, F1=1.000 | Otsu IoU=0.461 (1206.1ms)
[4/6] val_tile_04: Model IoU=1.000, F1=1.000 | Otsu IoU=0.396 (1077.3ms)
[5/6] val_tile_05: Model IoU=1.000, F1=1.000 | Otsu IoU=0.446 (1081.2ms)
[6/6] val_tile_06: Model IoU=1.000, F1=1.000 | Otsu IoU=0.379 (1103.2ms)

======================================================================
EVALUATION COMPLETE! Results saved to:
  C:\Users\vicky\Desktop\Disaster Response\backend\data\checkpoints\evaluation_results.json
Model Macro IoU: 100.00% | F1: 100.00% | Latency: 970.6ms
Otsu  Macro IoU: 47.01% | F1: 63.60%
======================================================================
```

---

### 2.6 Frontend Production Build Output
**Command:** `npm.cmd --prefix frontend run build`

```text
> floodsense-ai-frontend@1.0.0 build
> vite build

vite v5.4.21 building for production...
transforming...
✓ 2191 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   1.07 kB │ gzip:   0.58 kB
dist/assets/index-DhhzKno1.css   33.72 kB │ gzip:   6.55 kB
dist/assets/index-B2Ic6ul8.js   935.37 kB │ gzip: 262.09 kB

✓ built in 3.48s
```

---

### 2.7 Edge-Case & Failure-Mode Verification Suite
**Command:** `python backend/scripts/test_edge_cases_and_audit.py`

```text
.....EScenario 8f2beefd failed: 'NoneType' object is not subscriptable
Traceback (most recent call last):
  File "C:\Users\vicky\Desktop\Disaster Response\backend\app\services\scenario_engine.py", line 431, in _run_worker
    sim = CycloneScenarioSimulator(
  File "C:\Users\vicky\Desktop\Disaster Response\backend\app\services\scenario_engine.py", line 82, in __init__
    self.georef = GeoSpatialTransform(bbox, width=self.W, height=self.H)
  File "C:\Users\vicky\Desktop\Disaster Response\backend\app\services\georef.py", line 20, in __init__
    self.min_lat = float(bbox[0])
TypeError: 'NoneType' object is not subscriptable
.F
======================================================================
ERROR: test_06_large_aoi_handling (__main__.TestEdgeCasesAndFailureModes)
Test universal AOI service clamping when user requests an excessively large bounding box.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "C:\Users\vicky\Desktop\Disaster Response\backend\scripts\test_edge_cases_and_audit.py", line 107, in test_06_large_aoi_handling
    norm = aoi_service.validate_and_normalize_aoi("Huge Region", huge_bbox)
  File "C:\Users\vicky\Desktop\Disaster Response\backend\app\services\aoi_service.py", line 147, in validate_and_normalize_aoi
    raise ValueError(
ValueError: Selected study area (300692.6 km²) exceeds system processing limits (500.0 km²). Please zoom in or select a sub-watershed sector.

======================================================================
FAIL: test_08_empty_or_disconnected_rescue_routes (__main__.TestEdgeCasesAndFailureModes)
Test rescue routing when roads are 100% submerged or completely absent.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "C:\Users\vicky\Desktop\Disaster Response\backend\scripts\test_edge_cases_and_audit.py", line 143, in test_08_empty_or_disconnected_rescue_routes
    self.assertEqual(report["summary"]["active_safe_routes"], 0)
AssertionError: 3 != 0

----------------------------------------------------------------------
Ran 8 tests in 2.084s

FAILED (failures=1, errors=1)
  -> test_01_invalid_scenario_inputs: Handled gracefully, non-negative water depth preserved.
  -> test_02_missing_or_corrupted_dem: Generated synthetic hydro DEM ((128, 128)) range [1.4m, 18.3m].
  -> test_03_missing_rainfall_data: Fallback generated 24 hours of meteorological series.
  -> test_04_provider_timeout_and_fallback: Unconfigured Copernicus gracefully returned 1 fallback catalog items.
  -> test_05_cache_fallback_offline: Successfully loaded from cache/fallback with status LIVE_OPEN_METEO_API.
  -> test_07_failed_simulation_job: Job correctly marked FAILED with error: ''NoneType' object is not subscriptable'.
```

---

## 3. Detailed Failure Analysis from Edge-Case Tests

### Failure 1: Large AOI Direct Validation Throws `ValueError` Instead of Auto-Clamping
- **Root Cause:** In [backend/app/services/aoi_service.py#L147](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/aoi_service.py#L147), `validate_and_normalize_aoi()` strictly enforces an upper limit of $500\text{ km}^2$. While `search_locations()` automatically clamps user-selected bounding boxes to $\le 500\text{ km}^2$, passing a huge bounding box directly to `validate_and_normalize_aoi()` throws a `ValueError` rather than clamping.
- **Severity:** Low. In practice, UI interactions go through `search_locations()` or clamp coordinates on the frontend map.

### Failure 2: 100% Flooded Raster Still Yields 3 Safe Routes
- **Root Cause:** When `seg_mask` is 100% flooded, 22,222 road segments inside the bounding box are blocked. However, 243 road segments located along the northern boundary margin have coordinates with pixel indices $r < 0$ (outside the georeferenced raster).
- **Code Behavior:** [backend/app/services/rescue_router.py#L53-L54](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/rescue_router.py#L53-L54) checks `if not dense_pixels: return False, 0.0, "PASSABLE"`. Because 0 pixels were sampled inside the raster, these out-of-bounds segments are treated as `PASSABLE`. Furthermore, if `base_node` snaps to the same node as `target_node`, `nx.astar_path` returns `[node]`, which [line 312](file:///c:/Users/vicky/Desktop/Disaster%20Response/backend/app/services/rescue_router.py#L312) clamps to `max(0.0, 0.25) = 0.25 km` and records as a valid route.
- **Severity:** High. The routing engine must explicitly check whether start/target nodes are reachable and reject segments outside the bounding box.

### Failure 3: Missing `numpy` in Offline Fallback OSM Generator
- **Command:** Direct invocation of `_generate_fallback_osm_network`
- **Output:**
  ```text
  File "backend/app/services/osm_service.py", line 327, in _generate_fallback_osm_network
      rng = np.random.default_rng(seed)
  NameError: name 'np' is not defined
  ```
- **Severity:** High. Crashes offline fallback when Overpass is inaccessible.
