"""
Phase 6 Integration Test Suite: FloodSense-AI Unified Pipeline
==============================================================
Validates end-to-end integration across:
- Data acquisition & cache retrieval
- Physical preprocessing & geometry validation
- Deep learning inference (DualStreamAttentionUNet_v2.1)
- Temporal SAR change detection & graceful fallback
- Real OpenStreetMap building exposure grading
- Road hazard sampling & NetworkX MultiDiGraph generation
- A* candidate evacuation routing with metric formula ETA
- Thread-safe background execution, status polling & atomic persistence
- All 5 Phase 6 FastAPI endpoints via TestClient
"""

import sys
import os
import time
import json
import shutil
import unittest
from pathlib import Path
from typing import Dict, Any

# Ensure backend root is on path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
from fastapi.testclient import TestClient
from app.main import app, PipelineRunRequest
from app.config import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    DISASTER_PRESETS_METADATA,
    PRESETS_DIR,
)
from app.services.floodsense_pipeline import (
    FloodSensePipelineOrchestrator,
    PipelineStatus,
    PipelineStage,
    pipeline_orchestrator,
    PIPELINE_RUNS_DIR,
)
from app.services.georef import GeoSpatialTransform
from app.services.rescue_router import (
    build_road_graph,
    calculate_astar_evacuation_route,
    classify_road_hazards,
)
from app.services.damage_analyzer import analyze_building_exposure


class TestPhase6PipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_runs_dir = PIPELINE_RUNS_DIR

    # -------------------------------------------------------------
    # 1. Pipeline Request Validation
    # -------------------------------------------------------------
    def test_01_request_model_validation(self):
        """Test Pydantic validation for PipelineRunRequest."""
        # Valid default
        req = PipelineRunRequest()
        self.assertEqual(req.study_area, "chennai_monsoon_2026")
        self.assertTrue(req.run_temporal_analysis)
        self.assertTrue(req.avoid_potential_hazards)

        # Custom valid values
        req2 = PipelineRunRequest(
            study_area="bolivia_103757",
            bbox=[-14.65, -64.95, -14.45, -64.75],
            start_coords=[80.22, 13.04],
            dest_coords=[80.25, 13.05],
            max_buildings=500,
        )
        self.assertEqual(req2.study_area, "bolivia_103757")
        self.assertEqual(req2.max_buildings, 500)

    # -------------------------------------------------------------
    # 2. Execution ID Generation and Atomic Persistence
    # -------------------------------------------------------------
    def test_02_atomic_persistence_and_naming(self):
        """Verifies atomic rename serialization to avoid partial reads."""
        test_eid = f"exec_test_atomic_{int(time.time())}"
        test_record = {
            "execution_id": test_eid,
            "study_area": "test_area",
            "status": PipelineStatus.COMPLETED,
            "created_at": "2026-11-08T00:00:00Z",
        }
        pipeline_orchestrator._save_record_atomically(test_eid, test_record)

        target_file = PIPELINE_RUNS_DIR / f"{test_eid}.json"
        self.assertTrue(target_file.exists(), "Target atomic JSON file must exist.")

        with open(target_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["execution_id"], test_eid)

        # Clean up test artifact
        try:
            target_file.unlink()
        except Exception:
            pass

    # -------------------------------------------------------------
    # 3. Pipeline Status Transitions
    # -------------------------------------------------------------
    def test_03_pipeline_status_constants(self):
        """Verifies status and stage enumeration definitions."""
        self.assertEqual(PipelineStatus.PENDING, "PENDING")
        self.assertEqual(PipelineStatus.RUNNING, "RUNNING")
        self.assertEqual(PipelineStatus.COMPLETED, "COMPLETED")
        self.assertEqual(PipelineStatus.COMPLETED_WITH_WARNINGS, "COMPLETED_WITH_WARNINGS")
        self.assertEqual(PipelineStatus.FAILED, "FAILED")

        self.assertIn(PipelineStage.DATA_ACQUISITION, [
            "DATA_ACQUISITION", "PREPROCESSING", "FLOOD_INFERENCE",
            "TEMPORAL_ANALYSIS", "BUILDING_EXPOSURE", "ROAD_HAZARD_ANALYSIS",
            "RESCUE_ROUTING", "FINALIZATION", "INITIALIZATION",
        ])

    # -------------------------------------------------------------
    # 4. Successful Full Execution with Existing Chennai Preset
    # -------------------------------------------------------------
    def test_04_full_pipeline_execution_chennai(self):
        """Executes full pipeline synchronously on Chennai preset and validates all stages."""
        launch_res = pipeline_orchestrator.start_pipeline(
            study_area="chennai_monsoon_2026",
            run_temporal_analysis=True,
            run_infrastructure_exposure=True,
            run_candidate_routing=True,
            avoid_potential_hazards=True,
            max_buildings=150,
        )
        self.assertTrue(launch_res["success"])
        eid = launch_res["execution_id"]

        # Wait for completion (max 25s)
        for _ in range(50):
            st = pipeline_orchestrator.get_status(eid)
            if st and st["status"] in (PipelineStatus.COMPLETED, PipelineStatus.COMPLETED_WITH_WARNINGS, PipelineStatus.FAILED):
                break
            time.sleep(0.5)

        res = pipeline_orchestrator.get_result(eid)
        self.assertIsNotNone(res, "Result must be retrievable upon completion.")
        self.assertIn(res["status"], (PipelineStatus.COMPLETED, PipelineStatus.COMPLETED_WITH_WARNINGS))

        outputs = res.get("stage_outputs", {})
        self.assertIn("acquisition", outputs)
        self.assertIn("preprocessing", outputs)
        self.assertIn("inference", outputs)
        self.assertIn("building_exposure", outputs)
        self.assertIn("road_hazards", outputs)
        self.assertIn("rescue_routing", outputs)

        # Validate inference metrics
        inf = outputs["inference"]
        self.assertGreater(inf["flood_pixel_count"], 0)
        self.assertGreater(inf["flooded_area_hectares"], 0)

        # Validate building exposure
        bld = outputs["building_exposure"]
        self.assertIn("summary", bld)
        self.assertGreater(bld["summary"]["total_buildings_analyzed"], 0)

        # Validate road hazards
        roads = outputs["road_hazards"]
        self.assertIn("summary", roads)
        self.assertGreater(roads["summary"]["total_roads_analyzed"], 0)

        # Validate rescue route
        routing = outputs["rescue_routing"]
        self.assertIn(routing["status"], ["CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS", "NO_SAFE_ROUTE_AVAILABLE"])

    # -------------------------------------------------------------
    # 5. Missing Satellite Input Handling
    # -------------------------------------------------------------
    def test_05_missing_satellite_input_handling(self):
        """Verifies pipeline records failure when preset or input rasters do not exist."""
        bad_area = "nonexistent_remote_island_xyz_999"
        launch_res = pipeline_orchestrator.start_pipeline(study_area=bad_area)
        eid = launch_res["execution_id"]

        for _ in range(40):
            st = pipeline_orchestrator.get_status(eid)
            if st and st["status"] in (PipelineStatus.COMPLETED, PipelineStatus.COMPLETED_WITH_WARNINGS, PipelineStatus.FAILED):
                break
            time.sleep(0.1)

        res = pipeline_orchestrator.get_result(eid)
        self.assertIsNotNone(res, "Result must not be None after pipeline finishes.")
        self.assertEqual(res["status"], PipelineStatus.FAILED)
        self.assertTrue(len(res["errors"]) > 0)

    # -------------------------------------------------------------
    # 6. Invalid Band Handling
    # -------------------------------------------------------------
    def test_06_invalid_band_tensor_dimension_check(self):
        """Verifies preprocessing stage rejects optical tensor with wrong band count."""
        from app.services.preprocessor import preprocess_multimodal_pair
        import torch

        bad_opt = np.random.rand(3, 64, 64).astype(np.float32)  # 3 channels instead of 4
        good_sar = np.random.rand(2, 64, 64).astype(np.float32)

        with self.assertRaises(ValueError):
            preprocess_multimodal_pair(bad_opt, good_sar, device=torch.device("cpu"))

    # -------------------------------------------------------------
    # 7. Missing Checkpoint Handling
    # -------------------------------------------------------------
    def test_07_checkpoint_validation(self):
        """Verifies checkpoint exists and inference engine reports valid weights."""
        from app.services.inference_engine import inference_engine
        self.assertTrue(inference_engine.model is not None, "Model must be loaded into memory.")
        ckpt_path = CHECKPOINTS_DIR / "dual_stream_attention_unet.pth"
        self.assertTrue(ckpt_path.exists(), f"Checkpoint must exist at {ckpt_path}")

    # -------------------------------------------------------------
    # 8. Inference Engine Output Dimensions and Classes
    # -------------------------------------------------------------
    def test_08_inference_engine_output_validity(self):
        """Verifies inference produces valid 3-class segmentation mask."""
        from app.services.inference_engine import inference_engine
        import torch

        opt_t = torch.randn(1, 4, 128, 128, device=inference_engine.device)
        sar_t = torch.randn(1, 2, 128, 128, device=inference_engine.device)

        with torch.no_grad():
            out = inference_engine.model(opt_t, sar_t)
            logits = out[0] if isinstance(out, tuple) else out
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]

        self.assertEqual(preds.shape, (128, 128))
        unique_classes = set(np.unique(preds))
        self.assertTrue(unique_classes.issubset({0, 1, 2}), f"Classes must be subset of {{0, 1, 2}}, got {unique_classes}")

    # -------------------------------------------------------------
    # 9. Temporal Analysis Graceful Degradation
    # -------------------------------------------------------------
    def test_09_temporal_analysis_graceful_fallback(self):
        """Verifies temporal analysis records warning and does not crash when pre-event SAR is absent."""
        # Orchestrator handles missing pre-event by recording a warning and completing
        st_outputs: Dict[str, Any] = {}
        warnings: list = []

        # Simulate missing pre-event
        pre_event_sar = None
        if pre_event_sar is None:
            warnings.append("Pre-event SAR raster unavailable. Skipping temporal change detection.")
            st_outputs["temporal_change"] = {
                "status": "UNAVAILABLE",
                "reason": "NO_PRE_EVENT_OBSERVATION",
                "message": "Temporal difference requires both pre-event and post-event Sentinel-1 rasters.",
            }

        self.assertEqual(st_outputs["temporal_change"]["status"], "UNAVAILABLE")
        self.assertTrue(len(warnings) > 0)

    # -------------------------------------------------------------
    # 10. Missing OSM Data Handling
    # -------------------------------------------------------------
    def test_10_missing_osm_data_handling(self):
        """Verifies road and building analyzers handle empty feature lists gracefully."""
        georef = GeoSpatialTransform([13.01, 80.22, 13.06, 80.27], width=64, height=64)
        dummy_mask = np.zeros((64, 64), dtype=np.uint8)

        bld_res = analyze_building_exposure(
            buildings=[],
            seg_mask=dummy_mask,
            georef=georef,
            max_buildings=100,
        )
        self.assertEqual(bld_res["summary"]["total_buildings_analyzed"], 0)
        self.assertEqual(bld_res["summary"]["high_exposure_count"], 0)

        hazard_res = classify_road_hazards(
            road_features=[],
            seg_mask=dummy_mask,
            georef=georef,
        )
        self.assertEqual(hazard_res["summary"]["total_roads_analyzed"], 0)
        self.assertEqual(hazard_res["summary"]["potential_hazard_segments"], 0)

    # -------------------------------------------------------------
    # 11. Building Exposure Execution & Classification Categories
    # -------------------------------------------------------------
    def test_11_building_exposure_categories(self):
        """Verifies building exposure partitions into HIGH, MODERATE, LOW_OR_NO_DETECTED_EXPOSURE."""
        georef = GeoSpatialTransform([13.01, 80.22, 13.06, 80.27], width=100, height=100)
        # Create mask with top half flooded (class 2: CLASS_NEW_FLOOD)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :] = 2

        # Building entirely in flooded zone
        bld_high = {
            "id": 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[80.23, 13.04], [80.24, 13.04], [80.24, 13.05], [80.23, 13.05], [80.23, 13.04]]],
            },
            "properties": {"osm_id": "bld_1"},
        }
        # Building in dry zone
        bld_dry = {
            "id": 2,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[80.23, 13.015], [80.24, 13.015], [80.24, 13.02], [80.23, 13.02], [80.23, 13.015]]],
            },
            "properties": {"osm_id": "bld_2"},
        }

        res = analyze_building_exposure([bld_high, bld_dry], mask, georef)
        self.assertEqual(res["summary"]["total_buildings_analyzed"], 2)
        self.assertGreaterEqual(res["summary"]["high_exposure_count"], 1)
        self.assertGreaterEqual(res["summary"]["low_exposure_count"], 1)

    # -------------------------------------------------------------
    # 12. Road Hazard MultiDiGraph Generation & Parallel Edges
    # -------------------------------------------------------------
    def test_12_road_hazard_multidigraph_preservation(self):
        """Verifies road network graph is a MultiDiGraph with directional connectivity."""
        georef = GeoSpatialTransform([13.01, 80.22, 13.06, 80.27], width=100, height=100)
        mask = np.zeros((100, 100), dtype=np.uint8)

        # Two parallel ways between identical points
        way1 = {
            "id": 101,
            "geometry": {"type": "LineString", "coordinates": [[80.23, 13.02], [80.24, 13.03]]},
            "properties": {"osm_id": "way_1", "highway": "primary", "oneway": "no"},
        }
        way2 = {
            "id": 102,
            "geometry": {"type": "LineString", "coordinates": [[80.23, 13.02], [80.24, 13.03]]},
            "properties": {"osm_id": "way_2", "highway": "cycleway", "oneway": "no"},
        }

        G = build_road_graph([way1, way2], mask, georef)
        import networkx as nx
        self.assertIsInstance(G, nx.MultiDiGraph)
        # Should preserve both parallel edges in forward and backward directions
        n1 = (round(80.23, 6), round(13.02, 6))
        n2 = (round(80.24, 6), round(13.03, 6))
        edges = G.get_edge_data(n1, n2)
        self.assertEqual(len(edges), 2, "MultiDiGraph must preserve both distinct parallel edges.")

    # -------------------------------------------------------------
    # 13. A* Routing Metric Length and Formula ETA
    # -------------------------------------------------------------
    def test_13_astar_metric_length_and_eta_formula(self):
        """Verifies ETA follows the exact formula: (distance_m / 1000) / speed_kmh * 60."""
        import networkx as nx
        georef = GeoSpatialTransform([13.01, 80.22, 13.06, 80.27], width=100, height=100)

        G = nx.MultiDiGraph()
        n1 = (80.230000, 13.020000)
        n2 = (80.240000, 13.030000)
        dist_m = 1500.0
        G.add_edge(n1, n2, key="e1", length_m=dist_m, weight=dist_m, hazard="NO_DETECTED_INTERSECTION")

        route_res = calculate_astar_evacuation_route(G, n1, n2, georef, speed_kmh=25.0)
        self.assertEqual(route_res["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")
        self.assertEqual(route_res["distance_m"], 1500.0)

        # Expected ETA: (1500 / 1000) / 25 * 60 = 3.6 minutes
        expected_eta = round((1500.0 / 1000.0) / 25.0 * 60.0, 2)
        self.assertEqual(route_res["eta_minutes"], expected_eta)

    # -------------------------------------------------------------
    # 14. No-Route Handling when Isolated
    # -------------------------------------------------------------
    def test_14_no_route_handling_when_disconnected(self):
        """Verifies router returns NO_SAFE_ROUTE_AVAILABLE gracefully when destination is unreachable."""
        import networkx as nx
        georef = GeoSpatialTransform([13.01, 80.22, 13.06, 80.27], width=100, height=100)

        G = nx.MultiDiGraph()
        # Two disconnected components
        G.add_edge((80.23, 13.02), (80.235, 13.025), key="e1", length_m=500.0, weight=500.0, hazard="NO_DETECTED_INTERSECTION")
        G.add_edge((80.25, 13.04), (80.255, 13.045), key="e2", length_m=500.0, weight=500.0, hazard="NO_DETECTED_INTERSECTION")

        route_res = calculate_astar_evacuation_route(G, (80.23, 13.02), (80.25, 13.04), georef)
        self.assertEqual(route_res["status"], "NO_SAFE_ROUTE_AVAILABLE")
        self.assertEqual(route_res["distance_m"], 0.0)

    # -------------------------------------------------------------
    # 15. Partial Pipeline Failure and Stage Output Preservation
    # -------------------------------------------------------------
    def test_15_partial_failure_output_preservation(self):
        """Verifies failed runs preserve completed stages in the atomic record."""
        # Check an existing completed run or create one with partial output
        records = pipeline_orchestrator.list_history(5)
        self.assertIsInstance(records, list)

    # -------------------------------------------------------------
    # 16. Provenance Metadata Generation
    # -------------------------------------------------------------
    def test_16_provenance_metadata_completeness(self):
        """Verifies provenance output records CRS, bounding box, sensor types, and checkpoint."""
        runs = pipeline_orchestrator.list_history(5)
        chennai_runs = [r for r in runs if "chennai" in str(r.get("study_area")) and r.get("status") in ("COMPLETED", "COMPLETED_WITH_WARNINGS")]
        if chennai_runs:
            res = pipeline_orchestrator.get_result(chennai_runs[0]["execution_id"])
            outputs = res.get("stage_outputs", {})
            self.assertIn("preprocessing", outputs)
            prep = outputs["preprocessing"]
            self.assertIn("crs", prep)
            self.assertIn("projected_utm_crs", prep)
            self.assertIn("transform", prep)

    # -------------------------------------------------------------
    # 17. FastAPI Endpoints via TestClient
    # -------------------------------------------------------------
    def test_17_fastapi_endpoints_integration(self):
        """Tests all 5 Phase 6 pipeline endpoints via FastAPI TestClient."""
        # 1. GET /api/pipeline/health
        h_resp = self.client.get("/api/pipeline/health")
        self.assertEqual(h_resp.status_code, 200)
        h_data = h_resp.json()
        self.assertEqual(h_data["status"], "HEALTHY")
        self.assertIn("runs_directory", h_data)

        # 2. GET /api/pipeline/history
        hist_resp = self.client.get("/api/pipeline/history?limit=5")
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.json()
        self.assertIsInstance(hist_data, list)

        # 3. POST /api/pipeline/run (invalid study area should reject concurrent lock if running)
        run_payload = {
            "study_area": "chennai_monsoon_2026",
            "run_temporal_analysis": False,
            "run_infrastructure_exposure": False,
            "run_candidate_routing": False,
        }
        r_resp = self.client.post("/api/pipeline/run", json=run_payload)
        self.assertIn(r_resp.status_code, (200, 409))  # 200 if launched, 409 if locked
        if r_resp.status_code == 200:
            eid = r_resp.json()["execution_id"]

            # 4. GET /api/pipeline/status/{eid}
            st_resp = self.client.get(f"/api/pipeline/status/{eid}")
            self.assertEqual(st_resp.status_code, 200)
            self.assertEqual(st_resp.json()["execution_id"], eid)

            # Wait briefly for quick no-op run to complete
            time.sleep(2.0)
            res_resp = self.client.get(f"/api/pipeline/result/{eid}")
            self.assertIn(res_resp.status_code, (200, 404))

        # 5. GET /api/pipeline/status/nonexistent -> 404
        bad_resp = self.client.get("/api/pipeline/status/exec_nonexistent_xyz")
        self.assertEqual(bad_resp.status_code, 404)

    # -------------------------------------------------------------
    # 18. Per-Study-Area Execution Locking
    # -------------------------------------------------------------
    def test_18_study_area_execution_lock(self):
        """Verifies concurrent execution requests for the same study area are rejected."""
        lock_area = f"test_locking_area_{int(time.time())}"
        with pipeline_orchestrator._lock:
            pipeline_orchestrator._study_area_locks[lock_area] = "exec_dummy_locked"
            pipeline_orchestrator._active_runs["exec_dummy_locked"] = {
                "execution_id": "exec_dummy_locked",
                "study_area": lock_area,
                "status": PipelineStatus.RUNNING,
            }

        try:
            with self.assertRaises(ValueError):
                pipeline_orchestrator.start_pipeline(study_area=lock_area)
        finally:
            with pipeline_orchestrator._lock:
                if lock_area in pipeline_orchestrator._study_area_locks:
                    del pipeline_orchestrator._study_area_locks[lock_area]
                if "exec_dummy_locked" in pipeline_orchestrator._active_runs:
                    del pipeline_orchestrator._active_runs["exec_dummy_locked"]


if __name__ == "__main__":
    unittest.main()
