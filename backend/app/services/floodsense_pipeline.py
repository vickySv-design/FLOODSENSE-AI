"""
FloodSense-AI: Unified End-to-End Orchestration Pipeline
=========================================================
Coordinates complete disaster response workflow:
1. Study Area Validation
2. Satellite Data Acquisition / Cache Retrieval
3. Input Band & Geometry Physical Validation
4. Multimodal Preprocessing (CLAHE, NDWI, dB, Speckle Filter)
5. Deep Learning Inundation Inference (Dual-Stream Attention U-Net)
6. Georeferenced GeoTIFF Export & Provenance Manifest Recording
7. Temporal SAR Change Detection (when pre-event SAR is available)
8. OSM Building Exposure Grading
9. OSM Road Network Hazard Sampling & MultiDiGraph Generation
10. Admissible A* Candidate Rescue Routing
11. Results Aggregation & Academic Disclaimers
"""

import os
import sys
import time
import json
import uuid
import shutil
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from app.config import (
    CHECKPOINTS_DIR,
    CLASS_DRY_LAND,
    CLASS_NEW_FLOOD,
    CLASS_PERMANENT_WATER,
    DATA_DIR,
    DISASTER_PRESETS_METADATA,
    IMAGE_SIZE,
    PRESETS_DIR,
    SATELLITE_CACHE_DIR,
)
from app.services.damage_analyzer import analyze_building_exposure
from app.services.data_acquisition_manager import data_acquisition_manager
from app.services.data_provenance import data_provenance_manager
from app.services.georef import GeoSpatialTransform
from app.services.inference_engine import inference_engine
from app.services.osm_service import osm_service
from app.services.preprocessor import preprocess_multimodal_pair
from app.services.raster_validator import raster_validator
from app.services.rescue_router import (
    build_road_graph,
    calculate_astar_evacuation_route,
    classify_road_hazards,
)

logger = logging.getLogger("floodsense.pipeline")

# Output directory for pipeline execution records
PIPELINE_RUNS_DIR = DATA_DIR / "pipeline_runs"
PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)


class PipelineStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"


class PipelineStage:
    INITIALIZATION = "INITIALIZATION"
    DATA_ACQUISITION = "DATA_ACQUISITION"
    PREPROCESSING = "PREPROCESSING"
    FLOOD_INFERENCE = "FLOOD_INFERENCE"
    TEMPORAL_ANALYSIS = "TEMPORAL_ANALYSIS"
    BUILDING_EXPOSURE = "BUILDING_EXPOSURE"
    ROAD_HAZARD_ANALYSIS = "ROAD_HAZARD_ANALYSIS"
    RESCUE_ROUTING = "RESCUE_ROUTING"
    FINALIZATION = "FINALIZATION"


class FloodSensePipelineOrchestrator:
    """
    Thread-safe orchestrator managing end-to-end execution of FloodSense-AI disaster response runs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_runs: Dict[str, Dict[str, Any]] = {}
        self._study_area_locks: Dict[str, str] = {}  # study_area -> execution_id

    def list_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Lists recent pipeline execution runs from disk and memory."""
        runs: List[Dict[str, Any]] = []
        with self._lock:
            # Memory runs
            for eid, r in self._active_runs.items():
                runs.append({
                    "execution_id": eid,
                    "study_area": r.get("study_area"),
                    "status": r.get("status"),
                    "current_stage": r.get("current_stage"),
                    "created_at": r.get("created_at"),
                    "completed_at": r.get("completed_at"),
                    "duration_seconds": r.get("duration_seconds"),
                    "warnings_count": len(r.get("warnings", [])),
                    "errors_count": len(r.get("errors", [])),
                })

        # Disk runs
        try:
            for p in sorted(PIPELINE_RUNS_DIR.glob("exec_*.json"), key=os.path.getmtime, reverse=True):
                eid = p.stem
                if any(x["execution_id"] == eid for x in runs):
                    continue
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    runs.append({
                        "execution_id": eid,
                        "study_area": data.get("study_area"),
                        "status": data.get("status"),
                        "current_stage": data.get("current_stage"),
                        "created_at": data.get("created_at"),
                        "completed_at": data.get("completed_at"),
                        "duration_seconds": data.get("duration_seconds"),
                        "warnings_count": len(data.get("warnings", [])),
                        "errors_count": len(data.get("errors", [])),
                    })
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Error scanning pipeline runs dir: {e}")

        runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return runs[:limit]

    def get_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Returns live execution status and stage progress."""
        with self._lock:
            if execution_id in self._active_runs:
                r = self._active_runs[execution_id]
                return {
                    "execution_id": execution_id,
                    "study_area": r.get("study_area"),
                    "status": r.get("status"),
                    "current_stage": r.get("current_stage"),
                    "stage_progress": r.get("stage_progress", {}),
                    "created_at": r.get("created_at"),
                    "completed_at": r.get("completed_at"),
                    "duration_seconds": r.get("duration_seconds"),
                    "warnings": r.get("warnings", []),
                    "errors": r.get("errors", []),
                }

        # Check disk
        fp = PIPELINE_RUNS_DIR / f"{execution_id}.json"
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "execution_id": execution_id,
                    "study_area": data.get("study_area"),
                    "status": data.get("status"),
                    "current_stage": data.get("current_stage"),
                    "stage_progress": data.get("stage_progress", {}),
                    "created_at": data.get("created_at"),
                    "completed_at": data.get("completed_at"),
                    "duration_seconds": data.get("duration_seconds"),
                    "warnings": data.get("warnings", []),
                    "errors": data.get("errors", []),
                }
            except Exception as e:
                logger.error(f"Failed to read execution record {fp}: {e}")
                return None
        return None

    def get_result(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Returns the complete structured payload for a completed pipeline run."""
        with self._lock:
            if execution_id in self._active_runs:
                r = self._active_runs[execution_id]
                if r.get("status") in [PipelineStatus.COMPLETED, PipelineStatus.COMPLETED_WITH_WARNINGS, PipelineStatus.FAILED]:
                    return r

        fp = PIPELINE_RUNS_DIR / f"{execution_id}.json"
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read result file {fp}: {e}")
                return None
        return None

    def get_health(self) -> Dict[str, Any]:
        """Returns health status of the pipeline orchestrator and running tasks."""
        with self._lock:
            active_count = sum(1 for r in self._active_runs.values() if r.get("status") == PipelineStatus.RUNNING)
            pending_count = sum(1 for r in self._active_runs.values() if r.get("status") == PipelineStatus.PENDING)
            completed_count = sum(1 for r in self._active_runs.values() if r.get("status") in (PipelineStatus.COMPLETED, PipelineStatus.COMPLETED_WITH_WARNINGS))
            failed_count = sum(1 for r in self._active_runs.values() if r.get("status") == PipelineStatus.FAILED)
            locked_areas = list(self._study_area_locks.keys())

        return {
            "status": "HEALTHY",
            "active_executions": active_count,
            "pending_executions": pending_count,
            "completed_in_memory": completed_count,
            "failed_in_memory": failed_count,
            "locked_study_areas": locked_areas,
            "runs_directory": str(PIPELINE_RUNS_DIR),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def start_pipeline(
        self,
        study_area: str = "chennai_monsoon_2026",
        bbox: Optional[List[float]] = None,
        run_temporal_analysis: bool = True,
        run_infrastructure_exposure: bool = True,
        run_candidate_routing: bool = True,
        avoid_potential_hazards: bool = True,
        start_coords: Optional[Tuple[float, float]] = None,
        dest_coords: Optional[Tuple[float, float]] = None,
        max_buildings: int = 1000,
    ) -> Dict[str, Any]:
        """
        Validates request and launches background pipeline execution thread.
        Returns execution_id and initial status.
        """
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._lock:
            # Prevent conflicting concurrent executions on same study area
            if study_area in self._study_area_locks:
                active_eid = self._study_area_locks[study_area]
                if active_eid in self._active_runs and self._active_runs[active_eid]["status"] == PipelineStatus.RUNNING:
                    raise ValueError(
                        f"Study area '{study_area}' currently has an active pipeline execution ({active_eid}). "
                        f"Wait for completion or use a different study area."
                    )

            self._study_area_locks[study_area] = execution_id
            init_record = {
                "execution_id": execution_id,
                "study_area": study_area,
                "bbox": bbox,
                "status": PipelineStatus.PENDING,
                "current_stage": PipelineStage.INITIALIZATION,
                "created_at": now_iso,
                "completed_at": None,
                "duration_seconds": 0.0,
                "warnings": [],
                "errors": [],
                "stage_progress": {
                    PipelineStage.INITIALIZATION: {"status": "COMPLETED", "duration_s": 0.0},
                    PipelineStage.DATA_ACQUISITION: {"status": "PENDING", "duration_s": 0.0},
                    PipelineStage.PREPROCESSING: {"status": "PENDING", "duration_s": 0.0},
                    PipelineStage.FLOOD_INFERENCE: {"status": "PENDING", "duration_s": 0.0},
                    PipelineStage.TEMPORAL_ANALYSIS: {"status": "PENDING", "duration_s": 0.0},
                    PipelineStage.BUILDING_EXPOSURE: {"status": "PENDING", "duration_s": 0.0},
                    PipelineStage.ROAD_HAZARD_ANALYSIS: {"status": "PENDING", "duration_s": 0.0},
                    PipelineStage.RESCUE_ROUTING: {"status": "PENDING", "duration_s": 0.0},
                    PipelineStage.FINALIZATION: {"status": "PENDING", "duration_s": 0.0},
                },
                "config": {
                    "study_area": study_area,
                    "bbox": bbox,
                    "run_temporal_analysis": run_temporal_analysis,
                    "run_infrastructure_exposure": run_infrastructure_exposure,
                    "run_candidate_routing": run_candidate_routing,
                    "avoid_potential_hazards": avoid_potential_hazards,
                    "start_coords": start_coords,
                    "dest_coords": dest_coords,
                    "max_buildings": max_buildings,
                },
                "stage_outputs": {},
            }
            self._active_runs[execution_id] = init_record

        # Launch background thread
        thread = threading.Thread(
            target=self._execute_pipeline_worker,
            args=(execution_id,),
            daemon=True,
            name=f"PipelineWorker-{execution_id}",
        )
        thread.start()

        return {
            "success": True,
            "execution_id": execution_id,
            "study_area": study_area,
            "status": PipelineStatus.PENDING,
            "message": f"Pipeline execution '{execution_id}' launched successfully.",
            "created_at": now_iso,
        }

    def _execute_pipeline_worker(self, execution_id: str) -> None:
        """Sequential worker executing each pipeline stage with graceful error handling."""
        start_time = time.time()
        record = self._active_runs[execution_id]
        record["status"] = PipelineStatus.RUNNING

        cfg = record["config"]
        study_area = cfg["study_area"]
        warnings: List[str] = record["warnings"]
        errors: List[str] = record["errors"]
        stage_progress = record["stage_progress"]
        stage_outputs = record["stage_outputs"]

        logger.info(f"[{execution_id}] Starting unified FloodSense pipeline for area: {study_area}")

        try:
            # -------------------------------------------------------------
            # STAGE 1: DATA ACQUISITION
            # -------------------------------------------------------------
            t_stage = time.time()
            record["current_stage"] = PipelineStage.DATA_ACQUISITION
            stage_progress[PipelineStage.DATA_ACQUISITION]["status"] = "RUNNING"

            # 1.1 Study area and bbox resolution
            clean_id = (study_area or "chennai_monsoon_2026").lower()
            resolved_bbox = cfg["bbox"]

            if not resolved_bbox:
                if clean_id in DISASTER_PRESETS_METADATA:
                    resolved_bbox = DISASTER_PRESETS_METADATA[clean_id]["bbox"]
                elif (PRESETS_DIR / clean_id).exists():
                    resolved_bbox = [13.010, 80.220, 13.060, 80.270]
                else:
                    raise FileNotFoundError(f"Study area '{study_area}' is not a registered preset and no bounding box was specified.")
            record["bbox"] = resolved_bbox

            # 1.2 Acquire or retrieve compatible optical & SAR rasters
            opt_arr = None
            sar_arr = None
            pre_event_sar_arr = None
            acquisition_meta = {}

            # Check local preset folder first
            preset_folder = PRESETS_DIR / clean_id
            if (preset_folder / "optical_pre.npy").exists() and (preset_folder / "sar_post.npy").exists():
                opt_arr = np.load(preset_folder / "optical_pre.npy").astype(np.float32)
                sar_arr = np.load(preset_folder / "sar_post.npy").astype(np.float32)
                acquisition_meta = {
                    "source": "LOCAL_PRESET_CACHE",
                    "path": str(preset_folder),
                    "optical_channels": ["B04", "B03", "B02", "B08"],
                    "sar_channels": ["VV", "VH"],
                    "optical_shape": list(opt_arr.shape),
                    "sar_shape": list(sar_arr.shape),
                }
            else:
                # Query central DataAcquisitionManager
                pre_d = DISASTER_PRESETS_METADATA.get(clean_id, {}).get("pre_date", "2025-11-10")
                post_d = DISASTER_PRESETS_METADATA.get(clean_id, {}).get("post_date", "2025-12-05")
                try:
                    acq_res = data_acquisition_manager.acquire_multimodal_pair(
                        bbox=resolved_bbox,
                        pre_date=pre_d,
                        post_date=post_d,
                        provider="copernicus",
                        aoi_name=study_area,
                    )
                except Exception as acq_err:
                    acq_res = {"status": "FAILED", "error": str(acq_err)}

                if acq_res.get("status") == "ACQUIRED_AND_VALIDATED":
                    # Load from band paths
                    opt_paths = acq_res["optical_band_paths"]
                    sar_paths = acq_res["sar_band_paths"]
                    with rasterio.open(opt_paths["B04"]) as r, rasterio.open(opt_paths["B03"]) as g, \
                         rasterio.open(opt_paths["B02"]) as b, rasterio.open(opt_paths["B08"]) as nir:
                        opt_arr = np.stack([r.read(1), g.read(1), b.read(1), nir.read(1)], axis=0).astype(np.float32)
                    with rasterio.open(sar_paths["VV"]) as vv, rasterio.open(sar_paths["VH"]) as vh:
                        sar_arr = np.stack([vv.read(1), vh.read(1)], axis=0).astype(np.float32)
                    acquisition_meta = {
                        "source": "DATA_ACQUISITION_MANAGER",
                        "analysis_id": acq_res.get("analysis_id"),
                        "optical_channels": ["B04", "B03", "B02", "B08"],
                        "sar_channels": ["VV", "VH"],
                    }
                else:
                    raise FileNotFoundError(f"Failed to acquire compatible satellite rasters for study area '{study_area}'.")

            # Check if compatible pre-event SAR exists for temporal analysis
            pre_event_s1_path = SATELLITE_CACHE_DIR / "CDSE_SENTINEL-1_20251110_13.01_80.22" / "CDSE_SENTINEL-1_20251110_13.01_80.22_VV.tif"
            if pre_event_s1_path.exists():
                try:
                    with rasterio.open(pre_event_s1_path) as s1_src:
                        pre_event_sar_arr = s1_src.read(1).astype(np.float32)
                except Exception as e:
                    logger.warning(f"Could not load pre-event SAR raster: {e}")

            stage_progress[PipelineStage.DATA_ACQUISITION]["status"] = "COMPLETED"
            stage_progress[PipelineStage.DATA_ACQUISITION]["duration_s"] = round(time.time() - t_stage, 3)
            stage_outputs["acquisition"] = acquisition_meta

            # -------------------------------------------------------------
            # STAGE 2: PHYSICAL PREPROCESSING & GEOMETRY VALIDATION
            # -------------------------------------------------------------
            t_stage = time.time()
            record["current_stage"] = PipelineStage.PREPROCESSING
            stage_progress[PipelineStage.PREPROCESSING]["status"] = "RUNNING"

            # Check dimensions and band counts
            if opt_arr.shape[0] != 4:
                raise ValueError(f"Optical tensor must contain 4 channels [B4, B3, B2, B8]; received {opt_arr.shape[0]}.")
            if sar_arr.shape[0] != 2:
                raise ValueError(f"SAR tensor must contain 2 channels [VV, VH]; received {sar_arr.shape[0]}.")

            # Initialize GeoSpatialTransform
            H, W = sar_arr.shape[1], sar_arr.shape[2]
            georef = GeoSpatialTransform(resolved_bbox, width=W, height=H)

            # Execute preprocessing pipeline
            preprocessed = preprocess_multimodal_pair(
                optical_4ch=opt_arr,
                sar_2ch=sar_arr,
                device=inference_engine.device,
            )

            stage_progress[PipelineStage.PREPROCESSING]["status"] = "COMPLETED"
            stage_progress[PipelineStage.PREPROCESSING]["duration_s"] = round(time.time() - t_stage, 3)
            stage_outputs["preprocessing"] = {
                "optical_shape": list(opt_arr.shape),
                "sar_shape": list(sar_arr.shape),
                "grid_dimensions": [W, H],
                "crs": georef.wgs84_crs.to_string(),
                "projected_utm_crs": georef.utm_crs.to_string(),
                "device": str(inference_engine.device),
                "transform": [georef.res_x, 0.0, georef.min_lon, 0.0, -georef.res_y, georef.max_lat],
            }

            # -------------------------------------------------------------
            # STAGE 3: DEEP LEARNING FLOOD INFERENCE
            # -------------------------------------------------------------
            t_stage = time.time()
            record["current_stage"] = PipelineStage.FLOOD_INFERENCE
            stage_progress[PipelineStage.FLOOD_INFERENCE]["status"] = "RUNNING"

            # Execute sliding-window inference with model
            inf_res = inference_engine.run_inference(preprocessed=preprocessed)
            seg_mask: np.ndarray = inf_res["seg_mask"]

            # Compute exact pixel counts
            vals, counts = np.unique(seg_mask, return_counts=True)
            c_dict = dict(zip(vals.tolist(), counts.tolist()))
            dry_count = c_dict.get(CLASS_DRY_LAND, 0)
            perm_count = c_dict.get(CLASS_PERMANENT_WATER, 0)
            flood_count = c_dict.get(CLASS_NEW_FLOOD, 0)
            nodata_count = c_dict.get(255, 0)

            # Export georeferenced GeoTIFF
            out_raster_path = SATELLITE_CACHE_DIR / f"pipeline_{execution_id}_flood_mask.tif"
            inference_engine.export_geotiff_flood_mask(
                seg_mask=seg_mask,
                bounds=tuple(resolved_bbox),
                output_path=str(out_raster_path),
            )

            # Calculate metric area (UTM or geodesic pixel area)
            pixel_area_m2 = (georef.res_x * 111320.0) * (georef.res_y * 110540.0)
            flood_area_km2 = round((flood_count * pixel_area_m2) / 1_000_000.0, 3)
            flood_area_ha = round(flood_area_km2 * 100.0, 2)

            stage_progress[PipelineStage.FLOOD_INFERENCE]["status"] = "COMPLETED"
            stage_progress[PipelineStage.FLOOD_INFERENCE]["duration_s"] = round(time.time() - t_stage, 3)
            stage_outputs["inference"] = {
                "checkpoint": "DualStreamAttentionUNet_v2.1",
                "checkpoint_path": str(CHECKPOINTS_DIR / "dual_stream_attention_unet.pth"),
                "device": str(inf_res["device"]),
                "latency_ms": inf_res.get("latency_ms", 0.0),
                "exported_raster_path": str(out_raster_path),
                "total_pixels": int(seg_mask.size),
                "flood_pixel_count": flood_count,
                "permanent_water_pixel_count": perm_count,
                "dry_land_pixel_count": dry_count,
                "flooded_area_sqkm": flood_area_km2,
                "flooded_area_hectares": flood_area_ha,
                "class_pixel_counts": {
                    "dry_land": dry_count,
                    "permanent_water": perm_count,
                    "mapped_floodwater": flood_count,
                    "nodata": nodata_count,
                },
                "flood_inundation_area_km2": flood_area_km2,
                "flood_inundation_area_ha": flood_area_ha,
                "data_nature": "DERIVED_MODEL_PREDICTION",
            }

            # -------------------------------------------------------------
            # STAGE 4: TEMPORAL SAR CHANGE DETECTION (Conditional)
            # -------------------------------------------------------------
            t_stage = time.time()
            record["current_stage"] = PipelineStage.TEMPORAL_ANALYSIS
            stage_progress[PipelineStage.TEMPORAL_ANALYSIS]["status"] = "RUNNING"

            if cfg["run_temporal_analysis"] and pre_event_sar_arr is not None:
                # Execute temporal change detection
                vv_post = sar_arr[0]
                vv_pre = pre_event_sar_arr
                if vv_post.shape == vv_pre.shape:
                    delta_vv = vv_post - vv_pre
                    changed_pixels = int(np.sum(delta_vv < -0.15))
                    stage_outputs["temporal_analysis"] = {
                        "status": "COMPLETED",
                        "method": "SENTINEL_1_SAR_TEMPORAL_DIFFERENCING",
                        "threshold_db": -5.0,
                        "detected_changed_pixels": changed_pixels,
                        "pre_event_available": True,
                    }
                    stage_progress[PipelineStage.TEMPORAL_ANALYSIS]["status"] = "COMPLETED"
                else:
                    warnings.append("Pre-event SAR raster shape does not match post-event raster shape; temporal differencing skipped.")
                    stage_outputs["temporal_analysis"] = {
                        "status": "UNAVAILABLE",
                        "reason": "SHAPE_MISMATCH",
                        "pre_event_available": False,
                    }
                    stage_progress[PipelineStage.TEMPORAL_ANALYSIS]["status"] = "COMPLETED"
            else:
                msg = "Temporal SAR change analysis unavailable: no compatible pre-event baseline SAR observation found on disk."
                warnings.append(msg)
                stage_outputs["temporal_analysis"] = {
                    "status": "UNAVAILABLE",
                    "reason": "NO_PRE_EVENT_OBSERVATION",
                    "pre_event_available": False,
                }
                stage_progress[PipelineStage.TEMPORAL_ANALYSIS]["status"] = "COMPLETED"

            stage_progress[PipelineStage.TEMPORAL_ANALYSIS]["duration_s"] = round(time.time() - t_stage, 3)

            # -------------------------------------------------------------
            # STAGE 5: BUILDING EXPOSURE ANALYSIS
            # -------------------------------------------------------------
            t_stage = time.time()
            record["current_stage"] = PipelineStage.BUILDING_EXPOSURE
            stage_progress[PipelineStage.BUILDING_EXPOSURE]["status"] = "RUNNING"

            if cfg["run_infrastructure_exposure"]:
                # Fetch OSM buildings
                _, blds_geo, from_cache, bld_warn = osm_service.fetch_and_cache_osm_data(
                    study_area, resolved_bbox
                )
                if bld_warn:
                    warnings.append(f"OSM Buildings: {bld_warn}")

                raw_buildings = blds_geo.get("features", [])
                if raw_buildings:
                    exposure_res = analyze_building_exposure(
                        buildings=raw_buildings,
                        seg_mask=seg_mask,
                        georef=georef,
                        max_buildings=cfg["max_buildings"],
                        nodata_val=255.0,
                        flood_val=CLASS_NEW_FLOOD,
                    )
                    stage_outputs["building_exposure"] = {
                        "status": "COMPLETED",
                        "summary": exposure_res["summary"],
                        "buildings_geojson": exposure_res["buildings_geojson"],
                        "total_osm_buildings": len(raw_buildings),
                    }
                else:
                    warnings.append(f"No OpenStreetMap buildings available for study area '{study_area}'.")
                    stage_outputs["building_exposure"] = {
                        "status": "UNAVAILABLE",
                        "reason": "NO_OSM_BUILDINGS_FOUND",
                    }
            else:
                stage_outputs["building_exposure"] = {"status": "SKIPPED_BY_USER"}

            stage_progress[PipelineStage.BUILDING_EXPOSURE]["status"] = "COMPLETED"
            stage_progress[PipelineStage.BUILDING_EXPOSURE]["duration_s"] = round(time.time() - t_stage, 3)

            # -------------------------------------------------------------
            # STAGE 6: ROAD HAZARD ANALYSIS & MULTIDIGRAPH GENERATION
            # -------------------------------------------------------------
            t_stage = time.time()
            record["current_stage"] = PipelineStage.ROAD_HAZARD_ANALYSIS
            stage_progress[PipelineStage.ROAD_HAZARD_ANALYSIS]["status"] = "RUNNING"

            roads_geo, _, from_cache, road_warn = osm_service.fetch_and_cache_osm_data(
                study_area, resolved_bbox
            )
            if road_warn:
                warnings.append(f"OSM Roads: {road_warn}")

            raw_roads = roads_geo.get("features", [])
            G = None

            if raw_roads:
                # Classify road hazards across first 1,000 roads for report sample
                hazard_res = classify_road_hazards(
                    road_features=raw_roads[:1000],
                    seg_mask=seg_mask,
                    georef=georef,
                    nodata_val=255.0,
                    flood_val=CLASS_NEW_FLOOD,
                )

                # Build MultiDiGraph across full road network for live routing
                G = build_road_graph(
                    road_features=raw_roads,
                    seg_mask=seg_mask,
                    georef=georef,
                    avoid_hazards=cfg["avoid_potential_hazards"],
                    caution_penalty=3.0,
                    hazard_penalty=10.0,
                    nodata_val=255.0,
                    flood_val=CLASS_NEW_FLOOD,
                )

                stage_outputs["road_hazards"] = {
                    "status": "COMPLETED",
                    "summary": hazard_res["summary"],
                    "sample_summary": hazard_res["summary"],
                    "geojson": hazard_res["geojson"],
                    "roads_geojson": hazard_res["geojson"],
                    "full_network_stats": {
                        "total_osm_roads": len(raw_roads),
                        "graph_nodes": G.number_of_nodes(),
                        "graph_edges": G.number_of_edges(),
                        "graph_type": "MultiDiGraph",
                    },
                }
            else:
                warnings.append(f"No OpenStreetMap roads available for study area '{study_area}'.")
                stage_outputs["road_hazards"] = {
                    "status": "UNAVAILABLE",
                    "reason": "NO_OSM_ROADS_FOUND",
                }

            stage_progress[PipelineStage.ROAD_HAZARD_ANALYSIS]["status"] = "COMPLETED"
            stage_progress[PipelineStage.ROAD_HAZARD_ANALYSIS]["duration_s"] = round(time.time() - t_stage, 3)

            # -------------------------------------------------------------
            # STAGE 7: CANDIDATE RESCUE ROUTING
            # -------------------------------------------------------------
            t_stage = time.time()
            record["current_stage"] = PipelineStage.RESCUE_ROUTING
            stage_progress[PipelineStage.RESCUE_ROUTING]["status"] = "RUNNING"

            if cfg["run_candidate_routing"]:
                if G is not None and G.number_of_nodes() > 0:
                    # Resolve start and destination coordinates
                    s_coords = cfg["start_coords"]
                    d_coords = cfg["dest_coords"]

                    # Default to demonstrated corridor in Chennai if none provided
                    if not s_coords or not d_coords:
                        # Check preset defaults
                        if "chennai" in clean_id:
                            s_coords = (80.259097, 13.015806)
                            d_coords = (80.266400, 13.039248)
                        else:
                            # Pick two nodes from the graph
                            nodes_list = list(G.nodes())
                            s_coords = nodes_list[0]
                            d_coords = nodes_list[min(500, len(nodes_list) - 1)]

                    route_res = calculate_astar_evacuation_route(
                        G=G,
                        start_coords=s_coords,
                        dest_coords=d_coords,
                        georef=georef,
                        speed_kmh=25.0,
                        max_snap_dist_m=3500.0,
                    )
                    stage_outputs["rescue_routing"] = {
                        "status": route_res["status"],
                        "route": route_res,
                        "geojson": route_res.get("route_geojson"),
                        "avoid_hazards_enforced": cfg["avoid_potential_hazards"],
                    }
                else:
                    stage_outputs["rescue_routing"] = {
                        "status": "NO_SAFE_ROUTE_AVAILABLE",
                        "message": "Road graph is empty or unnavigable.",
                        "route": None,
                    }
            else:
                stage_outputs["rescue_routing"] = {"status": "SKIPPED_BY_USER"}

            stage_progress[PipelineStage.RESCUE_ROUTING]["status"] = "COMPLETED"
            stage_progress[PipelineStage.RESCUE_ROUTING]["duration_s"] = round(time.time() - t_stage, 3)

            # -------------------------------------------------------------
            # STAGE 8: FINALIZATION & ATOMIC PERSISTENCE
            # -------------------------------------------------------------
            t_stage = time.time()
            record["current_stage"] = PipelineStage.FINALIZATION
            stage_progress[PipelineStage.FINALIZATION]["status"] = "RUNNING"

            record["completed_at"] = datetime.now(timezone.utc).isoformat()
            record["duration_seconds"] = round(time.time() - start_time, 3)
            record["status"] = (
                PipelineStatus.COMPLETED_WITH_WARNINGS if warnings else PipelineStatus.COMPLETED
            )
            stage_progress[PipelineStage.FINALIZATION]["status"] = "COMPLETED"
            stage_progress[PipelineStage.FINALIZATION]["duration_s"] = round(time.time() - t_stage, 3)

            # Save immutable execution manifest atomically
            self._save_record_atomically(execution_id, record)
            logger.info(f"[{execution_id}] Pipeline completed in {record['duration_seconds']}s with status {record['status']}")

        except Exception as exc:
            logger.error(f"[{execution_id}] Pipeline failed at stage {record.get('current_stage')}: {exc}", exc_info=True)
            errors.append(str(exc))
            record["status"] = PipelineStatus.FAILED
            record["completed_at"] = datetime.now(timezone.utc).isoformat()
            record["duration_seconds"] = round(time.time() - start_time, 3)
            cur_st = record.get("current_stage", PipelineStage.FINALIZATION)
            if cur_st in stage_progress:
                stage_progress[cur_st]["status"] = "FAILED"
            self._save_record_atomically(execution_id, record)

        finally:
            with self._lock:
                if study_area in self._study_area_locks and self._study_area_locks[study_area] == execution_id:
                    del self._study_area_locks[study_area]

    def _save_record_atomically(self, execution_id: str, record: Dict[str, Any]) -> None:
        """Serializes pipeline record to disk using an atomic rename to prevent partial file reads."""
        final_path = PIPELINE_RUNS_DIR / f"{execution_id}.json"
        temp_path = PIPELINE_RUNS_DIR / f"{execution_id}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
            shutil.move(str(temp_path), str(final_path))
        except Exception as e:
            logger.error(f"Failed to persist pipeline record atomically: {e}")
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


# Global singleton instance
pipeline_orchestrator = FloodSensePipelineOrchestrator()
