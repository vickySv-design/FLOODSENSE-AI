from contextlib import asynccontextmanager
import json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field
import torch

from app.config import (
    DEVICE,
    DEVICE_NAME,
    DISASTER_PRESETS_METADATA,
    PRESETS_DIR,
)
from app.services.benchmark_suite import get_academic_benchmark_suite
from app.services.damage_analyzer import analyze_flood_damage
from app.services.gee_connector import gee_connector
from app.services.hydro_forecaster import simulate_hydrological_timeline
from app.services.inference_engine import inference_engine
from app.services.osm_service import osm_service
from app.services.preprocessor import (
    compute_sar_backscatter_histogram,
    decode_uploaded_optical,
    decode_uploaded_sar,
    preprocess_multimodal_pair,
    render_visual_layers,
)
from app.services.rescue_router import compute_osm_rescue_routes
from app.services.sitrep_generator import generate_disaster_sitrep
from scripts.init_demo_data import ensure_demo_data_initialized


class GEEFetchRequest(BaseModel):
    min_lat: float = Field(default=13.010, description="Southern latitude boundary")
    min_lon: float = Field(default=80.220, description="Western longitude boundary")
    max_lat: float = Field(default=13.060, description="Northern latitude boundary")
    max_lon: float = Field(default=80.270, description="Eastern longitude boundary")
    pre_date: str = Field(default="2025-11-10", description="Pre-disaster baseline date (YYYY-MM-DD)")
    post_date: str = Field(default="2025-12-05", description="Post-disaster SAR acquisition date (YYYY-MM-DD)")
    cloud_cover_max: float = Field(default=20.0, ge=0.0, le=100.0)
    region_name: Optional[str] = Field(default="Custom Copernicus AOI Sector")


class TimelineSimulateRequest(BaseModel):
    preset_id: str = Field(default="chennai_monsoon_2026")
    rainfall_mm_hr: float = Field(default=65.0, ge=0.0, le=150.0)
    horizon_hours: int = Field(default=24, description="Forecast horizon in hours: 0, 6, 12, 24, or 48")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_demo_data_initialized()
    inference_engine.load_checkpoint()
    yield


app = FastAPI(
    title="FloodSense-AI Geospatial Command API",
    description=(
        "Multi-Modal Sentinel-1 SAR & Sentinel-2 Optical Change Detection, "
        "Damage Mapping, and Dynamic A* Rescue Routing Engine"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _execute_full_geospatial_pipeline(
    optical_4ch: np.ndarray,
    sar_2ch: np.ndarray,
    scenario_meta: Dict[str, Any],
    gt_mask: Optional[np.ndarray] = None,
    preset_buildings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Runs the complete end-to-end FloodSense-AI pipeline:
    1. Lee-Sigma SAR speckle filtering + CLAHE + NDWI calculation
    2. PyTorch DualStreamAttentionUNet forward pass + Physics-Guided Neural Fusion
    3. Multi-modal Base64 visual layer rendering (Optical, SAR, NDWI, XAI Attention, Flood Mask)
    4. Spatial telemetry (10m/pixel -> km2), contour GeoJSON extraction, building damage grading
    5. NetworkX A* dynamic evacuation routing around flooded road segments
    """
    bbox = scenario_meta["bbox"]
    preprocessed = preprocess_multimodal_pair(optical_4ch, sar_2ch, DEVICE)

    inf_out = inference_engine.run_inference(
        preprocessed=preprocessed,
        ground_truth_mask=gt_mask,
        preset_metrics=scenario_meta.get("historical_metrics"),
    )

    seg_mask: np.ndarray = inf_out["seg_mask"]  # type: ignore[assignment]
    attention_map: np.ndarray = inf_out["attention_map"]  # type: ignore[assignment]

    visual_layers = render_visual_layers(
        optical_4ch=preprocessed["optical_clean"],  # type: ignore[arg-type]
        sar_filtered=preprocessed["sar_filtered"],  # type: ignore[arg-type]
        ndwi=preprocessed["ndwi"],  # type: ignore[arg-type]
        attention_map=attention_map,
        seg_mask=seg_mask,
    )

    sar_histogram = compute_sar_backscatter_histogram(
        vv_db=preprocessed["vv_db"],  # type: ignore[arg-type]
        vh_db=preprocessed["vh_db"],  # type: ignore[arg-type]
    )

    damage_report = analyze_flood_damage(
        seg_mask=seg_mask,
        bbox=bbox,
        preset_id=scenario_meta.get("id", "chennai_monsoon_2026"),
        population_density_factor=int(scenario_meta.get("population_density_factor", 32)),
    )

    rescue_report = compute_osm_rescue_routes(
        seg_mask=seg_mask,
        bbox=bbox,
        preset_id=scenario_meta.get("id", "chennai_monsoon_2026"),
    )

    # Assemble unified GeoJSON FeatureCollection for 1-click GIS export
    export_features: List[Dict[str, Any]] = []
    export_features.extend(damage_report["geojson_layers"]["new_flood"]["features"])
    export_features.extend(damage_report["geojson_layers"]["permanent_water"]["features"])
    export_features.extend(damage_report["geojson_layers"]["high_risk_buffer"]["features"])
    export_features.extend(rescue_report["geojson"]["safe_routes"]["features"])
    export_features.extend(rescue_report["geojson"]["severed_roads"]["features"])
    if "features" in damage_report.get("buildings_geojson", {}):
        export_features.extend(damage_report["buildings_geojson"]["features"][:120])

    # Provenance telemetry
    road_provenance = rescue_report.get("data_provenance", {})
    bld_provenance = damage_report.get("data_provenance", {})

    return {
        "scenario": scenario_meta,
        "engine": {
            "device": inf_out["device"],
            "cuda_available": torch.cuda.is_available(),
            "latency_ms": inf_out["latency_ms"],
            "architecture": inf_out["architecture"],
            "spatial_resolution_m": 10.0,
        },
        "telemetry": damage_report["telemetry"],
        "metrics": inf_out["metrics"],
        "buildings_summary": damage_report["buildings_summary"],
        "buildings": damage_report["buildings"],
        "rescue": {
            "summary": rescue_report["summary"],
            "routes": rescue_report["routes"],
            "waypoints": rescue_report["waypoints"],
        },
        "visual_layers": visual_layers,
        "charts": {
            "hydrology_breakdown": damage_report["hydrology_chart_data"],
            "sar_backscatter_curve": sar_histogram,
        },
        "geojson_layers": {
            "permanent_water": damage_report["geojson_layers"]["permanent_water"],
            "new_flood": damage_report["geojson_layers"]["new_flood"],
            "high_risk_buffer": damage_report["geojson_layers"]["high_risk_buffer"],
            "safe_routes": rescue_report["geojson"]["safe_routes"],
            "severed_roads": rescue_report["geojson"]["severed_roads"],
            "all_roads": rescue_report["geojson"]["all_roads"],
            "buildings": damage_report["buildings_geojson"],
        },
        # Phase 4 Dedicated GeoJSON Keys
        "roads_geojson": rescue_report["geojson"]["all_roads"],
        "blocked_roads_geojson": rescue_report["geojson"]["severed_roads"],
        "rescue_routes_geojson": rescue_report["geojson"]["safe_routes"],
        "buildings_geojson": damage_report["buildings_geojson"],
        # Phase 4 Provenance Metadata
        "data_provenance": {
            "road_data_source": road_provenance.get("road_data_source", "OpenStreetMap (ODbL)"),
            "building_data_source": bld_provenance.get("building_data_source", "OpenStreetMap (ODbL)"),
            "osm_cache_used": road_provenance.get("osm_cache_used", True),
            "road_features_count": rescue_report["summary"]["total_road_segments"],
            "building_features_count": damage_report["buildings_summary"]["total_buildings"],
            "blocked_road_count": rescue_report["summary"]["blocked_road_segments"],
            "available_route_count": rescue_report["summary"]["active_safe_routes"],
            "routing_status": (
                "PASSABLE_NAVIGABLE" if rescue_report["summary"]["active_safe_routes"] > 0 else "NO_SAFE_ROUTE_AVAILABLE"
            ),
            "data_warnings": road_provenance.get("data_warnings") or bld_provenance.get("data_warnings"),
        },
        "export_geojson": {
            "type": "FeatureCollection",
            "metadata": {
                "project": "FloodSense-AI Disaster Response SitRep",
                "scenario": scenario_meta["name"],
                "region": scenario_meta["region"],
                "bbox": bbox,
            },
            "features": export_features,
        },
    }


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    total_params = sum(p.numel() for p in inference_engine.model.parameters())
    return {
        "status": "OPERATIONAL",
        "service": "FloodSense-AI Dual-Stream SAR-Optical Engine",
        "device": DEVICE_NAME,
        "cuda_available": torch.cuda.is_available(),
        "model_architecture": "DualStreamAttentionUNet (Siamese 4-Stage + CMAF + Attention Gates)",
        "model_parameters": total_params,
        "weights_calibrated": inference_engine.weights_loaded,
        "presets_count": len(DISASTER_PRESETS_METADATA),
    }


@app.get("/api/presets")
async def list_disaster_presets() -> Dict[str, Any]:
    ensure_demo_data_initialized()
    return {
        "presets": list(DISASTER_PRESETS_METADATA.values()),
        "default_preset_id": "chennai_monsoon_2026",
    }


@app.post("/api/analyze")
async def analyze_disaster_scene(
    preset_id: Optional[str] = Form(default=None),
    custom_name: Optional[str] = Form(default=None),
    bbox_json: Optional[str] = Form(default=None),
    pre_optical_file: Optional[UploadFile] = File(default=None),
    post_sar_file: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    """
    Executes multi-modal change detection, damage mapping, and A* rescue routing.
    Supports either a pre-loaded disaster case study (`preset_id`) or custom uploaded
    Pre-Flood Sentinel-2 Optical + Post-Flood Sentinel-1 SAR satellite images.
    """
    ensure_demo_data_initialized()

    # Mode 1: Custom Uploaded Satellite Imagery Pair
    if pre_optical_file is not None or post_sar_file is not None:
        if pre_optical_file is None or post_sar_file is None:
            raise HTTPException(
                status_code=400,
                detail="Please upload both Pre-Disaster Optical and Post-Disaster SAR images.",
            )

        opt_bytes = await pre_optical_file.read()
        sar_bytes = await post_sar_file.read()
        optical_4ch = decode_uploaded_optical(opt_bytes)
        sar_2ch = decode_uploaded_sar(sar_bytes)

        bbox = [13.010, 80.220, 13.060, 80.270]
        if bbox_json:
            try:
                parsed = json.loads(bbox_json)
                if isinstance(parsed, list) and len(parsed) == 4:
                    bbox = [float(x) for x in parsed]
            except Exception:
                pass

        center = [
            round((bbox[0] + bbox[2]) * 0.5, 5),
            round((bbox[1] + bbox[3]) * 0.5, 5),
        ]
        scenario_meta = {
            "id": "custom_upload_mission",
            "name": custom_name or f"Custom Tactical Upload ({pre_optical_file.filename})",
            "region": "Custom Uploaded Satellite AOI",
            "event_date": "Live Mission Upload",
            "pre_date": "Pre-Event Baseline",
            "post_date": "Post-Event SAR Pass",
            "bbox": bbox,
            "center": center,
            "zoom": 14,
            "sensor_pair": "Uploaded Optical RGB/NIR + Post-Event SAR Composite",
            "description": "User-supplied multi-temporal satellite imagery analyzed via Physics-Guided Dual-Stream Attention U-Net.",
            "population_density_factor": 34,
        }
        return _execute_full_geospatial_pipeline(
            optical_4ch=optical_4ch,
            sar_2ch=sar_2ch,
            scenario_meta=scenario_meta,
        )

    # Mode 2: Pre-Loaded Disaster Case Study Preset
    chosen_id = preset_id or "chennai_monsoon_2026"
    if chosen_id not in DISASTER_PRESETS_METADATA:
        raise HTTPException(status_code=404, detail=f"Unknown preset_id: {chosen_id}")

    meta = DISASTER_PRESETS_METADATA[chosen_id]
    preset_folder = PRESETS_DIR / chosen_id

    opt_4ch = np.load(preset_folder / "optical_pre.npy")
    sar_2ch = np.load(preset_folder / "sar_post.npy")
    gt_mask = np.load(preset_folder / "ground_truth.npy")
    with open(preset_folder / "buildings.json", "r", encoding="utf-8") as f:
        buildings = json.load(f)

    return _execute_full_geospatial_pipeline(
        optical_4ch=opt_4ch,
        sar_2ch=sar_2ch,
        scenario_meta=meta,
        gt_mask=gt_mask,
        preset_buildings=buildings,
    )


@app.post("/api/gee-fetch")
async def fetch_and_analyze_gee(req: GEEFetchRequest) -> Dict[str, Any]:
    """
    Queries Google Earth Engine (or the offline Copernicus Multi-Band Simulator fallback)
    for the requested Lat/Lon bounding box and date window, then executes full flood analysis.
    """
    if req.max_lat <= req.min_lat or req.max_lon <= req.min_lon:
        raise HTTPException(
            status_code=400,
            detail="max_lat and max_lon must be strictly greater than min_lat and min_lon.",
        )

    bbox = [
        round(req.min_lat, 5),
        round(req.min_lon, 5),
        round(req.max_lat, 5),
        round(req.max_lon, 5),
    ]
    center = [
        round((bbox[0] + bbox[2]) * 0.5, 5),
        round((bbox[1] + bbox[3]) * 0.5, 5),
    ]

    opt_4ch, sar_2ch, gt_mask, gee_telemetry = gee_connector.fetch_sentinel_multimodal_tiles(
        bbox=bbox,
        pre_date=req.pre_date,
        post_date=req.post_date,
        cloud_cover_max=req.cloud_cover_max,
    )

    scenario_meta = {
        "id": "gee_live_acquisition",
        "name": req.region_name or f"Copernicus GEE Tile [{center[0]}°N, {center[1]}°E]",
        "region": f"BBox [{bbox[0]}, {bbox[1]}] to [{bbox[2]}, {bbox[3]}]",
        "event_date": req.post_date,
        "pre_date": req.pre_date,
        "post_date": req.post_date,
        "bbox": bbox,
        "center": center,
        "zoom": 14,
        "sensor_pair": "GEE Sentinel-2 SR (B4/B3/B2/B8) + Sentinel-1 GRD (VV/VH)",
        "description": (
            f"Automated Copernicus satellite acquisition ({gee_telemetry['source']}) "
            f"with max cloud threshold {req.cloud_cover_max}%."
        ),
        "population_density_factor": 30,
        "gee_telemetry": gee_telemetry,
    }

    return _execute_full_geospatial_pipeline(
        optical_4ch=opt_4ch,
        sar_2ch=sar_2ch,
        scenario_meta=scenario_meta,
        gt_mask=gt_mask,
    )


@app.post("/api/osm/refresh")
async def refresh_osm_cache(preset_id: str = Form(...)) -> Dict[str, Any]:
    """
    Phase 4 Endpoint: Manually re-downloads and refreshes real OpenStreetMap road networks
    and building footprints for the given preset AOI, updating the local GeoJSON cache.
    """
    if preset_id not in DISASTER_PRESETS_METADATA:
        raise HTTPException(status_code=404, detail=f"Unknown preset_id: {preset_id}")

    meta = DISASTER_PRESETS_METADATA[preset_id]
    roads_geo, blds_geo, from_cache, warning = osm_service.fetch_and_cache_osm_data(
        preset_id, meta["bbox"], force_refresh=True
    )
    roads_path, blds_path = osm_service.get_preset_cache_paths(preset_id)

    return {
        "preset_id": preset_id,
        "status": "REFRESHED",
        "road_features_count": len(roads_geo.get("features", [])),
        "building_features_count": len(blds_geo.get("features", [])),
        "roads_cache_file": str(roads_path.name),
        "buildings_cache_file": str(blds_path.name),
        "warning": warning,
    }


@app.post("/api/generate-sitrep")
async def create_ndrf_sitrep(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2 Module 1: Generates an Automated NDRF / District Collector AI Situation Report
    using Gemini 2.5 Flash (if GEMINI_API_KEY is set) or the deterministic NDRF Expert Rule Engine.
    """
    return generate_disaster_sitrep(payload)


@app.post("/api/simulate-timeline")
async def run_timeline_simulation(req: TimelineSimulateRequest) -> Dict[str, Any]:
    """
    Phase 2 Module 2: 48-Hour Flood Recession & Surge Forecast Simulator ([T+0h, +6h, +12h, +24h, +48h])
    using morphological hydrological propagation weighted by a DEM slope map.
    """
    return simulate_hydrological_timeline(
        preset_id=req.preset_id,
        rainfall_mm_hr=req.rainfall_mm_hr,
        horizon_hours=req.horizon_hours,
    )


@app.get("/api/benchmarks")
async def get_model_benchmarks() -> Dict[str, Any]:
    """
    Phase 2 Module 3: Returns comparative evaluation telemetry across 4 architectures
    tested on the Sen1Floods11 test split under >80% monsoon cloud cover conditions.
    """
    return get_academic_benchmark_suite()


