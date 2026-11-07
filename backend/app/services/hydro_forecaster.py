import json
from typing import Any, Dict, List
import cv2
import numpy as np

from app.config import (
    CLASS_NEW_FLOOD,
    CLASS_PERMANENT_WATER,
    DEVICE,
    DISASTER_PRESETS_METADATA,
    HIGH_RISK_BUFFER_PIXELS,
    PIXEL_AREA_KM2,
    PRESETS_DIR,
)
from app.services.damage_analyzer import (
    extract_mask_geojson_features,
    generate_deterministic_buildings,
    lonlat_to_pixel,
)
from app.services.inference_engine import inference_engine
from app.services.preprocessor import (
    compute_high_risk_buffer_mask,
    preprocess_multimodal_pair,
    render_visual_layers,
)
from scripts.init_demo_data import ensure_demo_data_initialized


def _propagate_flood_mask_dem(
    base_seg_mask: np.ndarray,
    dem_elevation: np.ndarray,
    rainfall_mm_hr: float,
    horizon_hours: int,
) -> np.ndarray:
    """
    Morphological Hydrological Propagation weighted by Digital Elevation Model (DEM) slope:
    - Equilibrium drainage rate is ~22 mm/hr.
    - When rainfall_mm_hr > 22 mm/hr: net runoff accumulation causes flood surge expansion
      via distance-transform & DEM-weighted morphological dilation.
    - When rainfall_mm_hr < 22 mm/hr: natural basin drainage & soil infiltration cause
      flood recession via DEM-weighted morphological erosion (preserving Class 1 Permanent Water).
    """
    if horizon_hours <= 0:
        return base_seg_mask.copy()

    perm_water = base_seg_mask == CLASS_PERMANENT_WATER
    base_flood = base_seg_mask == CLASS_NEW_FLOOD
    combined_water = (perm_water | base_flood).astype(np.uint8)

    # Net hydrological forcing in [-22 .. +128] mm/hr
    net_rain = rainfall_mm_hr - 22.0
    time_factor = np.sqrt(horizon_hours / 6.0)  # non-linear basin filling/draining curve

    out_mask = np.zeros_like(base_seg_mask, dtype=np.uint8)
    out_mask[perm_water] = CLASS_PERMANENT_WATER

    if abs(net_rain) <= 3.0:
        # Near steady-state hydrology
        out_mask[base_flood & (~perm_water)] = CLASS_NEW_FLOOD
        return out_mask

    if net_rain > 3.0:
        # SURGE EXPANSION: Distance from existing water + low DEM elevation determines inundation
        inv_water = (1 - combined_water).astype(np.uint8)
        dist_from_water = cv2.distanceTransform(inv_water, cv2.DIST_L2, 5)

        # Maximum pixel expansion radius scales with rainfall intensity and time horizon
        max_expand_px = (net_rain / 125.0) * 14.0 * time_factor
        # Low-elevation floodplains (low dem_elevation) flood further than steep ridges
        effective_dist = dist_from_water * (0.55 + 0.90 * dem_elevation)

        newly_inundated = (effective_dist <= max_expand_px) & (~perm_water)
        final_flood = (base_flood | newly_inundated) & (~perm_water)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_flood_u8 = cv2.morphologyEx(
            final_flood.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )
        out_mask[final_flood_u8 == 1] = CLASS_NEW_FLOOD
        out_mask[perm_water] = CLASS_PERMANENT_WATER
        return out_mask

    # DRAINAGE RECESSION (net_rain < -3.0): Higher elevation margins recede first
    recession_strength = (abs(net_rain) / 22.0) * 4.8 * time_factor
    dist_inside_flood = cv2.distanceTransform(
        base_flood.astype(np.uint8), cv2.DIST_L2, 5
    )
    # Higher elevation pixels inside the flood zone drain faster
    retention_score = dist_inside_flood * (1.15 - 0.55 * dem_elevation)
    remaining_flood = (retention_score > recession_strength) & base_flood & (~perm_water)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    remaining_u8 = cv2.morphologyEx(
        remaining_flood.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )
    out_mask[remaining_u8 == 1] = CLASS_NEW_FLOOD
    out_mask[perm_water] = CLASS_PERMANENT_WATER
    return out_mask


def simulate_hydrological_timeline(
    preset_id: str = "chennai_monsoon_2026",
    rainfall_mm_hr: float = 65.0,
    horizon_hours: int = 24,
) -> Dict[str, Any]:
    """
    Runs a 48-hour spatio-temporal flood progression / recession simulation across
    horizons [0, 6, 12, 24, 48] hours and renders the updated visual mask & GeoJSON
    for the requested `horizon_hours`.
    """
    ensure_demo_data_initialized()
    chosen_id = (
        preset_id
        if preset_id in DISASTER_PRESETS_METADATA
        else "chennai_monsoon_2026"
    )
    meta = DISASTER_PRESETS_METADATA[chosen_id]
    bbox = meta["bbox"]
    preset_folder = PRESETS_DIR / chosen_id

    opt_4ch = np.load(preset_folder / "optical_pre.npy")
    sar_2ch = np.load(preset_folder / "sar_post.npy")
    gt_mask = np.load(preset_folder / "ground_truth.npy")
    bld_path = preset_folder / "buildings.json"
    if bld_path.exists():
        with open(bld_path, "r", encoding="utf-8") as f:
            buildings = json.load(f)
    else:
        buildings = generate_deterministic_buildings(bbox, count=54)

    preprocessed = preprocess_multimodal_pair(opt_4ch, sar_2ch, DEVICE)
    inf_out = inference_engine.run_inference(
        preprocessed=preprocessed,
        ground_truth_mask=gt_mask,
        preset_metrics=meta.get("historical_metrics"),
    )
    base_seg_mask: np.ndarray = inf_out["seg_mask"]  # type: ignore[assignment]
    attention_map: np.ndarray = inf_out["attention_map"]  # type: ignore[assignment]

    # Derive normalized Digital Elevation Model (DEM) proxy from Optical Red/NIR + smooth terrain gradient
    raw_dem = 0.55 * opt_4ch[0] + 0.45 * opt_4ch[3]
    dem_smooth = cv2.GaussianBlur(raw_dem, (15, 15), 0)
    dem_norm = (dem_smooth - dem_smooth.min()) / (
        dem_smooth.max() - dem_smooth.min() + 1e-6
    )

    horizons = [0, 6, 12, 24, 48]
    selected_horizon = horizon_hours if horizon_hours in horizons else 24

    h, w = base_seg_mask.shape[:2]
    perm_water_km2 = round(
        float(np.sum(base_seg_mask == CLASS_PERMANENT_WATER) * PIXEL_AREA_KM2), 3
    )
    t0_flood_km2 = round(
        float(np.sum(base_seg_mask == CLASS_NEW_FLOOD) * PIXEL_AREA_KM2), 3
    )

    timeline_series: List[Dict[str, Any]] = []
    selected_mask = base_seg_mask.copy()

    for hr in horizons:
        sim_mask = _propagate_flood_mask_dem(
            base_seg_mask=base_seg_mask,
            dem_elevation=dem_norm,
            rainfall_mm_hr=rainfall_mm_hr,
            horizon_hours=hr,
        )
        if hr == selected_horizon:
            selected_mask = sim_mask.copy()

        sim_buffer = compute_high_risk_buffer_mask(sim_mask, HIGH_RISK_BUFFER_PIXELS)
        flood_px = int(np.sum(sim_mask == CLASS_NEW_FLOOD))
        buf_px = int(np.sum(sim_buffer))

        proj_flood_km2 = round(flood_px * PIXEL_AREA_KM2, 3)
        proj_buf_km2 = round(buf_px * PIXEL_AREA_KM2, 3)
        delta_pct = round(
            ((proj_flood_km2 - t0_flood_km2) / max(t0_flood_km2, 0.01)) * 100.0, 1
        )

        # Count buildings submerged or in buffer at horizon T+hr
        crit_cnt = 0
        buf_cnt = 0
        for bld in buildings:
            px, py = lonlat_to_pixel(bld["lon"], bld["lat"], bbox, w, h)
            y0, y1 = max(0, py - 1), min(h, py + 2)
            x0, x1 = max(0, px - 1), min(w, px + 2)
            p_seg = sim_mask[y0:y1, x0:x1]
            p_buf = sim_buffer[y0:y1, x0:x1]
            if np.any(p_seg == CLASS_NEW_FLOOD) or np.any(
                p_seg == CLASS_PERMANENT_WATER
            ):
                crit_cnt += 1
            elif np.any(p_buf):
                buf_cnt += 1

        timeline_series.append(
            {
                "horizon_label": f"T+{hr}h" if hr > 0 else "T+0h (Now)",
                "hours": hr,
                "projected_flood_km2": proj_flood_km2,
                "permanent_water_km2": perm_water_km2,
                "buffer_risk_km2": proj_buf_km2,
                "critical_submerged_buildings": crit_cnt,
                "buffer_risk_buildings": buf_cnt,
                "at_risk_buildings": crit_cnt + buf_cnt,
                "delta_from_t0_pct": delta_pct,
            }
        )

    # Render updated visual overlay for the selected horizon
    visuals = render_visual_layers(
        optical_4ch=preprocessed["optical_clean"],  # type: ignore[arg-type]
        sar_filtered=preprocessed["sar_filtered"],  # type: ignore[arg-type]
        ndwi=preprocessed["ndwi"],  # type: ignore[arg-type]
        attention_map=attention_map,
        seg_mask=selected_mask,
    )

    forecast_flood_features = extract_mask_geojson_features(
        selected_mask == CLASS_NEW_FLOOD,
        bbox=bbox,
        feature_type="forecast_flood",
        label=f"Projected Flood Inundation (T+{selected_horizon}h @ {rainfall_mm_hr} mm/hr)",
        color="#f43f5e",
        min_area_px=10.0,
    )

    if rainfall_mm_hr > 25.0:
        regime = "SURGE_EXPANSION (Active Rainfall Accumulation)"
    elif rainfall_mm_hr < 19.0:
        regime = "DRAINAGE_RECESSION (Post-Storm Basin Outflow)"
    else:
        regime = "EQUILIBRIUM_STEADY_STATE (Inflow ≈ Outflow)"

    selected_entry = next(
        (item for item in timeline_series if item["hours"] == selected_horizon),
        timeline_series[-1],
    )

    return {
        "preset_id": chosen_id,
        "scenario_name": meta["name"],
        "rainfall_mm_hr": rainfall_mm_hr,
        "selected_horizon_hours": selected_horizon,
        "hydrological_regime": regime,
        "selected_projection": selected_entry,
        "timeline": timeline_series,
        "forecast_overlay_b64": visuals["flood_overlay"],
        "forecast_flood_geojson": {
            "type": "FeatureCollection",
            "features": forecast_flood_features,
        },
    }

