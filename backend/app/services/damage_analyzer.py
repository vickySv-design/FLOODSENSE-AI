from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from shapely.geometry import Polygon, mapping, shape

from app.config import (
    CLASS_DRY_LAND,
    CLASS_NEW_FLOOD,
    CLASS_PERMANENT_WATER,
    HIGH_RISK_BUFFER_PIXELS,
    PIXEL_AREA_KM2,
)
from app.services.georef import GeoSpatialTransform
from app.services.osm_service import osm_service
from app.services.preprocessor import compute_high_risk_buffer_mask


def generate_deterministic_buildings(
    bbox: List[float], count: int = 92, seed: int = 2026
) -> List[Dict[str, Any]]:
    """
    Fallback deterministic building compound generator if offline or synthetic demo selected.
    """
    rng = np.random.default_rng(seed)
    min_lat, min_lon, max_lat, max_lon = bbox
    buildings: List[Dict[str, Any]] = []
    for i in range(count):
        u = float(rng.uniform(0.08, 0.92))
        v = float(rng.uniform(0.08, 0.92))
        lat = min_lat + u * (max_lat - min_lat)
        lon = min_lon + v * (max_lon - min_lon)
        buildings.append({
            "id": f"BLD-{101 + i}",
            "name": f"Structure #{101 + i}",
            "category": "Building",
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "elevation_m": round(float(rng.uniform(4.0, 15.0)), 1),
            "estimated_occupants": int(rng.integers(15, 80)),
        })
    return buildings


def lonlat_to_pixel(
    lon: float, lat: float, bbox: List[float], width: int = 256, height: int = 256
) -> Tuple[int, int]:
    georef = GeoSpatialTransform(bbox, width=width, height=height)
    col, row = georef.lonlat_to_pixel(lon, lat)
    return int(round(np.clip(col, 0, width - 1))), int(round(np.clip(row, 0, height - 1)))


def pixel_to_lonlat(
    col: float, row: float, bbox: List[float], width: int = 256, height: int = 256
) -> Tuple[float, float]:
    georef = GeoSpatialTransform(bbox, width=width, height=height)
    return georef.pixel_to_lonlat(col, row)


def extract_mask_geojson_features(
    binary_mask: np.ndarray,
    bbox: List[float],
    feature_type: str,
    label: str,
    color: str,
    min_area_px: float = 12.0,
) -> List[Dict[str, Any]]:
    """
    Extracts vector polygon contours from a binary mask using cv2.findContours + Shapely
    and transforms pixel coordinates into WGS84 GeoJSON [longitude, latitude] coordinates.
    """
    h, w = binary_mask.shape
    georef = GeoSpatialTransform(bbox, width=w, height=h)
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    features: List[Dict[str, Any]] = []
    for idx, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px or len(cnt) < 4:
            continue

        epsilon = 0.0035 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx if len(approx) >= 4 else cnt

        ring_coords: List[List[float]] = []
        for pt in pts:
            px, py = float(pt[0][0]), float(pt[0][1])
            lon, lat = georef.pixel_to_lonlat(px, py)
            ring_coords.append([lon, lat])

        if len(ring_coords) < 3:
            continue
        if ring_coords[0] != ring_coords[-1]:
            ring_coords.append(ring_coords[0])

        try:
            poly = Polygon(ring_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            geom = mapping(poly)
        except Exception:
            geom = {"type": "Polygon", "coordinates": [ring_coords]}

        area_km2 = round(float(area_px * PIXEL_AREA_KM2), 4)
        features.append(
            {
                "type": "Feature",
                "id": f"{feature_type}_{idx}",
                "properties": {
                    "feature_type": feature_type,
                    "label": label,
                    "area_km2": area_km2,
                    "area_pixels": int(area_px),
                    "color": color,
                },
                "geometry": geom,
            }
        )
    return features


def analyze_building_exposure(
    buildings: List[Dict[str, Any]],
    seg_mask: np.ndarray,
    georef: GeoSpatialTransform,
    max_buildings: int = 500,
    nodata_val: float = 255.0,
    flood_val: int = CLASS_NEW_FLOOD,
) -> Dict[str, Any]:
    """
    Evaluates real OSM building polygons against the georeferenced flood classification raster.
    Computes exact footprint intersection, valid sampled pixels, and exposed area percentage.

    Classifications (Phase 5 Standard):
    - HIGH_EXPOSURE: >= 50% footprint overlap with mapped flood mask.
    - MODERATE_EXPOSURE: 10% to 50% overlap.
    - LOW_OR_NO_DETECTED_EXPOSURE: < 10% overlap.
    - UNKNOWN: Outside raster bounds, invalid geometry, or insufficient valid pixels.
    """
    total_raw = len(buildings)
    is_osm = total_raw > 0
    bld_source = "OPENSTREETMAP_REAL" if is_osm else "SYNTHETIC_TEST_FIXTURE"

    # Sample representative subset if count is large to maintain sub-second response
    if total_raw > max_buildings:
        step = max(1, total_raw // max_buildings)
        selected_blds = buildings[::step][:max_buildings]
    else:
        selected_blds = buildings

    graded_features: List[Dict[str, Any]] = []
    counts = {
        "HIGH_EXPOSURE": 0,
        "MODERATE_EXPOSURE": 0,
        "LOW_OR_NO_DETECTED_EXPOSURE": 0,
        "UNKNOWN": 0,
    }

    color_map = {
        "HIGH_EXPOSURE": "#ef4444",
        "MODERATE_EXPOSURE": "#f59e0b",
        "LOW_OR_NO_DETECTED_EXPOSURE": "#10b981",
        "UNKNOWN": "#94a3b8",
    }

    for idx, bld in enumerate(selected_blds):
        props = bld.get("properties", {})
        geom = bld.get("geometry", {})
        bld_id = props.get("osm_id", props.get("id", f"bld_{idx}"))
        bld_name = props.get("name", f"OSM Structure #{bld_id}")
        bld_type = props.get("building_type", props.get("building", "building"))

        # Compute centroid coordinates
        c_lon = props.get("center_lon")
        c_lat = props.get("center_lat")
        poly = georef.validate_and_fix_polygon(geom)

        if poly is not None:
            if c_lon is None or c_lat is None:
                c_lon = float(poly.centroid.x)
                c_lat = float(poly.centroid.y)
            stats = georef.calculate_polygon_overlap_stats(
                poly=poly,
                raster=seg_mask,
                nodata_val=nodata_val,
                flood_val=flood_val,
            )
        else:
            stats = {
                "exposure_category": "UNKNOWN",
                "exposed_area_pct": 0.0,
                "valid_pixels": 0,
                "flood_pixels": 0,
                "reason": "INVALID_GEOMETRY",
            }
            if c_lon is None:
                c_lon = georef.min_lon
            if c_lat is None:
                c_lat = georef.min_lat

        cat = stats["exposure_category"]
        counts[cat] += 1
        color = color_map.get(cat, "#94a3b8")

        feature = {
            "type": "Feature",
            "id": f"bld/{bld_id}",
            "properties": {
                "building_id": str(bld_id),
                "osm_id": str(bld_id),
                "building_name": bld_name,
                "name": bld_name,
                "building_type": bld_type,
                "exposure_category": cat,
                "damage_status": cat,  # Backwards compatibility
                "exposed_area_pct": stats["exposed_area_pct"],
                "valid_sampled_pixels": stats["valid_pixels"],
                "flood_overlap_pixels": stats["flood_pixels"],
                "color": color,
                "center_lon": round(float(c_lon), 6),
                "center_lat": round(float(c_lat), 6),
                "estimated_water_depth_ft": None,  # Academic integrity: depth is not measured on binary masks
                "elevation_m": None,
                "data_source": bld_source,
                "scientific_disclaimer": "Spatial exposure estimate via flood-risk raster intersection; does not confirm physical structural damage.",
            },
            "geometry": geom if geom else (mapping(poly) if poly else None),
        }
        graded_features.append(feature)

    buildings_list = [
        {
            "id": f["properties"]["building_id"],
            "name": f["properties"]["building_name"],
            "category": f["properties"]["building_type"],
            "exposure_category": f["properties"]["exposure_category"],
            "status": f["properties"]["damage_status"],
            "color": f["properties"]["color"],
            "exposed_area_pct": f["properties"]["exposed_area_pct"],
            "valid_sampled_pixels": f["properties"]["valid_sampled_pixels"],
            "flood_overlap_pixels": f["properties"]["flood_overlap_pixels"],
            "lat": f["properties"]["center_lat"],
            "lon": f["properties"]["center_lon"],
            "estimated_occupants": 35,
            "estimated_water_depth_ft": None,
            "data_source": bld_source,
        }
        for f in graded_features
    ]

    summary = {
        "total_buildings_analyzed": len(graded_features),
        "total_osm_buildings_available": total_raw,
        "high_exposure_count": counts["HIGH_EXPOSURE"],
        "moderate_exposure_count": counts["MODERATE_EXPOSURE"],
        "low_exposure_count": counts["LOW_OR_NO_DETECTED_EXPOSURE"],
        "unknown_exposure_count": counts["UNKNOWN"],
        # Backwards compatible keys
        "critical_submerged": counts["HIGH_EXPOSURE"],
        "moderate_buffer_risk": counts["MODERATE_EXPOSURE"],
        "safe_dry": counts["LOW_OR_NO_DETECTED_EXPOSURE"],
        "total_buildings": len(graded_features),
        "data_source": bld_source,
    }

    return {
        "summary": summary,
        "features": graded_features,
        "buildings_list": buildings_list,
        "buildings_geojson": {
            "type": "FeatureCollection",
            "features": graded_features,
        },
    }


def analyze_flood_damage(
    seg_mask: np.ndarray,
    bbox: List[float],
    preset_id: str,
    population_density_factor: int = 32,
) -> Dict[str, Any]:
    """
    Georeferenced Damage & Exposure Mapping Pipeline:
    1. Computes pixel-to-km2 spatial telemetry using affine transform.
    2. Extracts real-world WGS84 GeoJSON contours for Permanent Water, New Flood, and 100m Risk Buffer.
    3. Loads real OpenStreetMap building footprints for the preset AOI.
    4. Computes building exposure categories using polygon footprint intersection.
    """
    h, w = seg_mask.shape[:2]
    georef = GeoSpatialTransform(bbox, width=w, height=h)
    buffer_mask = compute_high_risk_buffer_mask(seg_mask, HIGH_RISK_BUFFER_PIXELS)

    # Pixel telemetry calculations
    perm_water_px = int(np.sum(seg_mask == CLASS_PERMANENT_WATER))
    flood_px = int(np.sum(seg_mask == CLASS_NEW_FLOOD))
    dry_px = int(np.sum(seg_mask == CLASS_DRY_LAND))
    buffer_px = int(np.sum(buffer_mask))

    permanent_water_km2 = round(perm_water_px * PIXEL_AREA_KM2, 3)
    flooded_area_km2 = round(flood_px * PIXEL_AREA_KM2, 3)
    dry_land_km2 = round(dry_px * PIXEL_AREA_KM2, 3)
    buffer_risk_km2 = round(buffer_px * PIXEL_AREA_KM2, 3)
    total_aoi_km2 = round((h * w) * PIXEL_AREA_KM2, 3)

    flood_expansion_pct = round(
        (flooded_area_km2 / max(permanent_water_km2, 0.01)) * 100.0, 1
    )
    inundated_aoi_pct = round((flooded_area_km2 / max(total_aoi_km2, 0.01)) * 100.0, 1)

    # Extract GeoJSON contours in [longitude, latitude] order
    perm_water_features = extract_mask_geojson_features(
        seg_mask == CLASS_PERMANENT_WATER,
        bbox=bbox,
        feature_type="permanent_water",
        label="Baseline Permanent Water Body (Class 1)",
        color="#06b6d4",
        min_area_px=10.0,
    )
    flood_features = extract_mask_geojson_features(
        seg_mask == CLASS_NEW_FLOOD,
        bbox=bbox,
        feature_type="new_flood",
        label="Active Flood Inundation Zone (Class 2)",
        color="#f43f5e",
        min_area_px=8.0,
    )
    buffer_features = extract_mask_geojson_features(
        buffer_mask,
        bbox=bbox,
        feature_type="high_risk_buffer",
        label="100m High-Risk Flood Buffer",
        color="#f59e0b",
        min_area_px=14.0,
    )

    # Load Real OpenStreetMap Building Footprints
    _, blds_geo, from_cache, warning = osm_service.fetch_and_cache_osm_data(preset_id, bbox)
    raw_blds = blds_geo.get("features", [])

    exposure_res = analyze_building_exposure(
        buildings=raw_blds,
        seg_mask=seg_mask,
        georef=georef,
        max_buildings=250,
        nodata_val=255.0,
        flood_val=CLASS_NEW_FLOOD,
    )

    graded_buildings_geojson = exposure_res["features"]
    buildings_list = exposure_res["buildings_list"]
    buildings_summary = exposure_res["summary"]
    bld_data_source = buildings_summary["data_source"]

    # Estimated population exposure proportional to high and moderate exposure
    affected_population = int(
        buildings_summary["high_exposure_count"] * population_density_factor * 1.5
        + buildings_summary["moderate_exposure_count"] * population_density_factor * 0.6
    )
    total_affected_pop = max(
        affected_population,
        int(flooded_area_km2 * population_density_factor * 18),
    )
    buildings_summary["estimated_affected_population"] = total_affected_pop

    total_affected_pop = max(affected_population, int(flooded_area_km2 * population_density_factor * 18))

    hydrology_chart_data = [
        {"category": "Permanent Water", "pre_disaster_km2": permanent_water_km2, "post_disaster_km2": permanent_water_km2},
        {"category": "Flood Inundation", "pre_disaster_km2": 0.0, "post_disaster_km2": flooded_area_km2},
        {"category": "100m Risk Buffer", "pre_disaster_km2": 0.0, "post_disaster_km2": buffer_risk_km2},
        {"category": "Safe Dry Land", "pre_disaster_km2": round(total_aoi_km2 - permanent_water_km2, 3), "post_disaster_km2": round(max(dry_land_km2 - buffer_risk_km2, 0.0), 3)},
    ]

    return {
        "telemetry": {
            "permanent_water_km2": permanent_water_km2,
            "flooded_area_km2": flooded_area_km2,
            "buffer_risk_km2": buffer_risk_km2,
            "dry_land_km2": dry_land_km2,
            "total_aoi_km2": total_aoi_km2,
            "flood_expansion_pct": flood_expansion_pct,
            "inundated_aoi_pct": inundated_aoi_pct,
            "permanent_water_pixels": perm_water_px,
            "flooded_pixels": flood_px,
            "spatial_resolution_m": 10.0,
        },
        "buildings_summary": {
            "total_buildings": len(graded_buildings_geojson),
            "critical_submerged": buildings_summary.get("critical_submerged", buildings_summary.get("high_exposure_count", 0)),
            "moderate_buffer_risk": buildings_summary.get("moderate_buffer_risk", buildings_summary.get("moderate_exposure_count", 0)),
            "safe_dry": buildings_summary.get("safe_dry", buildings_summary.get("low_exposure_count", 0)),
            "high_exposure_count": buildings_summary.get("high_exposure_count", 0),
            "moderate_exposure_count": buildings_summary.get("moderate_exposure_count", 0),
            "low_exposure_count": buildings_summary.get("low_exposure_count", 0),
            "unknown_exposure_count": buildings_summary.get("unknown_exposure_count", 0),
            "estimated_affected_population": total_affected_pop,
            "data_source": bld_data_source,
        },
        "buildings": buildings_list,
        "buildings_geojson": {
            "type": "FeatureCollection",
            "features": graded_buildings_geojson,
        },
        "geojson_layers": {
            "permanent_water": {"type": "FeatureCollection", "features": perm_water_features},
            "new_flood": {"type": "FeatureCollection", "features": flood_features},
            "high_risk_buffer": {"type": "FeatureCollection", "features": buffer_features},
        },
        "hydrology_chart_data": hydrology_chart_data,
        "buffer_mask": buffer_mask,
        "data_provenance": {
            "building_data_source": bld_data_source,
            "osm_cache_used": from_cache,
            "total_osm_buildings": len(raw_blds),
            "data_warnings": warning,
        },
    }
