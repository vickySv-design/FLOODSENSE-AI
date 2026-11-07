"""Universal Study Area & AOI Management Service for FloodSense-AI.
Supports:
1. Location search / geocoding via OpenStreetMap Nominatim.
2. Geometry validation and bounding box extraction (GeoJSON, manual coordinates, polygons).
3. Area calculation in square kilometers via geodesic ellipsoidal projection.
4. Dynamic UTM zone computation.
5. Configurable area size limits and memory guardrails.
"""
import math
import json
import logging
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
import pyproj
from shapely.geometry import shape, box, Polygon, MultiPolygon

logger = logging.getLogger("floodsense.aoi_service")

# Configurable Guardrails
MIN_AOI_AREA_KM2 = 0.05
MAX_AOI_AREA_KM2 = 500.0   # Upper threshold to protect against memory exhaustion on edge devices
RECOMMENDED_MAX_KM2 = 120.0


def compute_bbox_area_km2(bbox: List[float]) -> float:
    """Computes exact metric area of a bounding box [min_lat, min_lon, max_lat, max_lon] in km^2."""
    min_lat, min_lon, max_lat, max_lon = bbox
    geod = pyproj.Geod(ellps="WGS84")
    # Polygon coordinates in lon, lat order
    lons = [min_lon, max_lon, max_lon, min_lon, min_lon]
    lats = [min_lat, min_lat, max_lat, max_lat, min_lat]
    area_m2, _ = geod.polygon_area_perimeter(lons, lats)
    return round(abs(area_m2) / 1_000_000.0, 3)


def get_utm_epsg_for_coords(lon: float, lat: float) -> int:
    """Returns local UTM EPSG code for a given coordinate."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return 32600 + zone if lat >= 0.0 else 32700 + zone


class UniversalAOIService:
    """Universal Area of Interest Manager."""

    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

    def search_location(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Geocodes a search string (city, district, river basin, landmark)
        via OpenStreetMap Nominatim API.
        """
        if not query or len(query.strip()) < 2:
            return []

        params = urllib.parse.urlencode({
            "q": query.strip(),
            "format": "json",
            "polygon_geojson": "1",
            "limit": str(limit),
            "addressdetails": "1",
        })
        url = f"{self.NOMINATIM_URL}?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FloodSenseAI-Universal-GIS/5.0 (academic disaster response platform)"},
        )

        results = []
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data:
                    # Nominatim returns boundingbox as [min_lat_str, max_lat_str, min_lon_str, max_lon_str]
                    raw_bbox = item.get("boundingbox", [])
                    if len(raw_bbox) == 4:
                        min_lat = float(raw_bbox[0])
                        max_lat = float(raw_bbox[1])
                        min_lon = float(raw_bbox[2])
                        max_lon = float(raw_bbox[3])
                        bbox = [round(min_lat, 5), round(min_lon, 5), round(max_lat, 5), round(max_lon, 5)]
                        area_km2 = compute_bbox_area_km2(bbox)

                        # Auto-clamp huge bboxes (e.g. entire countries) to a regional disaster-sized window around the center
                        lat = float(item.get("lat", 0.0))
                        lon = float(item.get("lon", 0.0))
                        if area_km2 > MAX_AOI_AREA_KM2:
                            # Clamp to ~8km x ~8km around center
                            delta_lat = 0.04
                            delta_lon = 0.04
                            bbox = [round(lat - delta_lat, 5), round(lon - delta_lon, 5), round(lat + delta_lat, 5), round(lon + delta_lon, 5)]
                            area_km2 = compute_bbox_area_km2(bbox)

                        results.append({
                            "display_name": item.get("display_name"),
                            "name": item.get("name") or query,
                            "type": item.get("type", "location"),
                            "lat": lat,
                            "lon": lon,
                            "bbox": bbox,
                            "area_km2": area_km2,
                            "geojson": item.get("geojson"),
                        })
        except Exception as e:
            logger.warning(f"Nominatim geocoding failed: {e}")

        return results

    def validate_and_normalize_aoi(
        self,
        name: str,
        bbox: Optional[List[float]] = None,
        geojson_geometry: Optional[Dict[str, Any]] = None,
        terrain_type: str = "auto",
    ) -> Dict[str, Any]:
        """
        Validates arbitrary user-selected AOI bounds or polygon.
        Enforces safety guardrails, computes area, local UTM CRS, and terrain profile.
        """
        if geojson_geometry:
            try:
                geom = shape(geojson_geometry)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                min_lon, min_lat, max_lon, max_lat = geom.bounds
                bbox = [round(min_lat, 5), round(min_lon, 5), round(max_lat, 5), round(max_lon, 5)]
            except Exception as e:
                raise ValueError(f"Invalid GeoJSON polygon geometry: {e}")

        if not bbox or len(bbox) != 4:
            raise ValueError("Bounding box must be provided as [min_lat, min_lon, max_lat, max_lon].")

        min_lat, min_lon, max_lat, max_lon = [float(c) for c in bbox]

        # Basic range validation
        if not (-90.0 <= min_lat < max_lat <= 90.0):
            raise ValueError(f"Invalid latitude range: [{min_lat}, {max_lat}]")
        if not (-180.0 <= min_lon < max_lon <= 180.0):
            raise ValueError(f"Invalid longitude range: [{min_lon}, {max_lon}]")

        area_km2 = compute_bbox_area_km2(bbox)

        if area_km2 < MIN_AOI_AREA_KM2:
            raise ValueError(f"Selected study area is too small ({area_km2:.3f} km²). Minimum allowable area is {MIN_AOI_AREA_KM2} km².")

        if area_km2 > MAX_AOI_AREA_KM2:
            raise ValueError(
                f"Selected study area ({area_km2:.1f} km²) exceeds system processing limits ({MAX_AOI_AREA_KM2} km²). "
                "Please zoom in or select a sub-watershed sector."
            )

        center_lat = round((min_lat + max_lat) * 0.5, 5)
        center_lon = round((min_lon + max_lon) * 0.5, 5)
        utm_epsg = get_utm_epsg_for_coords(center_lon, center_lat)

        # Auto-detect terrain type if 'auto'
        if terrain_type == "auto":
            # If within 0.5 degrees of coastal longitudes in southern/eastern India or low elevation
            # Heuristic detection; can be refined with DEM
            if abs(center_lat) < 20.0 and (abs(center_lon - 80.25) < 0.2 or abs(center_lon) > 100.0):
                detected_terrain = "coastal_urban"
            elif center_lat > 25.0 and center_lon > 90.0:
                detected_terrain = "river_floodplain"
            elif center_lat < 15.0 and 75.0 <= center_lon <= 77.0:
                detected_terrain = "mountainous_catchment"
            else:
                detected_terrain = "urban_drainage_basin"
        else:
            detected_terrain = terrain_type

        # Generate a deterministic clean AOI ID
        slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:24].strip("_") or "custom_aoi"
        coord_key = f"{int(abs(min_lat * 100))}_{int(abs(min_lon * 100))}"
        aoi_id = f"aoi_{slug}_{coord_key}"

        return {
            "id": aoi_id,
            "name": name,
            "bbox": [min_lat, min_lon, max_lat, max_lon],
            "center": [center_lat, center_lon],
            "area_km2": area_km2,
            "utm_epsg": utm_epsg,
            "utm_crs": f"EPSG:{utm_epsg}",
            "geographic_crs": "EPSG:4326",
            "terrain_type": detected_terrain,
            "is_within_recommended_size": area_km2 <= RECOMMENDED_MAX_KM2,
            "recommended_max_km2": RECOMMENDED_MAX_KM2,
            "geometry": geojson_geometry or {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]],
            },
        }


aoi_service = UniversalAOIService()

