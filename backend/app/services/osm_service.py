from datetime import datetime, timezone
import json
from pathlib import Path
import ssl
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

from app.config import OSM_CACHE_DIR, DISASTER_PRESETS_METADATA

OVERPASS_SERVERS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

OSM_DIRECT_BASE = "https://api.openstreetmap.org/api/0.6/map.json"

HIGHWAY_FILTER = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|track"
)


class OSMDataService:
    """
    Manages live downloading and local caching of real OpenStreetMap road networks
    and building footprints for disaster AOIs.
    Features multi-server fallback, sub-bounding box tiling, and robust caching.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or OSM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ssl_context = ssl._create_unverified_context()

    def get_preset_cache_paths(self, preset_id: str) -> Tuple[Path, Path]:
        clean_id = preset_id.replace("_monsoon_2026", "").replace("_surge", "").replace("_flashflood", "").replace("_kerala", "")
        # Standard filenames required by Phase 4 directive
        if "chennai" in clean_id:
            prefix = "chennai"
        elif "assam" in clean_id:
            prefix = "assam"
        elif "wayanad" in clean_id:
            prefix = "wayanad"
        else:
            prefix = clean_id

        roads_path = self.cache_dir / f"{prefix}_roads.geojson"
        buildings_path = self.cache_dir / f"{prefix}_buildings.geojson"
        return roads_path, buildings_path

    def _query_overpass(self, query: str, timeout_sec: int = 25) -> Optional[Dict[str, Any]]:
        """Queries Overpass servers with failover."""
        for server in OVERPASS_SERVERS:
            try:
                req = urllib.request.Request(
                    server,
                    data=query.encode("utf-8"),
                    headers={"User-Agent": "FloodSenseAI-Capstone/4.0 (academic disaster response research)"},
                )
                with urllib.request.urlopen(req, context=self.ssl_context, timeout=timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    if "elements" in data and len(data["elements"]) > 0:
                        return data
            except Exception:
                continue
        return None

    def _query_osm_direct_tile(self, bbox: List[float], timeout_sec: int = 15) -> Optional[Dict[str, Any]]:
        """
        Direct OSM 0.6 API query for small bbox: [min_lat, min_lon, max_lat, max_lon].
        Format: ?bbox=min_lon,min_lat,max_lon,max_lat
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        url = f"{OSM_DIRECT_BASE}?bbox={min_lon:.5f},{min_lat:.5f},{max_lon:.5f},{max_lat:.5f}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FloodSenseAI-Capstone/4.0 (academic disaster response research)"},
            )
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception:
            return None

    def _osm_elements_to_geojson(
        self, elements: List[Dict[str, Any]], bbox: List[float]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Parses raw OSM JSON elements into valid Road and Building GeoJSON FeatureCollections.
        Enforces standard [longitude, latitude] coordinate ordering in EPSG:4326.
        """
        nodes: Dict[int, Tuple[float, float]] = {}
        for elem in elements:
            if elem.get("type") == "node":
                nid = elem.get("id")
                lon = float(elem.get("lon"))
                lat = float(elem.get("lat"))
                nodes[nid] = (lon, lat)

        min_lat, min_lon, max_lat, max_lon = bbox

        road_features: List[Dict[str, Any]] = []
        building_features: List[Dict[str, Any]] = []

        for elem in elements:
            if elem.get("type") != "way":
                continue

            way_id = elem.get("id")
            tags = elem.get("tags", {})
            way_nodes = elem.get("nodes", [])

            # Reconstruct geometry: coordinates in [longitude, latitude] order
            coords: List[List[float]] = []
            for nid in way_nodes:
                if nid in nodes:
                    coords.append([nodes[nid][0], nodes[nid][1]])

            if len(coords) < 2:
                continue

            # Highway filtering
            if "highway" in tags:
                hwy = tags.get("highway")
                # Filter out irrelevant footways / paths
                if hwy in [
                    "motorway", "trunk", "primary", "secondary", "tertiary",
                    "unclassified", "residential", "living_street", "service", "track",
                    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"
                ]:
                    # Clip to AOI (at least one point within bounds)
                    has_inside = any(
                        (min_lon <= c[0] <= max_lon and min_lat <= c[1] <= max_lat) for c in coords
                    )
                    if not has_inside:
                        continue

                    # Extract bridge / tunnel / oneway attributes
                    is_bridge = tags.get("bridge") in ["yes", "viaduct", "aqueduct"]
                    is_tunnel = tags.get("tunnel") in ["yes", "building_passage"]
                    is_oneway = tags.get("oneway") in ["yes", "1", "true"]

                    road_features.append(
                        {
                            "type": "Feature",
                            "id": f"way/{way_id}",
                            "properties": {
                                "osm_id": way_id,
                                "name": tags.get("name", tags.get("ref", f"OSM {hwy.capitalize()} {way_id}")),
                                "highway": hwy,
                                "is_bridge": is_bridge,
                                "is_tunnel": is_tunnel,
                                "is_oneway": is_oneway,
                                "surface": tags.get("surface", "paved"),
                                "access": tags.get("access", "yes"),
                                "maxspeed": tags.get("maxspeed", None),
                                "data_source": "OPENSTREETMAP_LIVE",
                            },
                            "geometry": {
                                "type": "LineString",
                                "coordinates": coords,
                            },
                        }
                    )

            # Building footprint filtering
            if "building" in tags and len(coords) >= 4:
                # Closed polygon check
                if coords[0] != coords[-1]:
                    coords.append(coords[0])

                center_lon = sum(c[0] for c in coords) / len(coords)
                center_lat = sum(c[1] for c in coords) / len(coords)
                if not (min_lon <= center_lon <= max_lon and min_lat <= center_lat <= max_lat):
                    continue

                bld_name = tags.get("name", f"OSM Structure #{way_id}")
                bld_type = tags.get("building", "yes")

                building_features.append(
                    {
                        "type": "Feature",
                        "id": f"bld/{way_id}",
                        "properties": {
                            "osm_id": way_id,
                            "name": bld_name,
                            "building_type": bld_type,
                            "center_lon": round(center_lon, 6),
                            "center_lat": round(center_lat, 6),
                            "data_source": "OPENSTREETMAP_LIVE",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords],
                        },
                    }
                )

        roads_geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "source": "OpenStreetMap contributors (ODbL)",
                "osm_timestamp": datetime.now(timezone.utc).isoformat(),
                "bbox": bbox,
                "highway_filter": HIGHWAY_FILTER,
                "features_count": len(road_features),
            },
            "features": road_features,
        }

        buildings_geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "source": "OpenStreetMap contributors (ODbL)",
                "osm_timestamp": datetime.now(timezone.utc).isoformat(),
                "bbox": bbox,
                "features_count": len(building_features),
            },
            "features": building_features,
        }

        return roads_geojson, buildings_geojson

    def fetch_and_cache_osm_data(
        self, preset_id: str, bbox: List[float], force_refresh: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any], bool, Optional[str]]:
        """
        Retrieves real OSM roads and buildings for an AOI bbox.
        Priority:
        1. Local cache if exists and not force_refresh
        2. Live Overpass API with multi-mirror failover
        3. Live Direct OSM 0.6 tile-splitting API
        4. Graceful error reporting (explicit warning if unavailable)
        """
        roads_path, blds_path = self.get_preset_cache_paths(preset_id)

        # 1. Check local cache
        if not force_refresh and roads_path.exists() and blds_path.exists():
            try:
                with open(roads_path, "r", encoding="utf-8") as f:
                    cached_roads = json.load(f)
                with open(blds_path, "r", encoding="utf-8") as f:
                    cached_blds = json.load(f)
                if len(cached_roads.get("features", [])) > 0:
                    return cached_roads, cached_blds, True, None
            except Exception:
                pass

        # 2. Attempt Overpass query
        min_lat, min_lon, max_lat, max_lon = bbox
        overpass_query = f"""[out:json][timeout:30];
        (
          way["highway"~"{HIGHWAY_FILTER}"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """
        overpass_data = self._query_overpass(overpass_query, timeout_sec=25)

        if overpass_data and len(overpass_data.get("elements", [])) > 0:
            roads_geo, blds_geo = self._osm_elements_to_geojson(overpass_data["elements"], bbox)
            if len(roads_geo["features"]) > 0:
                with open(roads_path, "w", encoding="utf-8") as f:
                    json.dump(roads_geo, f, indent=2)
                with open(blds_path, "w", encoding="utf-8") as f:
                    json.dump(blds_geo, f, indent=2)
                return roads_geo, blds_geo, False, None

        # 3. Direct OSM 0.6 tile queries (split bbox into 4 manageable quadrants)
        mid_lat = 0.5 * (min_lat + max_lat)
        mid_lon = 0.5 * (min_lon + max_lon)
        quadrants = [
            [min_lat, min_lon, mid_lat, mid_lon],
            [min_lat, mid_lon, mid_lat, max_lon],
            [mid_lat, min_lon, max_lat, mid_lon],
            [mid_lat, mid_lon, max_lat, max_lon],
        ]

        all_elements: List[Dict[str, Any]] = []
        for q_box in quadrants:
            q_res = self._query_osm_direct_tile(q_box, timeout_sec=15)
            if q_res and "elements" in q_res:
                all_elements.extend(q_res["elements"])

        if len(all_elements) > 0:
            roads_geo, blds_geo = self._osm_elements_to_geojson(all_elements, bbox)
            if len(roads_geo["features"]) > 0:
                with open(roads_path, "w", encoding="utf-8") as f:
                    json.dump(roads_geo, f, indent=2)
                with open(blds_path, "w", encoding="utf-8") as f:
                    json.dump(blds_geo, f, indent=2)
                return roads_geo, blds_geo, False, None

        # 4. If neither cache nor live API is available: return empty with warning
        warning_msg = (
            f"REAL OPENSTREETMAP DATA UNAVAILABLE: Live Overpass and OSM API requests timed out or were blocked, "
            f"and no local cache exists at {roads_path.name}."
        )
        empty_fc = {"type": "FeatureCollection", "features": [], "metadata": {"warning": warning_msg}}
        return empty_fc, empty_fc, False, warning_msg


osm_service = OSMDataService()

