import math
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np

from app.config import (
    AVERAGE_RESCUE_SPEED_KMH,
    CLASS_NEW_FLOOD,
    CLASS_PERMANENT_WATER,
)
from app.services.georef import GeoSpatialTransform
from app.services.osm_service import osm_service


def sample_road_hazard_metrics(
    coords_lonlat: List[List[float]],
    seg_mask: np.ndarray,
    georef: GeoSpatialTransform,
    nodata_val: float = 255.0,
    flood_val: int = CLASS_NEW_FLOOD,
) -> Dict[str, Any]:
    """
    Samples flood raster values along an actual road geometry at ~10m metric intervals.
    Returns valid sample count, flood sample count, intersection percentage, and Phase 5 hazard category:
    - POTENTIAL_HAZARD: >= 30% of valid samples intersect mapped flood mask.
    - CAUTION: Between 5% and 30%.
    - NO_DETECTED_INTERSECTION: < 5%.
    - UNKNOWN: Outside raster bounds or < 2 valid samples.
    """
    h, w = seg_mask.shape[:2]
    if len(coords_lonlat) < 2:
        return {
            "hazard_category": "UNKNOWN",
            "flood_intersection_pct": 0.0,
            "valid_samples_count": 0,
            "flood_samples_count": 0,
            "reason": "INSUFFICIENT_VERTICES",
        }

    dense_pixels: List[Tuple[int, int]] = []
    for i in range(len(coords_lonlat) - 1):
        p1_lon, p1_lat = coords_lonlat[i]
        p2_lon, p2_lat = coords_lonlat[i + 1]

        c1, r1 = georef.lonlat_to_pixel(p1_lon, p1_lat)
        c2, r2 = georef.lonlat_to_pixel(p2_lon, p2_lat)

        dist_px = math.hypot(c2 - c1, r2 - r1)
        num_steps = max(int(math.ceil(dist_px)), 2)

        for step in range(num_steps):
            t = step / float(num_steps)
            c = int(round(c1 + t * (c2 - c1)))
            r = int(round(r1 + t * (r2 - r1)))
            dense_pixels.append((c, r))

    valid_samples = 0
    flood_samples = 0

    for c, r in dense_pixels:
        if 0 <= c < w and 0 <= r < h:
            val = float(seg_mask[r, c])
            if val != nodata_val:
                valid_samples += 1
                if int(val) == flood_val:
                    flood_samples += 1

    if valid_samples < 2:
        return {
            "hazard_category": "UNKNOWN",
            "flood_intersection_pct": 0.0,
            "valid_samples_count": valid_samples,
            "flood_samples_count": flood_samples,
            "reason": "OUTSIDE_RASTER_OR_NODATA",
        }

    flood_pct = round((flood_samples / float(valid_samples)) * 100.0, 2)

    if flood_pct >= 30.0:
        cat = "POTENTIAL_HAZARD"
    elif flood_pct >= 5.0:
        cat = "CAUTION"
    else:
        cat = "NO_DETECTED_INTERSECTION"

    return {
        "hazard_category": cat,
        "flood_intersection_pct": flood_pct,
        "valid_samples_count": valid_samples,
        "flood_samples_count": flood_samples,
        "reason": "SUCCESS",
    }


def classify_road_hazards(
    road_features: List[Dict[str, Any]],
    seg_mask: np.ndarray,
    georef: GeoSpatialTransform,
    nodata_val: float = 255.0,
    flood_val: int = CLASS_NEW_FLOOD,
) -> Dict[str, Any]:
    """
    Evaluates real OSM road LineStrings against the flood raster and classifies each into
    POTENTIAL_HAZARD, CAUTION, NO_DETECTED_INTERSECTION, or UNKNOWN.
    """
    color_map = {
        "POTENTIAL_HAZARD": "#ef4444",
        "CAUTION": "#f59e0b",
        "NO_DETECTED_INTERSECTION": "#10b981",
        "UNKNOWN": "#94a3b8",
    }

    classified_features: List[Dict[str, Any]] = []
    counts = {
        "POTENTIAL_HAZARD": 0,
        "CAUTION": 0,
        "NO_DETECTED_INTERSECTION": 0,
        "UNKNOWN": 0,
    }

    total_length_m = 0.0
    hazard_length_m = 0.0

    for feat in road_features:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        props = feat.get("properties", {})
        way_id = props.get("osm_id", feat.get("id", "way_unknown"))
        road_name = props.get("name", "OSM Street")
        highway = props.get("highway", "residential")
        is_bridge = bool(props.get("is_bridge", False))
        is_oneway = bool(props.get("is_oneway", False))

        if len(coords) < 2:
            continue

        length_m = georef.compute_line_length_m(coords)
        total_length_m += length_m

        stats = sample_road_hazard_metrics(
            coords_lonlat=coords,
            seg_mask=seg_mask,
            georef=georef,
            nodata_val=nodata_val,
            flood_val=flood_val,
        )

        cat = stats["hazard_category"]
        counts[cat] += 1
        color = color_map.get(cat, "#94a3b8")

        if cat == "POTENTIAL_HAZARD":
            hazard_length_m += length_m

        classified_features.append(
            {
                "type": "Feature",
                "id": f"road/{way_id}",
                "properties": {
                    "osm_id": str(way_id),
                    "road_name": road_name,
                    "highway": highway,
                    "hazard_category": cat,
                    "flood_intersection_pct": stats["flood_intersection_pct"],
                    "valid_samples_count": stats["valid_samples_count"],
                    "flood_samples_count": stats["flood_samples_count"],
                    "length_m": round(length_m, 1),
                    "length_km": round(length_m / 1000.0, 3),
                    "color": color,
                    "is_bridge": is_bridge,
                    "is_oneway": is_oneway,
                    "scientific_disclaimer": "Mapped intersection only; does not establish real-time road passability or structural integrity.",
                },
                "geometry": geom,
            }
        )

    summary = {
        "total_roads_analyzed": len(classified_features),
        "potential_hazard_count": counts["POTENTIAL_HAZARD"],
        "potential_hazard_segments": counts["POTENTIAL_HAZARD"],
        "caution_count": counts["CAUTION"],
        "caution_segments": counts["CAUTION"],
        "no_detected_intersection_count": counts["NO_DETECTED_INTERSECTION"],
        "clear_segments": counts["NO_DETECTED_INTERSECTION"],
        "unknown_count": counts["UNKNOWN"],
        "total_network_length_km": round(total_length_m / 1000.0, 2),
        "hazard_impacted_length_km": round(hazard_length_m / 1000.0, 2),
    }

    return {
        "summary": summary,
        "features": classified_features,
        "geojson": {
            "type": "FeatureCollection",
            "features": classified_features,
        },
    }


def build_road_graph(
    road_features: List[Dict[str, Any]],
    seg_mask: np.ndarray,
    georef: GeoSpatialTransform,
    avoid_hazards: bool = True,
    caution_penalty: float = 3.0,
    hazard_penalty: float = 10.0,
    nodata_val: float = 255.0,
    flood_val: int = CLASS_NEW_FLOOD,
) -> nx.MultiDiGraph:
    """
    Builds a NetworkX MultiDiGraph from real OSM road geometries with metric edge lengths in local UTM.
    Supports parallel road edges between identical intersection nodes without data loss.
    Respects OSM directional connectivity and one-way restrictions:
    - oneway in ["yes", "true", "1"]: Directed edges only along forward vertex order (u -> v).
    - oneway in ["-1", "reverse"]: Directed edges reversed against vertex order (v -> u).
    - oneway in ["no", "false", "0"]: Bidirectional edges (u -> v and v -> u).
    - Reversible/conditional (e.g. reversible, alternating, oneway:conditional): Handled conservatively
      as forward-only, recording conditional restriction attribute.
    - Missing/unspecified oneway tag:
        - motorway, motorway_link, roundabout: default to forward-only.
        - all other highway types: default to bidirectional.
    
    Strict Hazard Avoidance:
    - When avoid_hazards=True, POTENTIAL_HAZARD edges are completely excluded from the graph.
    - When avoid_hazards=False, POTENTIAL_HAZARD edges are retained with hazard_penalty multiplier (10x).
    - CAUTION edges are retained with caution_penalty multiplier (3x).
    - UNKNOWN edges are retained with 2x penalty.
    """
    G = nx.MultiDiGraph()

    for feat in road_features:
        coords = feat.get("geometry", {}).get("coordinates", [])
        props = feat.get("properties", {})
        way_id = props.get("osm_id", "")
        road_name = props.get("name", "OSM Street")
        highway = str(props.get("highway", "residential")).lower().strip()
        junction = str(props.get("junction", "")).lower().strip()
        oneway_val = str(props.get("oneway", "")).lower().strip()

        if len(coords) < 2:
            continue

        # Check conditional or reversible restrictions conservatively
        is_conditional = oneway_val in ["reversible", "alternating"] or "oneway:conditional" in props

        # Determine directional policy
        if oneway_val in ["yes", "true", "1"]:
            forward, backward = True, False
        elif oneway_val in ["-1", "reverse"]:
            forward, backward = False, True
        elif oneway_val in ["no", "false", "0"]:
            forward, backward = True, True
        elif is_conditional:
            # Conservative: do not assume unrestricted bidirectional flow on reversible/conditional roads
            forward, backward = True, False
        else:
            # OSM default convention for missing/unspecified oneway
            if highway in ["motorway", "motorway_link"] or junction == "roundabout":
                forward, backward = True, False
            else:
                forward, backward = True, True

        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            seg_coords = [p1, p2]
            length_m = max(georef.compute_line_length_m(seg_coords), 0.5)

            stats = sample_road_hazard_metrics(
                coords_lonlat=seg_coords,
                seg_mask=seg_mask,
                georef=georef,
                nodata_val=nodata_val,
                flood_val=flood_val,
            )
            cat = stats["hazard_category"]

            # Hazard Cost Multiplier Assignment
            if cat == "POTENTIAL_HAZARD":
                if avoid_hazards:
                    # Strict hazard avoidance: completely exclude potential hazard edges
                    continue
                else:
                    multiplier = hazard_penalty
            elif cat == "CAUTION":
                multiplier = caution_penalty
            elif cat == "UNKNOWN":
                multiplier = 2.0
            else:
                multiplier = 1.0

            cost = length_m * multiplier

            u = (round(p1[0], 6), round(p1[1], 6))
            v = (round(p2[0], 6), round(p2[1], 6))

            edge_attrs = {
                "weight": cost,
                "length_m": length_m,
                "hazard_category": cat,
                "flood_pct": stats["flood_intersection_pct"],
                "osm_id": str(way_id),
                "road_name": road_name,
                "highway": highway,
                "oneway": props.get("oneway"),
                "is_conditional": is_conditional,
            }

            if forward:
                key_fwd = f"{way_id}_{i}_fwd"
                G.add_edge(
                    u,
                    v,
                    key=key_fwd,
                    coordinates=seg_coords,
                    **edge_attrs,
                )

            if backward:
                key_bwd = f"{way_id}_{i}_bwd"
                G.add_edge(
                    v,
                    u,
                    key=key_bwd,
                    coordinates=[p2, p1],
                    **edge_attrs,
                )

    return G


def calculate_astar_evacuation_route(
    G: Union[nx.Graph, nx.DiGraph, nx.MultiDiGraph],
    start_coords: Tuple[float, float],
    dest_coords: Tuple[float, float],
    georef: GeoSpatialTransform,
    speed_kmh: float = 25.0,
    max_snap_dist_m: float = 3500.0,
) -> Dict[str, Any]:
    """
    Calculates candidate evacuation route using NetworkX A* with Euclidean metric heuristic.
    Supports Graph, DiGraph, and MultiDiGraph networks while preserving lowest-cost parallel edges.
    Snaps start and destination coordinates to nearest nodes in the navigable road network.
    Calculates transit ETA using the exact formula: ETA_minutes = (distance_m / 1000) / speed_kmh * 60.
    """
    if G.number_of_nodes() == 0:
        return {
            "status": "NO_SAFE_ROUTE_AVAILABLE",
            "message": "Road graph is empty or all navigable roads are submerged.",
            "distance_m": 0.0,
            "distance_km": 0.0,
            "eta_minutes": 0.0,
            "segments_count": 0,
            "route_geojson": None,
        }

    # Snap start and destination to nearest nodes
    def find_nearest_node(lon: float, lat: float) -> Tuple[Optional[Tuple[float, float]], float]:
        best_node = None
        min_dist = float("inf")
        for node in G.nodes():
            d = georef.compute_line_length_m([[lon, lat], [node[0], node[1]]])
            if d < min_dist:
                min_dist = d
                best_node = node
        return best_node, min_dist

    start_node, start_snap_dist = find_nearest_node(start_coords[0], start_coords[1])
    dest_node, dest_snap_dist = find_nearest_node(dest_coords[0], dest_coords[1])

    if start_node is None or start_snap_dist > max_snap_dist_m:
        return {
            "status": "START_POINT_TOO_FAR_FROM_ROAD_NETWORK",
            "message": f"Start location is {start_snap_dist:.0f}m from nearest navigable road (limit {max_snap_dist_m}m).",
            "distance_m": 0.0,
            "distance_km": 0.0,
            "eta_minutes": 0.0,
            "segments_count": 0,
            "route_geojson": None,
        }

    if dest_node is None or dest_snap_dist > max_snap_dist_m:
        return {
            "status": "DEST_POINT_TOO_FAR_FROM_ROAD_NETWORK",
            "message": f"Destination location is {dest_snap_dist:.0f}m from nearest navigable road (limit {max_snap_dist_m}m).",
            "distance_m": 0.0,
            "distance_km": 0.0,
            "eta_minutes": 0.0,
            "segments_count": 0,
            "route_geojson": None,
        }

    def astar_heuristic(u: Tuple[float, float], v: Tuple[float, float]) -> float:
        # Admissible Euclidean straight-line distance in meters
        return georef.compute_line_length_m([[u[0], u[1]], [v[0], v[1]]])

    try:
        path_nodes = nx.astar_path(
            G, start_node, dest_node, heuristic=astar_heuristic, weight="weight"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {
            "status": "NO_SAFE_ROUTE_AVAILABLE",
            "message": "No continuous navigable route connects start and destination avoiding mapped flood hazards.",
            "distance_m": 0.0,
            "distance_km": 0.0,
            "eta_minutes": 0.0,
            "segments_count": 0,
            "route_geojson": None,
        }

    # Reconstruct path coordinates and metric length
    full_coords: List[List[float]] = []
    total_length_m = 0.0
    caution_segments = 0

    for i in range(len(path_nodes) - 1):
        u_n, v_n = path_nodes[i], path_nodes[i + 1]
        raw_edge_data = G.get_edge_data(u_n, v_n)

        # In MultiDiGraph, pick the optimal edge among parallel edges connecting u_n to v_n
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) and isinstance(raw_edge_data, dict):
            # Check if dict of edge keys or single edge
            if any(isinstance(v, dict) for v in raw_edge_data.values()):
                edge_data = min(raw_edge_data.values(), key=lambda e: e.get("weight", float("inf")))
            else:
                edge_data = raw_edge_data
        else:
            edge_data = raw_edge_data or {}

        length_m = edge_data.get("length_m", 0.0)
        total_length_m += length_m

        if edge_data.get("hazard_category") == "CAUTION":
            caution_segments += 1

        edge_coords = edge_data.get("coordinates", [[u_n[0], u_n[1]], [v_n[0], v_n[1]]])
        if full_coords:
            full_coords.extend(edge_coords[1:])
        else:
            full_coords.extend(edge_coords)
            full_coords.extend(edge_coords)

    total_km = round(total_length_m / 1000.0, 3)
    # Exact ETA: ETA_minutes = (distance_m / 1000) / speed_kmh * 60
    eta_mins = round(((total_length_m / 1000.0) / speed_kmh) * 60.0, 2)

    route_geojson = {
        "type": "Feature",
        "id": "A_STAR_EVACUATION_ROUTE",
        "properties": {
            "feature_type": "evacuation_route",
            "status": "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS",
            "distance_m": round(total_length_m, 1),
            "distance_km": total_km,
            "eta_minutes": eta_mins,
            "speed_kmh": speed_kmh,
            "segments_count": len(path_nodes) - 1,
            "caution_segments_count": caution_segments,
            "start_node": start_node,
            "dest_node": dest_node,
            "start_snap_dist_m": round(start_snap_dist, 1),
            "dest_snap_dist_m": round(dest_snap_dist, 1),
            "color": "#10b981" if caution_segments == 0 else "#f59e0b",
            "scientific_disclaimer": "FloodSense-AI identifies potential infrastructure exposure through spatial intersection with a mapped flood-risk layer. It does not independently confirm structural damage, actual road closures, or operationally safe evacuation routes. All candidate routes require local verification.",
        },
        "geometry": {
            "type": "LineString",
            "coordinates": full_coords,
        },
    }

    return {
        "status": "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS",
        "message": "Candidate evacuation route successfully calculated avoiding mapped flood hazards based on available OSM geometry and satellite flood-risk layer.",
        "distance_m": round(total_length_m, 1),
        "distance_km": total_km,
        "eta_minutes": eta_mins,
        "speed_kmh": speed_kmh,
        "segments_count": len(path_nodes) - 1,
        "caution_segments_count": caution_segments,
        "start_snapped": {"coords": start_node, "snap_distance_m": round(start_snap_dist, 1)},
        "dest_snapped": {"coords": dest_node, "snap_distance_m": round(dest_snap_dist, 1)},
        "route_geojson": route_geojson,
    }


def _sample_line_flood_status(
    coords_lonlat: List[List[float]],
    seg_mask: np.ndarray,
    georef: GeoSpatialTransform,
    is_bridge: bool = False,
) -> Tuple[bool, float, str]:
    """Backwards-compatible legacy helper."""
    stats = sample_road_hazard_metrics(coords_lonlat, seg_mask, georef)
    is_blocked = stats["hazard_category"] == "POTENTIAL_HAZARD"
    pct = stats["flood_intersection_pct"]
    reason = "BLOCKED_ACTIVE_FLOOD" if is_blocked else "PASSABLE"
    return is_blocked, pct, reason


def _snap_to_nearest_graph_node(
    lon: float, lat: float, G: nx.Graph, max_dist_m: float = 3500.0
) -> Optional[Tuple[float, float]]:
    """Snaps a point to the nearest node in the NetworkX road graph."""
    best_node = None
    min_dist_sq = float("inf")

    for node in G.nodes():
        node_lon, node_lat = node
        d_sq = (node_lon - lon) ** 2 + (node_lat - lat) ** 2
        if d_sq < min_dist_sq:
            min_dist_sq = d_sq
            best_node = node

    return best_node


def compute_osm_rescue_routes(
    seg_mask: np.ndarray,
    bbox: List[float],
    preset_id: str,
    custom_emergency_base: Optional[Tuple[float, float]] = None,
    custom_destinations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Constructs a true OpenStreetMap-based NetworkX road graph, samples actual road geometries
    against the georeferenced flood classification raster, identifies blocked roads,
    and runs A* shortest-path evacuation routing.
    """
    georef = GeoSpatialTransform(bbox, width=seg_mask.shape[1], height=seg_mask.shape[0])

    # 1. Load Real OpenStreetMap roads (cached or live)
    roads_geo, _, from_cache, warning = osm_service.fetch_and_cache_osm_data(preset_id, bbox)
    road_features = roads_geo.get("features", [])

    if not road_features:
        return {
            "summary": {
                "active_safe_routes": 0,
                "shortest_route_km": 0.0,
                "fastest_eta_mins": 0.0,
                "average_eta_mins": 0.0,
                "blocked_road_segments": 0,
                "total_road_segments": 0,
                "network_integrity_pct": 0.0,
            },
            "routes": [],
            "waypoints": [],
            "geojson": {
                "safe_routes": {"type": "FeatureCollection", "features": []},
                "severed_roads": {"type": "FeatureCollection", "features": []},
                "all_roads": {"type": "FeatureCollection", "features": []},
            },
            "data_provenance": {
                "road_data_source": "UNAVAILABLE",
                "osm_cache_used": False,
                "data_warnings": warning or "No OSM road features available.",
            },
        }

    # 2. Build NetworkX Directed/Undirected Graph from real OSM way geometries
    G_safe = nx.Graph()
    G_all = nx.Graph()

    blocked_road_features: List[Dict[str, Any]] = []
    total_segments = 0
    blocked_count = 0

    for feat in road_features:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        props = feat.get("properties", {})
        is_bridge = bool(props.get("is_bridge", False))
        is_oneway = bool(props.get("is_oneway", False))
        road_name = props.get("name", "OSM Street")
        highway_cls = props.get("highway", "residential")
        way_id = props.get("osm_id", "")

        if len(coords) < 2:
            continue

        # Inspect whole line string and split into node-to-node sub-segments
        for seg_idx in range(len(coords) - 1):
            p1 = coords[seg_idx]
            p2 = coords[seg_idx + 1]
            seg_coords = [p1, p2]

            total_segments += 1
            length_m = max(georef.compute_line_length_m(seg_coords), 1.0)
            length_km = length_m / 1000.0

            is_blocked, submergence_pct, block_reason = _sample_line_flood_status(
                seg_coords, seg_mask, georef, is_bridge=is_bridge
            )

            u = (round(p1[0], 6), round(p1[1], 6))
            v = (round(p2[0], 6), round(p2[1], 6))

            G_all.add_edge(
                u, v,
                weight=length_km,
                distance_km=length_km,
                road_name=road_name,
                highway=highway_cls,
                coordinates=seg_coords,
                is_blocked=is_blocked,
            )

            if is_blocked:
                blocked_count += 1
                blocked_road_features.append(
                    {
                        "type": "Feature",
                        "id": f"blocked_{way_id}_{seg_idx}",
                        "properties": {
                            "feature_type": "severed_road",
                            "osm_id": way_id,
                            "road_name": road_name,
                            "highway": highway_cls,
                            "status": "BLOCKED",
                            "reason": block_reason,
                            "submergence_pct": submergence_pct,
                            "length_km": round(length_km, 3),
                            "is_bridge": is_bridge,
                            "color": "#ef4444",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": seg_coords,
                        },
                    }
                )
            else:
                # Add unblocked road edge to A* navigable graph
                G_safe.add_edge(
                    u, v,
                    weight=length_km,
                    distance_km=length_km,
                    road_name=road_name,
                    highway=highway_cls,
                    coordinates=seg_coords,
                )

    # 3. Select Safe Emergency Response Base
    all_safe_nodes = list(G_safe.nodes())
    if not all_safe_nodes:
        all_safe_nodes = list(G_all.nodes())

    largest_cc = (
        max(nx.connected_components(G_safe), key=len)
        if G_safe.number_of_nodes() > 0
        else set(all_safe_nodes)
    )

    if custom_emergency_base:
        base_node = _snap_to_nearest_graph_node(custom_emergency_base[0], custom_emergency_base[1], G_safe)
    else:
        # Select connected road node on dry high ground (northwestern high margin)
        target_lon = bbox[1] + 0.18 * (bbox[3] - bbox[1])
        target_lat = bbox[2] - 0.18 * (bbox[2] - bbox[0])
        candidate_nodes = list(largest_cc) or all_safe_nodes
        base_node = min(
            candidate_nodes,
            key=lambda n: (n[0] - target_lon) ** 2 + (n[1] - target_lat) ** 2,
        )

    # 4. Select Evacuation Destinations (Affected Building Clusters or Strategic Hubs)
    destination_anchors = [
        {"name": "Sector Alpha — Riverbank Relief Staging Zone", "u": 0.72, "v": 0.76},
        {"name": "Sector Bravo — Urban Transit Corridor Shelter", "u": 0.28, "v": 0.74},
        {"name": "Sector Charlie — Southern Medical Center", "u": 0.76, "v": 0.26},
    ]

    safe_route_features: List[Dict[str, Any]] = []
    route_summaries: List[Dict[str, Any]] = []
    waypoints: List[Dict[str, Any]] = [
        {
            "id": "CMD_BASE",
            "role": "EMERGENCY_BASE",
            "name": "NDRF / District Emergency Command Base",
            "lon": base_node[0],
            "lat": base_node[1],
        }
    ]

    route_colors = ["#10b981", "#34d399", "#06b6d4"]

    def astar_heuristic(u_node: Tuple[float, float], v_node: Tuple[float, float]) -> float:
        # Admissible Euclidean heuristic in kilometers
        e1, n1 = georef.lonlat_to_utm(u_node[0], u_node[1])
        e2, n2 = georef.lonlat_to_utm(v_node[0], v_node[1])
        return math.hypot(e2 - e1, n2 - n1) / 1000.0

    for idx, dest_spec in enumerate(destination_anchors):
        target_lon = bbox[1] + dest_spec["u"] * (bbox[3] - bbox[1])
        target_lat = bbox[0] + dest_spec["v"] * (bbox[2] - bbox[0])

        target_node = min(
            list(largest_cc) or all_safe_nodes,
            key=lambda n: (n[0] - target_lon) ** 2 + (n[1] - target_lat) ** 2,
        )

        waypoints.append(
            {
                "id": f"EVAC_{idx + 1}",
                "role": "EVACUATION_ZONE",
                "name": dest_spec["name"],
                "lon": target_node[0],
                "lat": target_node[1],
            }
        )

        route_id = f"OSM-ROUTE-0{idx + 1}"

        try:
            if base_node not in G_safe or target_node not in G_safe:
                raise nx.NetworkXNoPath("Base or target node is submerged or isolated from safe road network")

            path_nodes = nx.astar_path(
                G_safe, base_node, target_node, heuristic=astar_heuristic, weight="weight"
            )
            # Reconstruct exact road geometries along path
            full_path_coords: List[List[float]] = []
            total_km = 0.0

            for i in range(len(path_nodes) - 1):
                u_n, v_n = path_nodes[i], path_nodes[i + 1]
                edge_data = G_safe.get_edge_data(u_n, v_n)
                total_km += edge_data.get("distance_km", 0.0)
                edge_coords = edge_data.get("coordinates", [[u_n[0], u_n[1]], [v_n[0], v_n[1]]])
                if full_path_coords:
                    full_path_coords.extend(edge_coords[1:])
                else:
                    full_path_coords.extend(edge_coords)

            if len(path_nodes) < 2 or not full_path_coords:
                route_summary = {
                    "route_id": route_id,
                    "destination": dest_spec["name"],
                    "distance_km": 0.0,
                    "eta_minutes": 0.0,
                    "segments_count": 0,
                    "status": "NO_SAFE_ROUTE_AVAILABLE",
                    "color": "#94a3b8",
                }
                route_summaries.append(route_summary)
                continue

            total_km = round(max(total_km, 0.25), 3)
            eta_mins = round((total_km / AVERAGE_RESCUE_SPEED_KMH) * 60.0, 2)

            route_summary = {
                "route_id": route_id,
                "destination": dest_spec["name"],
                "distance_km": total_km,
                "eta_minutes": eta_mins,
                "segments_count": len(path_nodes) - 1,
                "status": "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS",
                "color": route_colors[idx % len(route_colors)],
            }
            route_summaries.append(route_summary)

            safe_route_features.append(
                {
                    "type": "Feature",
                    "id": route_id,
                    "properties": {
                        "feature_type": "rescue_route",
                        **route_summary,
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": full_path_coords,
                    },
                }
            )

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Explicit academic requirement: Never fabricate synthetic lines when road is blocked
            route_summary = {
                "route_id": route_id,
                "destination": dest_spec["name"],
                "distance_km": 0.0,
                "eta_minutes": 0.0,
                "segments_count": 0,
                "status": "NO_SAFE_ROUTE_AVAILABLE",
                "color": "#94a3b8",
            }
            route_summaries.append(route_summary)

    shortest_dist = min([r["distance_km"] for r in route_summaries if r["distance_km"] > 0], default=0.0)
    fastest_eta = min([r["eta_minutes"] for r in route_summaries if r["eta_minutes"] > 0], default=0.0)
    avg_eta = round(float(np.mean([r["eta_minutes"] for r in route_summaries if r["eta_minutes"] > 0])), 1) if any(r["eta_minutes"] > 0 for r in route_summaries) else 0.0

    network_integrity = round(((total_segments - blocked_count) / max(total_segments, 1)) * 100.0, 1)

    # Also run full road hazard classification for comprehensive GIS inspection
    hazard_eval = classify_road_hazards(road_features, seg_mask, georef)
    all_roads_geojson = hazard_eval["geojson"]

    return {
        "summary": {
            "active_safe_routes": len(safe_route_features),
            "shortest_route_km": shortest_dist,
            "fastest_eta_mins": fastest_eta,
            "average_eta_mins": avg_eta,
            "blocked_road_segments": blocked_count,
            "total_road_segments": total_segments,
            "network_integrity_pct": network_integrity,
            "hazard_breakdown": hazard_eval["summary"],
        },
        "routes": route_summaries,
        "waypoints": waypoints,
        "geojson": {
            "safe_routes": {
                "type": "FeatureCollection",
                "features": safe_route_features,
            },
            "severed_roads": {
                "type": "FeatureCollection",
                "features": blocked_road_features,
            },
            "all_roads": all_roads_geojson,
        },
        "data_provenance": {
            "road_data_source": "OpenStreetMap contributors (ODbL / EPSG:4326)",
            "osm_cache_used": from_cache,
            "total_osm_ways": len(road_features),
            "data_warnings": warning,
        },
    }
