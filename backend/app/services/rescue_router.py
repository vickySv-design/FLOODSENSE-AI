import math
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np

from app.config import (
    AVERAGE_RESCUE_SPEED_KMH,
    CLASS_NEW_FLOOD,
    CLASS_PERMANENT_WATER,
)
from app.services.georef import GeoSpatialTransform
from app.services.osm_service import osm_service


def _sample_line_flood_status(
    coords_lonlat: List[List[float]],
    seg_mask: np.ndarray,
    georef: GeoSpatialTransform,
    is_bridge: bool = False,
) -> Tuple[bool, float, str]:
    """
    Samples flood and permanent water pixels along an actual OSM road segment geometry.
    - Transform geographic coordinates to raster pixel row & column via affine transform.
    - Sample at intervals approximately equal to 1 raster pixel (10m).
    - Rules:
      1. Road intersecting Class 2 (New Flood) is BLOCKED.
      2. Road crossing Class 1 (Permanent Water) without an OSM bridge tag is BLOCKED.
      3. Road with an OSM bridge crossing Class 1 is PASSABLE unless Class 2 flood overtops it.
      4. Bridge intersecting Class 2 is BLOCKED.
    """
    h, w = seg_mask.shape[:2]
    if len(coords_lonlat) < 2:
        return False, 0.0, "PASSABLE"

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
            if 0 <= c < w and 0 <= r < h:
                dense_pixels.append((c, r))

    if not dense_pixels:
        # Roads outside the raster domain cannot be confirmed safe
        return True, 100.0, "BLOCKED_OUT_OF_BOUNDS"

    class2_hits = 0
    class1_hits = 0

    for c, r in dense_pixels:
        val = int(seg_mask[r, c])
        if val == CLASS_NEW_FLOOD:
            class2_hits += 1
        elif val == CLASS_PERMANENT_WATER:
            class1_hits += 1

    total_samples = float(len(dense_pixels))
    flood_ratio = class2_hits / total_samples
    water_ratio = class1_hits / total_samples

    # Flood Blocking Rules
    if class2_hits > 0:
        return True, round(flood_ratio * 100.0, 1), "BLOCKED_ACTIVE_FLOOD"

    if class1_hits > 0 and not is_bridge:
        return True, round(water_ratio * 100.0, 1), "BLOCKED_SUBMERGED_WATER_CROSSING"

    return False, 0.0, "PASSABLE"


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

            total_km = round(max(total_km, 0.25), 2)
            eta_mins = round((total_km / AVERAGE_RESCUE_SPEED_KMH) * 60.0 + 2.0, 1)

            route_summary = {
                "route_id": route_id,
                "destination": dest_spec["name"],
                "distance_km": total_km,
                "eta_minutes": eta_mins,
                "segments_count": len(path_nodes) - 1,
                "status": "PASSABLE_VERIFIED_OSM",
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

    return {
        "summary": {
            "active_safe_routes": len(safe_route_features),
            "shortest_route_km": shortest_dist,
            "fastest_eta_mins": fastest_eta,
            "average_eta_mins": avg_eta,
            "blocked_road_segments": blocked_count,
            "total_road_segments": total_segments,
            "network_integrity_pct": network_integrity,
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
            "all_roads": roads_geo,
        },
        "data_provenance": {
            "road_data_source": "OpenStreetMap contributors (ODbL / EPSG:4326)",
            "osm_cache_used": from_cache,
            "total_osm_ways": len(road_features),
            "data_warnings": warning,
        },
    }
