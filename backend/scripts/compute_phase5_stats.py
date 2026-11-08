import json
import networkx as nx
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.georef import GeoSpatialTransform
from app.services.damage_analyzer import analyze_building_exposure
from app.services.rescue_router import classify_road_hazards, build_road_graph, calculate_astar_evacuation_route

raster_path = "backend/data/satellite_cache/flood_mask_024d3597.tif"
buildings_path = "backend/data/osm_cache/chennai_buildings.geojson"
roads_path = "backend/data/osm_cache/chennai_roads.geojson"

georef = GeoSpatialTransform.from_raster_file(raster_path)
with open(buildings_path, "r", encoding="utf-8") as f:
    b_geojson = json.load(f)
with open(roads_path, "r", encoding="utf-8") as f:
    r_geojson = json.load(f)

print(f"Total buildings in file: {len(b_geojson['features'])}")
print(f"Total roads in file: {len(r_geojson['features'])}")

import rasterio

with rasterio.open(raster_path) as src:
    seg_mask = src.read(1)
    nodata_val = src.nodata if src.nodata is not None else 255.0

print(f"Loaded raster: shape={seg_mask.shape}, nodata={nodata_val}")

# Analyze buildings (sample 1000 for fast stats and demo)
t0 = time.time()
b_exposure = analyze_building_exposure(b_geojson["features"], seg_mask, georef, max_buildings=1000, nodata_val=nodata_val)
t_build = time.time() - t0
print(f"Building analysis took {t_build:.2f}s:")
print("Building breakdown:", b_exposure["summary"])

# Analyze roads
t0 = time.time()
r_hazards = classify_road_hazards(r_geojson["features"][:1000], seg_mask, georef, nodata_val=nodata_val)
t_roads = time.time() - t0
print(f"Road analysis took {t_roads:.2f}s:")
print("Road breakdown:", r_hazards["summary"])

# Build MultiDiGraph from all roads
t0 = time.time()
G_all = build_road_graph(r_geojson["features"], seg_mask, georef, avoid_hazards=True)
t_graph = time.time() - t0
print(f"Full MultiDiGraph build ({len(r_geojson['features'])} roads) took {t_graph:.2f}s:")
print(f"MultiDiGraph nodes: {G_all.number_of_nodes()}, edges: {G_all.number_of_edges()}")

# Check weakly connected components for MultiDiGraph
components = list(nx.weakly_connected_components(G_all))
print(f"Weakly connected components: {len(components)}, largest component nodes: {len(max(components, key=len))}")

# Pick two nodes from the largest component
start_pt = (80.259097, 13.015806)
dest_pt = (80.2664, 13.039248)
print(f"Routing between node {start_pt} and {dest_pt}...")

route = calculate_astar_evacuation_route(
    G_all,
    start_coords=start_pt,
    dest_coords=dest_pt,
    georef=georef,
    speed_kmh=25.0,
)
print("Route status:", route["status"])
print("Route details:", {
    "distance_m": route.get("distance_m"),
    "distance_km": route.get("distance_km"),
    "eta_minutes": route.get("eta_minutes"),
    "segments_count": route.get("segments_count"),
    "hazard_segments_count": route.get("hazard_segments_count", 0)
})


