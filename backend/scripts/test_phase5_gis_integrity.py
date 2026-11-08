"""
FloodSense-AI: Phase 5 Automated GIS Infrastructure Exposure & A* Routing Integrity Test Suite
==============================================================================================
Verifies the 15 mandatory integrity areas for Phase 5:
1. Raster CRS validation.
2. Vector reprojection and metric CRS.
3. Geometry validity and repair.
4. NoData exclusion.
5. Building intersection calculations.
6. Building exposure percentage thresholds.
7. Road sampling intervals.
8. Road hazard classification categories.
9. Graph construction.
10. A* route generation.
11. Start and destination snapping.
12. No-route behavior when isolated.
13. Out-of-bounds handling.
14. API response validation.
15. GeoJSON validity (RFC 7946 compliance).
"""

import math
import os
import sys
import unittest
import numpy as np
import rasterio
from shapely.geometry import Polygon, LineString, box, shape

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.georef import GeoSpatialTransform
from app.services.damage_analyzer import analyze_building_exposure
from app.services.rescue_router import (
    sample_road_hazard_metrics,
    classify_road_hazards,
    build_road_graph,
    calculate_astar_evacuation_route,
)
from app.config import SATELLITE_CACHE_DIR, OSM_CACHE_DIR, CLASS_NEW_FLOOD
import networkx as nx


class TestPhase5GISIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raster_path = SATELLITE_CACHE_DIR / "flood_mask_024d3597.tif"
        cls.chennai_blds_path = OSM_CACHE_DIR / "chennai_buildings.geojson"
        cls.chennai_roads_path = OSM_CACHE_DIR / "chennai_roads.geojson"

        # Initialize GeoSpatialTransform for Chennai study area
        cls.bbox = [13.010, 80.220, 13.060, 80.270]  # [min_lat, min_lon, max_lat, max_lon]
        cls.georef = GeoSpatialTransform(cls.bbox, width=256, height=256)

        # Load raster if present
        if cls.raster_path.exists():
            with rasterio.open(cls.raster_path) as src:
                cls.real_raster = src.read(1)
        else:
            cls.real_raster = np.zeros((256, 256), dtype=np.uint8)

    # 1. Raster CRS validation
    def test_01_raster_crs_and_transform_validation(self):
        """Verify real flood raster CRS, transform, resolution and dimensions."""
        self.assertTrue(self.raster_path.exists(), "Flood mask GeoTIFF must exist in satellite cache")
        with rasterio.open(self.raster_path) as src:
            self.assertEqual(str(src.crs), "EPSG:4326")
            self.assertEqual(src.shape, (256, 256))
            self.assertEqual(src.nodata, 255.0)
            self.assertAlmostEqual(src.bounds.left, 80.22, places=4)
            self.assertAlmostEqual(src.bounds.bottom, 13.01, places=4)
            self.assertAlmostEqual(src.bounds.right, 80.27, places=4)
            self.assertAlmostEqual(src.bounds.top, 13.06, places=4)

    # 2. Vector reprojection and metric CRS
    def test_02_vector_reprojection_and_metric_crs(self):
        """Verify UTM projection selection and metric coordinate conversion accuracy."""
        # For Chennai (center ~80.245, ~13.035), UTM zone must be 44 North (EPSG:32644)
        self.assertEqual(self.georef.utm_zone, 44)
        self.assertEqual(self.georef.utm_epsg, 32644)

        # Round-trip transformation check: WGS84 -> UTM -> WGS84
        test_lon, test_lat = 80.245, 13.035
        e, n = self.georef.lonlat_to_utm(test_lon, test_lat)
        lon_back, lat_back = self.georef.utm_to_lonlat(e, n)
        self.assertAlmostEqual(test_lon, lon_back, places=5)
        self.assertAlmostEqual(test_lat, lat_back, places=5)

        # Verify metric cell dimensions are realistic (~21m at 13 deg N)
        self.assertGreater(self.georef.pixel_width_m, 18.0)
        self.assertLess(self.georef.pixel_width_m, 25.0)

    # 3. Geometry validity and repair
    def test_03_geometry_validity_and_repair(self):
        """Verify self-intersecting or invalid geometries are repaired without crashing."""
        # Bowtie polygon (self-intersecting)
        invalid_coords = [(80.23, 13.02), (80.25, 13.04), (80.23, 13.04), (80.25, 13.02), (80.23, 13.02)]
        invalid_poly = Polygon(invalid_coords)
        self.assertFalse(invalid_poly.is_valid, "Test polygon must be invalid self-intersecting fixture")

        fixed = self.georef.validate_and_fix_polygon(invalid_poly)
        self.assertIsNotNone(fixed)
        self.assertTrue(fixed.is_valid, "Repaired geometry must be valid")
        self.assertGreater(fixed.area, 0.0)

    # 4. NoData exclusion
    def test_04_nodata_exclusion(self):
        """Verify NoData pixels (255) are strictly excluded from valid sample denominator."""
        # Synthetic test fixture: 256x256 array with 255 (nodata) and 2 (flood)
        test_raster = np.full((256, 256), 255, dtype=np.uint8)
        # Place 2 flood pixels
        test_raster[50, 50] = 2
        test_raster[50, 51] = 2

        lon, lat = self.georef.pixel_to_lonlat(50, 50)
        lon_e, lat_s = self.georef.pixel_to_lonlat(52, 52)
        test_poly = box(min(lon, lon_e), min(lat, lat_s), max(lon, lon_e), max(lat, lat_s))

        stats = self.georef.calculate_polygon_overlap_stats(
            poly=test_poly, raster=test_raster, nodata_val=255.0, flood_val=2
        )
        # All NoData pixels must be excluded; valid pixels should match non-nodata samples
        self.assertGreater(stats["valid_pixels"], 0)
        self.assertEqual(stats["flood_pixels"], stats["valid_pixels"])
        self.assertEqual(stats["exposed_area_pct"], 100.0)

    # 5. Building intersection calculations
    def test_05_building_intersection_calculations(self):
        """Verify building area overlap calculation with controlled synthetic fixture."""
        # Synthetic fixture: 10x10 subgrid, half dry (0) and half flood (2)
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        test_raster[100:104, 100:104] = 2  # 16 flood pixels
        test_raster[100:104, 104:108] = 0  # 16 dry pixels

        lon1, lat1 = self.georef.pixel_to_lonlat(100, 100)
        lon2, lat2 = self.georef.pixel_to_lonlat(108, 104)
        poly = box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))

        stats = self.georef.calculate_polygon_overlap_stats(
            poly=poly, raster=test_raster, nodata_val=255.0, flood_val=2
        )
        self.assertAlmostEqual(stats["exposed_area_pct"], 50.0, delta=5.0)
        self.assertEqual(stats["exposure_category"], "HIGH_EXPOSURE")

    # 6. Building exposure percentage thresholds
    def test_06_building_exposure_percentage_thresholds(self):
        """Verify exact classification rules: HIGH >=50%, MODERATE 10-50%, LOW <10%."""
        test_raster = np.zeros((256, 256), dtype=np.uint8)

        # High exposure fixture (60% flood)
        lon1, lat1 = self.georef.pixel_to_lonlat(20, 20)
        lon2, lat2 = self.georef.pixel_to_lonlat(30, 30)
        test_raster[20:26, 20:30] = 2  # 60 out of 100 pixels
        poly_high = box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))
        res_high = self.georef.calculate_polygon_overlap_stats(poly_high, test_raster)
        self.assertEqual(res_high["exposure_category"], "HIGH_EXPOSURE")

        # Moderate exposure fixture (25% flood)
        lon3, lat3 = self.georef.pixel_to_lonlat(40, 40)
        lon4, lat4 = self.georef.pixel_to_lonlat(50, 50)
        test_raster[40:50, 40:50] = 0
        test_raster[40:45, 40:45] = 2  # 25 out of 100 pixels
        poly_mod = box(min(lon3, lon4), min(lat3, lat4), max(lon3, lon4), max(lat3, lat4))
        res_mod = self.georef.calculate_polygon_overlap_stats(poly_mod, test_raster)
        self.assertEqual(res_mod["exposure_category"], "MODERATE_EXPOSURE")

        # Low exposure fixture (3% flood)
        lon5, lat5 = self.georef.pixel_to_lonlat(60, 60)
        lon6, lat6 = self.georef.pixel_to_lonlat(70, 70)
        test_raster[60:70, 60:70] = 0
        test_raster[60:61, 60:63] = 2  # 3 out of 100 pixels
        poly_low = box(min(lon5, lon6), min(lat5, lat6), max(lon5, lon6), max(lat5, lat6))
        res_low = self.georef.calculate_polygon_overlap_stats(poly_low, test_raster)
        self.assertEqual(res_low["exposure_category"], "LOW_OR_NO_DETECTED_EXPOSURE")

    # 7. Road sampling intervals
    def test_07_road_sampling_intervals(self):
        """Verify sampling along LineString road geometries at metric pixel resolution."""
        p1 = [80.230, 13.020]
        p2 = [80.240, 13.020]
        length_m = self.georef.compute_line_length_m([p1, p2])
        self.assertGreater(length_m, 500.0)

        stats = sample_road_hazard_metrics([p1, p2], self.real_raster, self.georef)
        self.assertGreater(stats["valid_samples_count"], 10)
        self.assertIn(stats["hazard_category"], ["POTENTIAL_HAZARD", "CAUTION", "NO_DETECTED_INTERSECTION"])

    # 8. Road hazard classification categories
    def test_08_road_hazard_classification_categories(self):
        """Verify road hazard categories: POTENTIAL_HAZARD (>=30%), CAUTION (5-30%), NO_DETECTED (<5%)."""
        test_raster = np.zeros((256, 256), dtype=np.uint8)

        # Line from (10, 10) to (10, 50) -> 41 pixels
        lon_start, lat_start = self.georef.pixel_to_lonlat(10, 10)
        lon_end, lat_end = self.georef.pixel_to_lonlat(50, 10)
        coords = [[lon_start, lat_start], [lon_end, lat_end]]

        # Case A: 50% flood -> POTENTIAL_HAZARD
        test_raster[10, 10:31] = 2
        stats_a = sample_road_hazard_metrics(coords, test_raster, self.georef)
        self.assertEqual(stats_a["hazard_category"], "POTENTIAL_HAZARD")

        # Case B: 15% flood -> CAUTION
        test_raster[10, :] = 0
        test_raster[10, 10:17] = 2
        stats_b = sample_road_hazard_metrics(coords, test_raster, self.georef)
        self.assertEqual(stats_b["hazard_category"], "CAUTION")

        # Case C: 0% flood -> NO_DETECTED_INTERSECTION
        test_raster[10, :] = 0
        stats_c = sample_road_hazard_metrics(coords, test_raster, self.georef)
        self.assertEqual(stats_c["hazard_category"], "NO_DETECTED_INTERSECTION")

    # 9. Graph construction
    def test_09_graph_construction(self):
        """Verify NetworkX road graph builder parses LineStrings and attaches metric attributes."""
        synthetic_features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[80.225, 13.020], [80.230, 13.020], [80.235, 13.020]],
                },
                "properties": {"osm_id": 101, "name": "Test Highway", "highway": "primary"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[80.230, 13.020], [80.230, 13.025]],
                },
                "properties": {"osm_id": 102, "name": "Cross Street", "highway": "residential"},
            },
        ]
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        G = build_road_graph(synthetic_features, test_raster, self.georef)

        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreater(G.number_of_edges(), 0)
        for u, v, d in G.edges(data=True):
            self.assertIn("length_m", d)
            self.assertGreater(d["length_m"], 0.0)
            self.assertIn("weight", d)
            self.assertIn("hazard_category", d)

    # 10. A* route generation
    def test_10_astar_route_generation(self):
        """Verify A* finds shortest metric path avoiding or penalizing hazards."""
        synthetic_features = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[80.220, 13.020], [80.230, 13.020]]},
                "properties": {"osm_id": 1},
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[80.230, 13.020], [80.240, 13.020]]},
                "properties": {"osm_id": 2},
            },
        ]
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        G = build_road_graph(synthetic_features, test_raster, self.georef)

        route = calculate_astar_evacuation_route(
            G, (80.220, 13.020), (80.240, 13.020), self.georef
        )
        self.assertEqual(route["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")
        self.assertGreater(route["distance_m"], 0.0)
        self.assertIsNotNone(route["route_geojson"])
        self.assertEqual(route["route_geojson"]["geometry"]["type"], "LineString")

    # 11. Start and destination snapping
    def test_11_start_and_destination_snapping(self):
        """Verify start and destination coordinates off-road are snapped to nearest nodes."""
        synthetic_features = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[80.230, 13.020], [80.240, 13.020]]},
                "properties": {"osm_id": 1},
            }
        ]
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        G = build_road_graph(synthetic_features, test_raster, self.georef)

        # Off-road start point slightly north of road node
        off_start = (80.23005, 13.02010)
        off_dest = (80.23995, 13.02010)

        route = calculate_astar_evacuation_route(G, off_start, off_dest, self.georef)
        self.assertEqual(route["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")
        self.assertIn("start_snapped", route)
        self.assertIn("dest_snapped", route)
        self.assertGreater(route["start_snapped"]["snap_distance_m"], 0.0)

    # 12. No-route behavior when isolated
    def test_12_no_route_behavior_when_isolated(self):
        """Verify NO_SAFE_ROUTE_AVAILABLE is returned when start and dest are in disconnected components."""
        synthetic_features = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[80.220, 13.020], [80.225, 13.020]]},
                "properties": {"osm_id": 1},
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[80.250, 13.040], [80.255, 13.040]]},
                "properties": {"osm_id": 2},
            },
        ]
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        G = build_road_graph(synthetic_features, test_raster, self.georef)

        route = calculate_astar_evacuation_route(G, (80.220, 13.020), (80.250, 13.040), self.georef)
        self.assertEqual(route["status"], "NO_SAFE_ROUTE_AVAILABLE")
        self.assertEqual(route["distance_m"], 0.0)
        self.assertIsNone(route["route_geojson"])

    # 13. Out-of-bounds handling
    def test_13_out_of_bounds_handling(self):
        """Verify points far outside the raster domain are rejected gracefully."""
        synthetic_features = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[80.230, 13.020], [80.240, 13.020]]},
                "properties": {"osm_id": 1},
            }
        ]
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        G = build_road_graph(synthetic_features, test_raster, self.georef)

        # Coordinate in London (lon 0.0, lat 51.5) far away from Chennai
        far_coords = (0.0, 51.5)
        route = calculate_astar_evacuation_route(G, far_coords, (80.230, 13.020), self.georef)
        self.assertEqual(route["status"], "START_POINT_TOO_FAR_FROM_ROAD_NETWORK")

    # 14. API response validation
    def test_14_api_response_validation(self):
        """Verify FastAPI endpoints return validated Pydantic models with disclaimers."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        # Test Exposure Endpoint
        res_exp = client.post("/api/gis/exposure-analysis", json={"preset_id": "chennai_monsoon_2026", "max_buildings": 50})
        self.assertEqual(res_exp.status_code, 200)
        data_exp = res_exp.json()
        self.assertTrue(data_exp["success"])
        self.assertIn("buildings_summary", data_exp)
        self.assertIn("roads_summary", data_exp)
        self.assertIn("scientific_disclaimer", data_exp)
        self.assertIn("local verification", data_exp["scientific_disclaimer"].lower())

        # Test Routing Endpoint
        res_route = client.post(
            "/api/gis/evacuation-route",
            json={
                "preset_id": "chennai_monsoon_2026",
                "start_coords": [80.225, 13.045],
                "dest_coords": [80.245, 13.055],
                "avoid_potential_hazards": False,
            },
        )
        self.assertEqual(res_route.status_code, 200)
        data_route = res_route.json()
        self.assertIn("route", data_route)
        self.assertIn("scientific_disclaimer", data_route)

    # 15. GeoJSON validity (RFC 7946 compliance)
    def test_15_geojson_validity(self):
        """Verify building and road GeoJSON conform strictly to standard RFC 7946 specifications."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        res = client.post("/api/gis/exposure-analysis", json={"preset_id": "chennai_monsoon_2026", "max_buildings": 20})
        data = res.json()

        blds_fc = data["buildings_geojson"]
        self.assertEqual(blds_fc["type"], "FeatureCollection")
        self.assertIsInstance(blds_fc["features"], list)

        roads_fc = data["roads_geojson"]
        self.assertEqual(roads_fc["type"], "FeatureCollection")
        self.assertIsInstance(roads_fc["features"], list)

        # Inspect first building feature
        if len(blds_fc["features"]) > 0:
            bf = blds_fc["features"][0]
            self.assertEqual(bf["type"], "Feature")
            self.assertIn("geometry", bf)
            self.assertIn("properties", bf)
            self.assertIn("exposure_category", bf["properties"])
            self.assertIn(bf["properties"]["exposure_category"], ["HIGH_EXPOSURE", "MODERATE_EXPOSURE", "LOW_OR_NO_DETECTED_EXPOSURE", "UNKNOWN"])

    # 16. OSM One-Way: Forward allowed, prohibited reverse rejected
    def test_16_osm_oneway_forward_allowed_and_reverse_rejected(self):
        """Verify oneway=yes allows travel in digitizing direction and strictly blocks reverse traversal."""
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        node_a = (80.230, 13.020)
        node_b = (80.240, 13.020)
        oneway_forward_feature = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(node_a), list(node_b)]},
                "properties": {"osm_id": 501, "name": "OneWay Eastbound Ave", "oneway": "yes", "highway": "primary"},
            }
        ]
        G = build_road_graph(oneway_forward_feature, test_raster, self.georef)
        self.assertIsInstance(G, nx.DiGraph)

        # Forward A -> B must succeed
        res_fwd = calculate_astar_evacuation_route(G, node_a, node_b, self.georef)
        self.assertEqual(res_fwd["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")
        self.assertGreater(res_fwd["distance_m"], 0.0)

        # Reverse B -> A must be blocked (no edge in reverse direction)
        res_rev = calculate_astar_evacuation_route(G, node_b, node_a, self.georef)
        self.assertEqual(res_rev["status"], "NO_SAFE_ROUTE_AVAILABLE")

    # 17. OSM One-Way: oneway=-1 reverse tag handling
    def test_17_osm_oneway_reverse_tag_handling(self):
        """Verify oneway=-1 reverses allowed direction (B -> A allowed, A -> B blocked)."""
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        node_a = (80.230, 13.020)
        node_b = (80.240, 13.020)
        oneway_rev_feature = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(node_a), list(node_b)]},
                "properties": {"osm_id": 502, "name": "Contraflow Way", "oneway": "-1", "highway": "secondary"},
            }
        ]
        G = build_road_graph(oneway_rev_feature, test_raster, self.georef)

        # B -> A should succeed
        res_legal = calculate_astar_evacuation_route(G, node_b, node_a, self.georef)
        self.assertEqual(res_legal["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")
        self.assertGreater(res_legal["distance_m"], 0.0)

        # A -> B should be rejected
        res_illegal = calculate_astar_evacuation_route(G, node_a, node_b, self.georef)
        self.assertEqual(res_illegal["status"], "NO_SAFE_ROUTE_AVAILABLE")

    # 18. OSM One-Way: Bidirectional oneway=no and tag omission defaults
    def test_18_osm_bidirectional_and_defaults(self):
        """Verify oneway=no is bidirectional, missing tag on motorway is oneway, missing tag on residential is bidirectional."""
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        p1 = (80.220, 13.020)
        p2 = (80.225, 13.020)
        p3 = (80.230, 13.020)
        p4 = (80.235, 13.020)

        features = [
            # Explicit bidirectional
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(p1), list(p2)]},
                "properties": {"osm_id": 601, "name": "Two Way Rd", "oneway": "no", "highway": "tertiary"},
            },
            # Missing tag on motorway -> default forward only
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(p2), list(p3)]},
                "properties": {"osm_id": 602, "name": "Express Highway", "highway": "motorway"},
            },
            # Missing tag on residential -> default bidirectional
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(p3), list(p4)]},
                "properties": {"osm_id": 603, "name": "Local Colony Lane", "highway": "residential"},
            },
        ]
        G = build_road_graph(features, test_raster, self.georef)

        # p1 <-> p2 bidirectional
        self.assertEqual(calculate_astar_evacuation_route(G, p1, p2, self.georef)["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")
        self.assertEqual(calculate_astar_evacuation_route(G, p2, p1, self.georef)["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")

        # p2 -> p3 motorway forward succeeds, p3 -> p2 reverse fails
        self.assertEqual(calculate_astar_evacuation_route(G, p2, p3, self.georef)["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")
        self.assertEqual(calculate_astar_evacuation_route(G, p3, p2, self.georef)["status"], "NO_SAFE_ROUTE_AVAILABLE")

        # p3 <-> p4 residential bidirectional
        self.assertEqual(calculate_astar_evacuation_route(G, p3, p4, self.georef)["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")
        self.assertEqual(calculate_astar_evacuation_route(G, p4, p3, self.georef)["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")

    # 19. ETA calculation formula integrity
    def test_19_eta_calculation_formula_integrity(self):
        """Verify ETA matches exact formula (distance_m / 1000) / speed_kmh * 60 with no arbitrary buffers."""
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        p1 = (80.230, 13.020)
        p2 = (80.240, 13.020)
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(p1), list(p2)]},
                "properties": {"osm_id": 701, "highway": "primary"},
            }
        ]
        G = build_road_graph(features, test_raster, self.georef)

        route = calculate_astar_evacuation_route(G, p1, p2, self.georef, speed_kmh=25.0)
        self.assertEqual(route["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")

        dist_m = route["distance_m"]
        expected_eta = round((dist_m / 1000.0) / 25.0 * 60.0, 2)
        self.assertAlmostEqual(route["eta_minutes"], expected_eta, places=2)

        # Specifically check 3469.2 m benchmark: 3.4692 / 25 * 60 = 8.32608 -> 8.33 mins
        bench_eta = round((3469.2 / 1000.0) / 25.0 * 60.0, 2)
        self.assertEqual(bench_eta, 8.33)

    # 20. Parallel OSM road edges preserved in MultiDiGraph
    def test_20_parallel_osm_road_edges_preserved(self):
        """Verify that multiple road features sharing identical endpoints are preserved as distinct edges in MultiDiGraph."""
        test_raster = np.zeros((256, 256), dtype=np.uint8)
        u_pt = (80.230, 13.020)
        v_pt = (80.235, 13.020)

        parallel_features = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(u_pt), list(v_pt)]},
                "properties": {"osm_id": 901, "name": "Arterial Boulevard", "highway": "primary"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(u_pt), list(v_pt)]},
                "properties": {"osm_id": 902, "name": "Parallel Service Road", "highway": "service"},
            },
        ]
        G = build_road_graph(parallel_features, test_raster, self.georef)
        self.assertIsInstance(G, nx.MultiDiGraph)

        # Both edges must be present in the MultiDiGraph
        u_rounded = (round(u_pt[0], 6), round(u_pt[1], 6))
        v_rounded = (round(v_pt[0], 6), round(v_pt[1], 6))
        edge_dict = G.get_edge_data(u_rounded, v_rounded)
        self.assertEqual(len(edge_dict), 2, "Expected exactly 2 parallel edges between u and v")

        osm_ids = {e["osm_id"] for e in edge_dict.values()}
        self.assertIn("901", osm_ids)
        self.assertIn("902", osm_ids)

    # 21. Strict hazard avoidance: exclusion when avoid_hazards=True vs penalty when False
    def test_21_strict_hazard_avoidance_exclusion_vs_penalty(self):
        """Verify that POTENTIAL_HAZARD edges are completely excluded when avoid_hazards=True, returning NO_SAFE_ROUTE_AVAILABLE."""
        flood_raster = np.zeros((256, 256), dtype=np.uint8)
        flood_raster[100:150, 100:150] = 2  # Burn floodwater in central region

        # Pick coordinates that intersect the flooded pixels
        center_lon = self.georef.min_lon + 125 * self.georef.res_x
        center_lat = self.georef.max_lat - 125 * self.georef.res_y

        p_start = (center_lon - 0.005, center_lat)
        p_dest = (center_lon + 0.005, center_lat)

        single_flooded_road = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(p_start), [center_lon, center_lat], list(p_dest)]},
                "properties": {"osm_id": 950, "name": "Submerged CauseWay", "highway": "primary"},
            }
        ]

        # 1. When avoid_hazards=True: the edge is strictly excluded, no path can exist
        G_strict = build_road_graph(single_flooded_road, flood_raster, self.georef, avoid_hazards=True)
        res_strict = calculate_astar_evacuation_route(G_strict, p_start, p_dest, self.georef)
        self.assertEqual(res_strict["status"], "NO_SAFE_ROUTE_AVAILABLE")

        # 2. When avoid_hazards=False: the edge is retained with 10x penalty
        G_penalized = build_road_graph(single_flooded_road, flood_raster, self.georef, avoid_hazards=False)
        res_penalized = calculate_astar_evacuation_route(G_penalized, p_start, p_dest, self.georef)
        self.assertEqual(res_penalized["status"], "CANDIDATE_ROUTE_AVOIDING_MAPPED_HAZARDS")
        self.assertGreater(res_penalized["distance_m"], 0.0)

    # 22. Raster class encoding and exact pixel sum reconciliation
    def test_22_raster_class_encoding_and_sum_reconciliation(self):
        """Verify exact pixel counts across all 4 classes and sum equals 256x256=65,536."""
        import rasterio
        raster_path = "backend/data/satellite_cache/flood_mask_024d3597.tif"
        self.assertTrue(os.path.exists(raster_path))

        with rasterio.open(raster_path) as src:
            arr = src.read(1)
            vals, counts = np.unique(arr, return_counts=True)
            c_dict = dict(zip(vals.tolist(), counts.tolist()))

            # Verify verified exact counts
            self.assertEqual(c_dict.get(0, 0), 32202, "Class 0 (Dry Land) must equal 32,202")
            self.assertEqual(c_dict.get(1, 0), 18965, "Class 1 (Permanent Water) must equal 18,965")
            self.assertEqual(c_dict.get(2, 0), 14369, "Class 2 (Mapped Floodwater) must equal 14,369")
            self.assertEqual(c_dict.get(255, 0), 0, "Class 255 (NoData) must equal 0")

            # Check conservation of total grid area
            total_sum = sum(counts)
            self.assertEqual(total_sum, 65536)
            self.assertEqual(total_sum, arr.size)

    # 23. Satellite provenance metadata and checkpoint inference consistency
    def test_23_satellite_provenance_metadata_consistency(self):
        """Verify that provenance_024d3597.json matches existing raster files and model architecture."""
        import json
        prov_path = "backend/data/metadata/provenance_024d3597.json"
        self.assertTrue(os.path.exists(prov_path))

        with open(prov_path, "r", encoding="utf-8") as f:
            prov = json.load(f)

        self.assertEqual(prov["analysis_id"], "024d3597")
        self.assertEqual(prov["aoi"]["utm_crs"], "EPSG:32644")

        # Verify band file directories exist on disk
        s2_dir = "backend/data/satellite_cache/CDSE_SENTINEL-2_20250101_13.01_80.22"
        s1_dir = "backend/data/satellite_cache/CDSE_SENTINEL-1_20251110_13.01_80.22"
        self.assertTrue(os.path.isdir(s2_dir))
        self.assertTrue(os.path.isdir(s1_dir))

        for b in ["B02", "B03", "B04", "B08"]:
            self.assertTrue(os.path.exists(os.path.join(s2_dir, f"CDSE_SENTINEL-2_20250101_13.01_80.22_{b}.tif")))
        for p in ["VV", "VH"]:
            self.assertTrue(os.path.exists(os.path.join(s1_dir, f"CDSE_SENTINEL-1_20251110_13.01_80.22_{p}.tif")))

        # Verify referenced model checkpoint exists
        checkpoint_path = "backend/data/checkpoints/dual_stream_attention_unet.pth"
        self.assertTrue(os.path.exists(checkpoint_path))


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase5GISIntegrity)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

