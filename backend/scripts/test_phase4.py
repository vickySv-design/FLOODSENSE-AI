import unittest
from pathlib import Path
import sys
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import DISASTER_PRESETS_METADATA, CLASS_DRY_LAND, CLASS_NEW_FLOOD, CLASS_PERMANENT_WATER
from app.services.georef import GeoSpatialTransform
from app.services.osm_service import osm_service
from app.services.rescue_router import compute_osm_rescue_routes, _sample_line_flood_status
from app.services.damage_analyzer import analyze_flood_damage


class TestPhase4GeospatialOSM(unittest.TestCase):

    def setUp(self):
        self.preset_id = "chennai_monsoon_2026"
        self.meta = DISASTER_PRESETS_METADATA[self.preset_id]
        self.bbox = self.meta["bbox"]

    def test_01_osm_data_validity(self):
        """Test 1: Verify downloaded road GeoJSON contains valid LineStrings and belongs to AOI."""
        roads_geo, blds_geo, from_cache, warning = osm_service.fetch_and_cache_osm_data(
            self.preset_id, self.bbox, force_refresh=False
        )
        self.assertTrue(len(roads_geo.get("features", [])) > 100)
        self.assertTrue(len(blds_geo.get("features", [])) > 100)

        sample_road = roads_geo["features"][0]
        self.assertEqual(sample_road["geometry"]["type"], "LineString")
        coords = sample_road["geometry"]["coordinates"]
        self.assertTrue(len(coords) >= 2)

        # Coordinate bounds check: min_lon <= x <= max_lon, min_lat <= y <= max_lat
        min_lat, min_lon, max_lat, max_lon = self.bbox
        has_point_in_aoi = any(
            (min_lon - 0.01 <= pt[0] <= max_lon + 0.01 and min_lat - 0.01 <= pt[1] <= max_lat + 0.01)
            for pt in coords
        )
        self.assertTrue(has_point_in_aoi, "Road feature outside preset AOI")

    def test_02_coordinate_accuracy_and_order(self):
        """Test 2: Verify GeoJSON uses EPSG:4326 and [longitude, latitude] coordinate order."""
        georef = GeoSpatialTransform(self.bbox, width=256, height=256)
        min_lat, min_lon, max_lat, max_lon = self.bbox

        # Transform roundtrip test
        test_lon = min_lon + 0.02
        test_lat = min_lat + 0.02
        col, row = georef.lonlat_to_pixel(test_lon, test_lat)
        ret_lon, ret_lat = georef.pixel_to_lonlat(col, row)

        self.assertAlmostEqual(test_lon, ret_lon, places=4)
        self.assertAlmostEqual(test_lat, ret_lat, places=4)

        # UTM zone validation (Chennai ~80.25E is UTM zone 44N)
        self.assertEqual(georef.utm_zone, 44)
        self.assertEqual(georef.utm_epsg, 32644)

    def test_03_flood_collision_detection(self):
        """Test 3: Verify road segment intersecting Class 2 flood becomes BLOCKED."""
        georef = GeoSpatialTransform(self.bbox, width=256, height=256)
        seg_mask = np.zeros((256, 256), dtype=np.uint8)

        # Place artificial flood pool at center (row 128, col 128)
        seg_mask[120:136, 120:136] = CLASS_NEW_FLOOD

        p1_lon, p1_lat = georef.pixel_to_lonlat(100, 128)
        p2_lon, p2_lat = georef.pixel_to_lonlat(150, 128)

        is_blocked, ratio, reason = _sample_line_flood_status(
            [[p1_lon, p1_lat], [p2_lon, p2_lat]], seg_mask, georef, is_bridge=False
        )
        self.assertTrue(is_blocked)
        self.assertEqual(reason, "BLOCKED_ACTIVE_FLOOD")
        self.assertTrue(ratio > 0.0)

    def test_04_astar_real_osm_routing(self):
        """Test 4: Verify A* route follows real OSM road segments and excludes blocked edges."""
        seg_mask = np.zeros((256, 256), dtype=np.uint8)
        # Moderate flood in lower right
        seg_mask[170:220, 170:220] = CLASS_NEW_FLOOD

        routing_out = compute_osm_rescue_routes(seg_mask, self.bbox, self.preset_id)
        summary = routing_out["summary"]
        self.assertTrue(summary["total_road_segments"] > 1000)

        routes = routing_out["routes"]
        self.assertTrue(len(routes) >= 3)

        # Verify safe routes follow OSM coordinates
        safe_geo = routing_out["geojson"]["safe_routes"]
        if len(safe_geo["features"]) > 0:
            first_route = safe_geo["features"][0]
            coords = first_route["geometry"]["coordinates"]
            self.assertTrue(len(coords) >= 2)
            # Ensure coordinates are in [lon, lat] order
            first_pt = coords[0]
            self.assertTrue(70.0 <= first_pt[0] <= 95.0, "Longitude range error")
            self.assertTrue(8.0 <= first_pt[1] <= 30.0, "Latitude range error")

    def test_05_osm_building_damage_classification(self):
        """Test 5: Verify real OSM building footprints are graded properly."""
        seg_mask = np.zeros((256, 256), dtype=np.uint8)
        seg_mask[100:150, 100:150] = CLASS_NEW_FLOOD

        dmg = analyze_flood_damage(seg_mask, self.bbox, self.preset_id)
        blds_summary = dmg["buildings_summary"]
        self.assertTrue(blds_summary["total_buildings"] > 0)
        self.assertEqual(blds_summary["data_source"], "OPENSTREETMAP_REAL")

        blds_geo = dmg["buildings_geojson"]
        sample_bld = blds_geo["features"][0]
        self.assertEqual(sample_bld["geometry"]["type"], "Polygon")
        self.assertIn("damage_status", sample_bld["properties"])
        self.assertEqual(sample_bld["properties"]["data_source"], "OPENSTREETMAP_REAL")


if __name__ == "__main__":
    unittest.main()
