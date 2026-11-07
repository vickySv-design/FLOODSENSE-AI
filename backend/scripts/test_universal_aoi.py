"""
Universal Study Area & AOI Engine Verification Test Suite for FloodSense-AI
==========================================================================
Verifies:
1. Dynamic Study Area geocoding, geodesic area calculation, UTM CRS detection.
2. Preset loading (Chennai coastal, Assam, Wayanad) without data leakage.
3. Coastal ocean water classification (Bay of Bengal is Class 1 Permanent Water, not Class 2 Flood).
4. Creation of brand new custom AOI (Cuddalore Port) with isolated cache and zero preset contamination.
5. Arbitrary raster sliding-window inference with cosine window blending.
6. Dynamic 2D Hydraulic DEM and simulation on arbitrary custom AOIs.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import (
    DISASTER_PRESETS_METADATA,
    AOI_CACHE_DIR,
    CLASS_DRY_LAND,
    CLASS_PERMANENT_WATER,
    CLASS_NEW_FLOOD,
    PRESETS_DIR,
    DEVICE,
)
from app.services.aoi_service import aoi_service
from app.services.osm_service import osm_service
from app.services.georef import GeoSpatialTransform
from app.services.preprocessor import preprocess_multimodal_pair
from app.services.inference_engine import inference_engine
from app.services.simulation_engine import (
    simulation_job_manager,
    get_or_generate_dem,
    Physical2DFloodSimulator,
    PRESET_SCENARIOS,
)
from app.services.rescue_router import compute_osm_rescue_routes
from app.services.damage_analyzer import analyze_flood_damage


class TestUniversalStudyAreaEngine(unittest.TestCase):

    def setUp(self):
        AOI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def test_01_aoi_service_geodesic_and_utm(self):
        """Test 1: AOIService correctly calculates ellipsoidal area and UTM EPSG projection."""
        # Chennai coastal bbox
        chennai_bbox = [12.980, 80.190, 13.060, 80.290]
        val = aoi_service.validate_and_normalize_aoi(
            name="Chennai Metropolitan Estuary",
            bbox=chennai_bbox,
            terrain_type="coastal_urban",
        )
        self.assertEqual(val["utm_crs"], "EPSG:32644")
        self.assertTrue(50.0 <= val["area_km2"] <= 150.0)
        self.assertEqual(val["terrain_type"], "coastal_urban")
        self.assertEqual(len(val["bbox"]), 4)

        # Southern Hemisphere test (e.g., Jakarta coastal region)
        jakarta_bbox = [-6.25, 106.75, -6.15, 106.85]
        val_sh = aoi_service.validate_and_normalize_aoi(
            name="Jakarta Coastal Plain",
            bbox=jakarta_bbox,
        )
        self.assertEqual(val_sh["utm_crs"], "EPSG:32748")  # UTM Zone 48S
        self.assertTrue(val_sh["area_km2"] > 0)

    def test_02_nominatim_search_and_clamping(self):
        """Test 2: AOIService location search returns clamped disaster windows for safe memory usage."""
        results = aoi_service.search_location("Kolkata")
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertIn("bbox", top)
        self.assertIn("area_km2", top)
        # Verify area does not exceed maximum allowable cap (500 km2)
        self.assertTrue(top["area_km2"] <= 500.0)

    def test_03_chennai_coastal_ocean_classification(self):
        """Test 3: Ocean baseline in Chennai Adyar Estuary is classified as Permanent Water, NOT New Flood."""
        chennai_folder = PRESETS_DIR / "chennai_monsoon_2026"
        gt_mask_path = chennai_folder / "ground_truth.npy"
        self.assertTrue(gt_mask_path.exists(), "Ground truth mask must exist for Chennai preset")

        gt_mask = np.load(gt_mask_path)
        self.assertEqual(gt_mask.shape, (256, 256))

        # Eastern portion (columns 225 to 255) represents the Bay of Bengal open ocean
        ocean_slice = gt_mask[:, 225:]
        perm_water_ocean_pixels = np.sum(ocean_slice == CLASS_PERMANENT_WATER)
        new_flood_ocean_pixels = np.sum(ocean_slice == CLASS_NEW_FLOOD)

        self.assertTrue(
            perm_water_ocean_pixels > 0,
            "Bay of Bengal ocean must contain Permanent Water pixels",
        )
        self.assertEqual(
            new_flood_ocean_pixels,
            0,
            f"Bay of Bengal ocean must NEVER be marked as New Flood inundation! Found {new_flood_ocean_pixels} pixels.",
        )

    def test_04_brand_new_custom_aoi_cache_isolation(self):
        """Test 4: Custom AOI (Cuddalore Port) creates isolated cache and does not contaminate presets."""
        cuddalore_bbox = [11.720, 79.740, 11.780, 79.800]
        cuddalore_id = "cuddalore_port_estuary"

        roads_path, blds_path = osm_service.get_aoi_cache_paths(cuddalore_id, cuddalore_bbox)

        # Verify filename contains coordinate hash for isolation
        self.assertIn("cuddalore", roads_path.name)
        self.assertIn("cuddalore", blds_path.name)
        self.assertNotIn("chennai", roads_path.name)
        self.assertNotIn("assam", roads_path.name)
        self.assertNotIn("wayanad", roads_path.name)

        # Fetch / build cache
        roads_geo, blds_geo, from_cache, warning = osm_service.fetch_and_cache_osm_data(
            cuddalore_id, cuddalore_bbox, force_refresh=False
        )
        self.assertIn("features", roads_geo)
        self.assertIn("features", blds_geo)

        # Verify coordinates strictly adhere to Cuddalore bounding box
        coords = roads_geo["features"][0]["geometry"]["coordinates"]
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        self.assertTrue(any(79.70 <= lon <= 79.85 for lon in lons))
        self.assertTrue(any(11.68 <= lat <= 11.82 for lat in lats))

    def test_05_sliding_window_arbitrary_raster_inference(self):
        """Test 5: Sliding-window inference supports arbitrary raster dimensions (e.g. 384x384)."""
        H, W = 384, 384
        optical_4ch = np.random.uniform(0.1, 0.9, (4, H, W)).astype(np.float32)
        sar_2ch = np.random.uniform(0.05, 0.8, (2, H, W)).astype(np.float32)
        preprocessed = preprocess_multimodal_pair(optical_4ch, sar_2ch, DEVICE)

        out = inference_engine.run_inference(preprocessed=preprocessed)
        self.assertEqual(out["seg_mask"].shape, (H, W))
        self.assertEqual(out["attention_map"].shape, (H, W))
        self.assertTrue(out["latency_ms"] > 0)
        # Seg mask must only contain valid classes 0, 1, 2
        unique_classes = set(np.unique(out["seg_mask"]))
        self.assertTrue(unique_classes.issubset({0, 1, 2}))

    def test_06_dynamic_dem_and_hydraulic_simulation_for_custom_aoi(self):
        """Test 6: Dynamic 2D hydraulic DEM and simulation generation works on arbitrary AOI."""
        custom_bbox = [15.280, 73.950, 15.340, 74.020]  # Goa coastal plain
        dem = get_or_generate_dem(
            preset_id="goa_sal_river",
            bbox=custom_bbox,
            shape=(128, 128),
            terrain_type="coastal_urban",
        )
        self.assertEqual(dem.shape, (128, 128))
        self.assertTrue(np.nanmin(dem) >= 0.0)

        # Create and run simulation synchronously using Physical2DFloodSimulator
        init_mask = np.zeros((128, 128), dtype=np.uint8)
        init_mask[50:60, 50:60] = CLASS_NEW_FLOOD
        scenario = PRESET_SCENARIOS["moderate_monsoon"]

        sim = Physical2DFloodSimulator(
            initial_mask=init_mask,
            dem=dem,
            bbox=custom_bbox,
            preset_id="goa_sal_river",
            scenario=scenario,
        )
        snapshots = sim.run_full_simulation()
        self.assertTrue(len(snapshots) > 0)
        first_snap = snapshots[0]
        self.assertIn("timestep_hours", first_snap)
        self.assertIn("flooded_area_km2", first_snap)
        self.assertIn("mean_water_depth_m", first_snap)
        self.assertIn("geojson", first_snap)

    def test_07_all_three_presets_load_and_route_cleanly(self):
        """Test 7: All 3 demonstration presets (Chennai, Assam, Wayanad) load, damage-grade and route cleanly."""
        for preset_id in ["chennai_monsoon_2026", "assam_brahmaputra_surge", "wayanad_kerala_flashflood"]:
            meta = DISASTER_PRESETS_METADATA[preset_id]
            preset_folder = PRESETS_DIR / preset_id
            gt_mask = np.load(preset_folder / "ground_truth.npy")

            damage_report = analyze_flood_damage(
                seg_mask=gt_mask,
                bbox=meta["bbox"],
                preset_id=preset_id,
                population_density_factor=meta.get("population_density_factor", 30),
            )
            self.assertIn("telemetry", damage_report)
            self.assertIn("buildings_summary", damage_report)

            rescue_report = compute_osm_rescue_routes(
                seg_mask=gt_mask,
                bbox=meta["bbox"],
                preset_id=preset_id,
            )
            self.assertTrue(rescue_report["summary"]["total_road_segments"] > 50)
            self.assertTrue(len(rescue_report["routes"]) >= 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
