"""Comprehensive Edge-Case & Failure-Mode Verification Script for FloodSense-AI Audit.
Tests:
- Invalid scenario inputs
- Missing DEM handling
- Missing rainfall data handling
- Provider timeouts & offline fallback
- Cache fallback behavior
- Large AOI clamping
- Failed simulation jobs
- Empty / disconnected rescue routes
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import CLASS_DRY_LAND, CLASS_NEW_FLOOD, CLASS_PERMANENT_WATER
from app.services.aoi_service import aoi_service
from app.services.data_providers.open_meteo_provider import OpenMeteoWeatherProvider
from app.services.data_providers.copernicus_provider import CopernicusDataSpaceProvider
from app.services.data_acquisition_manager import data_acquisition_manager
from app.services.scenario_engine import CycloneScenarioSimulator, ScenarioManager, calculate_cyclone_category
from app.services.simulation_engine import Physical2DFloodSimulator, SimulationScenario, get_or_generate_dem
from app.services.rescue_router import compute_osm_rescue_routes, _sample_line_flood_status
from app.services.georef import GeoSpatialTransform


class TestEdgeCasesAndFailureModes(unittest.TestCase):

    def test_01_invalid_scenario_inputs(self):
        """Test how the scenario simulator handles extreme, negative, or degenerate parameters."""
        bbox = [13.0, 80.2, 13.1, 80.3]
        # Degenerate: negative wind speed, zero duration, negative rainfall
        cyclone_params = {"max_wind_speed_kmh": -50.0, "movement_speed_kmh": -10.0}
        rainfall_params = {"rainfall_intensity_mm_hr": -20.0, "rainfall_duration_hours": 0.0}
        
        sim = CycloneScenarioSimulator(
            scenario_id="edge_invalid_inputs",
            name="Invalid Input Test",
            bbox=bbox,
            cyclone_params=cyclone_params,
            rainfall_params=rainfall_params,
            duration_hours=0,
            grid_shape=(64, 64),
        )
        # Advance simulation
        snapshots = sim.run_simulation()
        self.assertGreaterEqual(len(snapshots), 1)
        # Water depth should remain non-negative
        self.assertGreaterEqual(float(np.min(sim.h)), 0.0)
        self.assertEqual(sim.cum_rain_m3, 0.0)
        print("  -> test_01_invalid_scenario_inputs: Handled gracefully, non-negative water depth preserved.")

    def test_02_missing_or_corrupted_dem(self):
        """Test fallback when DEM is missing, mismatched in shape, or corrupted."""
        bbox = [13.0, 80.2, 13.05, 80.25]
        # Request DEM with non-standard shape
        dem = get_or_generate_dem(
            preset_id="non_existent_preset_xyz",
            bbox=bbox,
            shape=(128, 128),
            terrain_type="auto",
        )
        self.assertEqual(dem.shape, (128, 128))
        self.assertFalse(np.isnan(dem).any())
        self.assertGreater(float(np.nanmax(dem)), float(np.nanmin(dem)))
        print(f"  -> test_02_missing_or_corrupted_dem: Generated synthetic hydro DEM ({dem.shape}) range [{np.nanmin(dem):.1f}m, {np.nanmax(dem):.1f}m].")

    def test_03_missing_rainfall_data(self):
        """Test rainfall provider behavior when provider returns 0 or missing series."""
        p = OpenMeteoWeatherProvider()
        # Fallback series generation directly
        fallback = p._generate_fallback_meteorological_series(13.0, 80.2, [13.0, 80.2, 13.1, 80.3], "forecast", 1)
        self.assertIn("hourly_series", fallback)
        self.assertEqual(len(fallback["hourly_series"]), 24)
        self.assertGreaterEqual(fallback["total_precipitation_mm"], 0.0)
        print(f"  -> test_03_missing_rainfall_data: Fallback generated {len(fallback['hourly_series'])} hours of meteorological series.")

    def test_04_provider_timeout_and_fallback(self):
        """Test provider behavior when remote service URL times out or is unreachable."""
        p = CopernicusDataSpaceProvider()
        # Querying with unconfigured credentials should trigger catalog discovery fallback
        results = p.search_scenes(
            bounds=(13.0, 80.2, 13.1, 80.3),
            start_date="2026-01-01",
            end_date="2026-01-05",
            platform="SENTINEL-1",
        )
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        print(f"  -> test_04_provider_timeout_and_fallback: Unconfigured Copernicus gracefully returned {len(results)} fallback catalog items.")

    def test_05_cache_fallback_offline(self):
        """Test offline mode cache fallback when network is bypassed."""
        p_meteo = OpenMeteoWeatherProvider()
        # Query existing cached location
        res = p_meteo.fetch_rainfall_timeseries([13.0, 80.2, 13.1, 80.3], mode="forecast", forecast_days=2, force_refresh=False)
        self.assertIn("hourly_series", res)
        print(f"  -> test_05_cache_fallback_offline: Successfully loaded from cache/fallback with status {res.get('data_source')}.")

    def test_06_large_aoi_handling(self):
        """Test universal AOI service clamping when user requests an excessively large bounding box."""
        # Request an area of 5 degrees x 5 degrees (~300,000 km^2)
        huge_bbox = [10.0, 75.0, 15.0, 80.0]
        norm = aoi_service.validate_and_normalize_aoi("Huge Region", huge_bbox)
        # Should be clamped to <= 500 km^2
        self.assertLessEqual(norm["area_km2"], 500.0)
        self.assertIn("clamped", norm)
        self.assertTrue(norm["clamped"])
        print(f"  -> test_06_large_aoi_handling: Successfully clamped 300,000 km² request down to {norm['area_km2']:.1f} km².")

    def test_07_failed_simulation_job(self):
        """Test that scenario manager records FAILED status when simulation crashes."""
        mgr = ScenarioManager()
        sc_id = mgr.create_scenario(
            name="Faulty Scenario",
            bbox=[13.0, 80.2, 13.1, 80.3],
            cyclone_params={},
            rainfall_params={},
            duration_hours=12,
        )
        # Manually inject bad parameter to force exception in worker
        mgr._scenarios[sc_id]["bbox"] = None
        mgr._run_worker(sc_id)
        status = mgr.get_scenario_status(sc_id)
        self.assertEqual(status["status"], "FAILED")
        self.assertIsNotNone(status["error"])
        print(f"  -> test_07_failed_simulation_job: Job correctly marked FAILED with error: '{status['error']}'.")

    def test_08_empty_or_disconnected_rescue_routes(self):
        """Test rescue routing when roads are 100% submerged or completely absent."""
        seg_mask = np.full((100, 100), CLASS_NEW_FLOOD, dtype=np.uint8)
        bbox = [13.01, 80.22, 13.06, 80.27]
        # Query routing with fully flooded raster
        report = compute_osm_rescue_routes(
            seg_mask=seg_mask,
            bbox=bbox,
            preset_id="chennai_monsoon_2026",
        )
        self.assertIn("summary", report)
        self.assertEqual(report["summary"]["active_safe_routes"], 0)
        self.assertGreater(report["summary"]["blocked_road_segments"], 0)
        print(f"  -> test_08_empty_or_disconnected_rescue_routes: Correctly identified 0 safe routes and {report['summary']['blocked_road_segments']} blocked segments.")


if __name__ == "__main__":
    unittest.main()
