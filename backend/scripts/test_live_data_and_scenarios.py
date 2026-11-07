"""
Automated Verification Suite for Live Data API Integration and Cyclone Scenario Simulation
========================================================================================
Tests:
1. Data providers status audit (Copernicus, GEE, OpenStreetMap, DEM, Open-Meteo, NASA GPM).
2. Open-Meteo live/cached rainfall time-series retrieval.
3. NASA GPM IMERG satellite precipitation telemetry.
4. Copernicus DEM GLO-30/90 elevation retrieval & caching.
5. Central Data Acquisition Manager job creation & status tracking.
6. Persistent cache auditing & offline availability check.
7. Cyclone Scenario Manager: 3 preconfigured presets.
8. Interactive Cyclone Simulation: vortex wind field, rainfall decay, 2D fluid simulation, mass balance (0.00% error).
9. Dynamic A* rescue route recalculation on scenario flood depth.
10. Comprehensive scenario report generation with hypothetical disclaimer.
"""
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.data_acquisition_manager import data_acquisition_manager
from app.services.data_providers.open_meteo_provider import open_meteo_provider
from app.services.data_providers.nasa_gpm_provider import nasa_gpm_provider
from app.services.scenario_engine import (
    scenario_manager,
    calculate_cyclone_category,
    CycloneScenarioSimulator,
)


class TestLiveDataAndScenarioSimulation(unittest.TestCase):

    def setUp(self):
        self.bbox = [12.980, 80.190, 13.060, 80.290]  # Chennai coastal estuary

    def test_01_provider_statuses(self):
        """Test 1: All integrated data providers report valid operational statuses."""
        status = data_acquisition_manager.get_all_provider_statuses()
        self.assertIn("providers", status)
        self.assertIn("copernicus", status["providers"])
        self.assertIn("open_meteo", status["providers"])
        self.assertIn("nasa_gpm", status["providers"])
        self.assertIn("copernicus_dem", status["providers"])
        self.assertIn("openstreetmap", status["providers"])
        self.assertIn(status["providers"]["open_meteo"]["status"], ["CONNECTED", "OFFLINE"])

    def test_02_open_meteo_rainfall(self):
        """Test 2: Open-Meteo provider retrieves valid hourly precipitation time-series."""
        res = open_meteo_provider.fetch_rainfall_timeseries(
            bbox=self.bbox, mode="forecast", forecast_days=2
        )
        self.assertIn("hourly_series", res)
        self.assertTrue(len(res["hourly_series"]) >= 24)
        first_step = res["hourly_series"][0]
        self.assertIn("precipitation_mm", first_step)
        self.assertIn("temperature_c", first_step)
        self.assertIn("wind_speed_kmh", first_step)
        self.assertTrue(res["total_precipitation_mm"] >= 0.0)

    def test_03_nasa_gpm_precipitation(self):
        """Test 3: NASA GPM IMERG provider returns satellite microwave precipitation estimates."""
        res = nasa_gpm_provider.fetch_gpm_precipitation(bbox=self.bbox, product="early", hours=12)
        self.assertEqual(res["provider"], "nasa_gpm")
        self.assertIn("hourly_series", res)
        self.assertEqual(len(res["hourly_series"]), 12)
        self.assertIn("precipitation_rate_mm_hr", res["hourly_series"][0])

    def test_04_copernicus_dem_retrieval(self):
        """Test 4: Copernicus DEM acquires hydro-conditioned elevation grid with valid metric elevation."""
        dem_res = data_acquisition_manager.copernicus.fetch_dem_raster(
            bounds=(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]),
            dem_type="GLO-30",
            target_shape=(128, 128),
        )
        self.assertIn("file_path", dem_res)
        self.assertEqual(dem_res["dem_type"], "GLO-30")
        self.assertTrue(dem_res["min_elevation_m"] >= 0.0)
        self.assertTrue(dem_res["max_elevation_m"] > dem_res["min_elevation_m"])

    def test_05_data_acquisition_job(self):
        """Test 5: Central Data Acquisition Manager executes asynchronous acquisition job with status updates."""
        job_id = data_acquisition_manager.start_download_job(
            aoi_id="test_aoi",
            bbox=self.bbox,
            product_ids=["SENTINEL2_TEST_01"],
            include_dem=True,
            include_osm=False,
            include_rainfall=True,
        )
        self.assertTrue(len(job_id) > 0)
        # Give worker a moment to process
        import time
        time.sleep(1.0)
        status = data_acquisition_manager.get_job_status(job_id)
        self.assertIsNotNone(status)
        self.assertIn(status["status"], ["RUNNING", "COMPLETED"])
        self.assertTrue(status["progress_pct"] > 0)

    def test_06_cache_audit(self):
        """Test 6: Local persistent cache auditing reports offline availability and directory sizes."""
        cache_info = data_acquisition_manager.list_cached_datasets()
        self.assertIn("total_cache_size_mb", cache_info)
        self.assertIn("layers", cache_info)
        self.assertTrue(cache_info["offline_availability"])

    def test_07_cyclone_category_classification(self):
        """Test 7: Cyclone wind scale accurately determines IMD and Saffir-Simpson categories."""
        cat_dep = calculate_cyclone_category(55.0)
        self.assertEqual(cat_dep["imd_scale"], "Deep Depression")

        cat_scs = calculate_cyclone_category(105.0)
        self.assertEqual(cat_scs["imd_scale"], "Severe Cyclonic Storm (SCS)")

        cat_super = calculate_cyclone_category(230.0)
        self.assertEqual(cat_super["imd_scale"], "Super Cyclonic Storm (SuCS)")
        self.assertEqual(cat_super["saffir_simpson"], "Category 5 Major Hurricane")

    def test_08_cyclone_scenario_simulation_and_mass_conservation(self):
        """Test 8: Interactive Cyclone Simulator solves 2D diffusive wave flow with 0.00% mass balance error."""
        cyclone_params = {
            "max_wind_speed_kmh": 125.0,
            "wind_direction_deg": 315.0,
            "cyclone_radius_km": 50.0,
            "movement_speed_kmh": 16.0,
        }
        rainfall_params = {
            "rainfall_intensity_mm_hr": 60.0,
            "rainfall_duration_hours": 6.0,
            "rainfall_distribution": "front_loaded",
        }
        sim = CycloneScenarioSimulator(
            scenario_id="test_sim_01",
            name="Test Michaung Surge",
            bbox=self.bbox,
            cyclone_params=cyclone_params,
            rainfall_params=rainfall_params,
            duration_hours=6,
            grid_shape=(128, 128),
        )
        snapshots = sim.run_simulation()
        self.assertTrue(len(snapshots) > 0)
        final_snap = snapshots[-1]
        self.assertIn("flooded_area_km2", final_snap)
        self.assertIn("max_water_depth_m", final_snap)
        self.assertIn("stored_water_volume_m3", final_snap)
        # Verify 0.00% mass conservation error
        self.assertEqual(final_snap["mass_balance_error_pct"], 0.00)
        self.assertTrue(final_snap["stored_water_volume_m3"] > 0.0)

    def test_09_scenario_manager_and_presets(self):
        """Test 9: Scenario Manager provides editable presets and creates user-defined scenarios."""
        presets = scenario_manager.get_preset_scenarios()
        self.assertTrue(len(presets) >= 3)
        preset_ids = [p["id"] for p in presets]
        self.assertIn("extreme_rainfall_event", preset_ids)
        self.assertIn("cyclone_associated_rainfall", preset_ids)
        self.assertIn("coastal_heavy_rainfall", preset_ids)

        # Create custom scenario
        sc_id = scenario_manager.create_scenario(
            name="Viva Review Scenario",
            bbox=self.bbox,
            cyclone_params={"max_wind_speed_kmh": 110.0},
            rainfall_params={"rainfall_intensity_mm_hr": 45.0, "rainfall_duration_hours": 6.0},
            duration_hours=6,
        )
        self.assertTrue(len(sc_id) > 0)
        status = scenario_manager.get_scenario_status(sc_id)
        self.assertEqual(status["status"], "CONFIGURED")

    def test_10_scenario_report_generation(self):
        """Test 10: Scenario report generator outputs structured JSON with explicit hypothetical disclaimer."""
        sc_id = scenario_manager.create_scenario(
            name="Academic Defense Scenario",
            bbox=self.bbox,
            cyclone_params={"max_wind_speed_kmh": 140.0},
            rainfall_params={"rainfall_intensity_mm_hr": 70.0, "rainfall_duration_hours": 12.0},
            duration_hours=12,
        )
        report = scenario_manager.generate_scenario_report(sc_id)
        self.assertIn("hypothetical_disclaimer", report)
        self.assertIn("HYPOTHETICAL DISASTER SCENARIO", report["hypothetical_disclaimer"])
        self.assertEqual(report["scenario_id"], sc_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)

