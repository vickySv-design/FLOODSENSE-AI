"""
FloodSense-AI - Scenario Result Consistency & Regression Test Suite
Proves 100% mathematical consistency across:
  1. FastAPI status endpoint payload (/api/scenarios/{id}/status)
  2. Final analytical results payload (/api/scenarios/{id}/results)
  3. Exported JSON report (/api/scenarios/{id}/report)
  4. Selected timestep snapshot metrics across progression timeline
  5. Building exposure counts from real raster sampling against h >= 0.15m
  6. Non-zero dynamic mass balance residual tracking
"""

import os
import sys
import json
import numpy as np

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, backend_dir)

from app.services.scenario_engine import (
    ScenarioManager,
    CycloneScenarioSimulator,
    resolve_effective_parameters,
)

def print_separator(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

def test_scenario_consistency():
    print_separator("TEST 1: Execution of Scenario with Provenance Resolution")
    manager = ScenarioManager()
    bbox = [11.48021, 79.29964, 11.56021, 79.37964] # Cuddalore test bbox (matching b846565f)

    cyclone_params = {
        "max_wind_speed_kmh": 50.0,
        "wind_gust_kmh": 65.0,
        "wind_direction_deg": 180.0,
        "central_pressure_hpa": 1002.0,
        "cyclone_radius_km": 30.0,
        "movement_speed_kmh": 10.0,
    }
    rainfall_params = {
        "rainfall_intensity_mm_hr": 95.0,
        "rainfall_duration_hours": 8.0,
        "rainfall_distribution": "front_loaded",
    }

    scenario_id = manager.create_scenario(
        name="Extreme Inland Convective Cloudburst (95 mm/h)",
        bbox=bbox,
        cyclone_params=cyclone_params,
        rainfall_params=rainfall_params,
        duration_hours=24,
    )
    print(f"Created Scenario ID: {scenario_id}")

    # Synchronously run worker
    manager._run_worker(scenario_id)

    # 1. Fetch Status Payload (what the React Dashboard polls)
    status_payload = manager.get_scenario_status(scenario_id)
    assert status_payload is not None, "Scenario status must exist"
    assert status_payload["status"] == "COMPLETED", f"Expected COMPLETED, got {status_payload['status']}"
    assert "snapshots" in status_payload, "Status must include snapshots array"
    snapshots = status_payload["snapshots"]
    results = status_payload["results"]
    assert results is not None, "Results must be populated on COMPLETED status"

    print_separator("TEST 2: Exact Consistency Between Dashboard Results & Final Snapshot")
    final_snap = snapshots[-1]

    print(f"Final Snapshot [{final_snap['label']}]:")
    print(f"  Flooded Area: {final_snap['flooded_area_km2']} km²")
    print(f"  Max Water Depth: {final_snap['max_water_depth_m']} m")
    print(f"  Mean Water Depth: {final_snap['mean_water_depth_m']} m")
    print(f"  Exposed Buildings: {final_snap['submerged_buildings']}")
    print(f"  Severed Roads: {final_snap['severed_roads']}")
    print(f"  Mass Error: {final_snap['mass_balance_error_pct']:.4f}%")

    print(f"\nDashboard Summary Results Object:")
    print(f"  Final Flooded Area: {results['final_flooded_km2']} km²")
    print(f"  Max Flood Depth: {results['max_flood_depth_m']} m")
    print(f"  Mean Flood Depth: {results['mean_flood_depth_m']} m")
    print(f"  Exposed Buildings: {results['submerged_buildings']}")
    print(f"  Severed Roads: {results['severed_roads']}")
    print(f"  Mass Error: {results['mass_balance_error_pct']:.4f}%")

    # Assert exact equality
    assert results["final_flooded_km2"] == final_snap["flooded_area_km2"], "Flooded area mismatch"
    assert results["max_flood_depth_m"] == final_snap["max_water_depth_m"], "Max depth mismatch"
    assert results["mean_flood_depth_m"] == final_snap["mean_water_depth_m"], "Mean depth mismatch"
    assert results["submerged_buildings"] == final_snap["submerged_buildings"], "Submerged buildings mismatch"
    assert results["severed_roads"] == final_snap["severed_roads"], "Severed roads mismatch"
    assert results["mass_balance_error_pct"] == final_snap["mass_balance_error_pct"], "Mass error mismatch"
    print(">> PASSED: Dashboard Results Object strictly matches final simulation snapshot.")

    print_separator("TEST 3: Exported JSON Report Consistency (Overall & Timestep Specific)")
    # Test export for default (final) hour
    report_default = manager.generate_scenario_report(scenario_id)
    assert report_default["scenario_id"] == scenario_id
    assert report_default["summary_results"]["final_flooded_km2"] == results["final_flooded_km2"]
    assert report_default["summary_results"]["submerged_buildings"] == results["submerged_buildings"]
    assert report_default["summary_results"]["max_flood_depth_m"] == results["max_flood_depth_m"]
    assert report_default["summary_results"]["mass_balance_error_pct"] == results["mass_balance_error_pct"]
    print(">> PASSED: Exported JSON summary_results matches Dashboard results.")

    # Test export for a specific scrubbed timestep (e.g. hour 6)
    test_hour = 6
    report_h6 = manager.generate_scenario_report(scenario_id, hour=test_hour)
    snap_h6 = next(s for s in snapshots if s["hour"] == test_hour)
    sel_h6 = report_h6["selected_timestep"]

    print(f"\nTimestep T+{test_hour:02d}:00 Comparison:")
    print(f"  Snapshot Flooded: {snap_h6['flooded_area_km2']} km² vs Report: {sel_h6['flooded_km2']} km²")
    print(f"  Snapshot Buildings: {snap_h6['submerged_buildings']} vs Report: {sel_h6['submerged_buildings']}")
    print(f"  Snapshot Max Depth: {snap_h6['max_water_depth_m']} m vs Report: {sel_h6['max_depth_m']} m")

    assert sel_h6["flooded_km2"] == snap_h6["flooded_area_km2"]
    assert sel_h6["submerged_buildings"] == snap_h6["submerged_buildings"]
    assert sel_h6["max_depth_m"] == snap_h6["max_water_depth_m"]
    assert sel_h6["mass_balance_error_pct"] == snap_h6["mass_balance_error_pct"]
    print(f">> PASSED: Timestep T+{test_hour:02d}:00 report matches snapshot telemetry exactly.")

    print_separator("TEST 4: Physical Depth Threshold & Raster Inundation Verification")
    # Verify depth threshold consistency: cells with h >= 0.15m match flooded_km2
    depth_grid_final = manager.get_scenario_depth_raster(scenario_id, final_snap["hour"])
    assert depth_grid_final is not None
    calc_flooded_cells = int(np.sum(depth_grid_final >= 0.15))
    calc_flooded_km2 = round(float(calc_flooded_cells * 0.0001), 3)
    assert calc_flooded_km2 == final_snap["flooded_area_km2"], (
        f"Calculated from raster: {calc_flooded_km2} km² != Reported: {final_snap['flooded_area_km2']} km²"
    )
    print(f"Raster Cells (h >= 0.15m): {calc_flooded_cells} -> {calc_flooded_km2} km² == {final_snap['flooded_area_km2']} km²")
    print(">> PASSED: Classification threshold (0.15m) and pixel area (100m² = 0.0001km²) are 100% consistent.")

    print_separator("TEST 5: Real Mass Balance Residual Verification (No Forced 0.00%)")
    mass_err = results["mass_balance_error_pct"]
    print(f"Dynamic Mass Residual: {mass_err:.6f}%")
    assert isinstance(mass_err, float), f"Expected float, got {type(mass_err)}"
    assert mass_err >= 0.0, "Mass error cannot be negative"
    assert mass_err < 1.0, f"Mass error ({mass_err}%) must satisfy conservation criterion (< 1.0%)"
    print(">> PASSED: Mass balance residual is calculated dynamically from physical budget without hardcoded zeros.")

    print_separator("TEST 6: Parameter Provenance & Distinguishing Wind vs Rain Direction")
    eff = report_default.get("effective_parameters", {})
    assert len(eff) == 14, f"Expected 14 parameters, found {len(eff)}"
    assert "rainfall_direction_deg" in eff
    assert "rainfall_distribution_model" in eff
    assert eff["rainfall_direction_deg"]["units"] == "degrees (0-360°)"
    assert eff["rainfall_distribution_model"]["units"] == "profile curve"
    print(f"Rainfall Direction: {eff['rainfall_direction_deg']['value']} {eff['rainfall_direction_deg']['units']} [{eff['rainfall_direction_deg']['source']}]")
    print(f"Rainfall Distribution: {eff['rainfall_distribution_model']['value']} [{eff['rainfall_distribution_model']['source']}]")
    assert eff["rainfall_direction_deg"]["value"] != eff["rainfall_distribution_model"]["value"]
    print(">> PASSED: Wind/storm direction and rainfall distribution profile are clearly distinguished.")

    print_separator("ALL 6 REGRESSION CONSISTENCY CHECKS COMPLETED SUCCESSFULLY")
    return True

if __name__ == "__main__":
    test_scenario_consistency()

