import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestUIAPIIntegration(unittest.TestCase):
    def test_01_health_and_presets(self):
        r = client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn(data.get("status"), ["healthy", "OPERATIONAL"])

        r_pres = client.get("/api/presets")
        self.assertEqual(r_pres.status_code, 200)
        pdata = r_pres.json()
        self.assertIn("presets", pdata)
        self.assertGreaterEqual(len(pdata["presets"]), 3)
        print(">> Test 1 Passed: Health & Presets Endpoints Healthy")

    def test_02_analyze_preset(self):
        r = client.post("/api/analyze", data={"preset_id": "chennai_monsoon_2026"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("telemetry", data)
        self.assertIn("buildings_summary", data)
        self.assertIn("rescue", data)
        self.assertIn("geojson_layers", data)
        print(">> Test 2 Passed: /api/analyze returns valid multi-modal payload")

    def test_03_benchmarks(self):
        r = client.get("/api/benchmarks")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("architectures", data)
        print(">> Test 3 Passed: /api/benchmarks returns valid ablation data")

    def test_04_data_providers_and_cache(self):
        r_prov = client.get("/api/data-providers/status")
        self.assertEqual(r_prov.status_code, 200)
        r_cache = client.get("/api/data/cache")
        self.assertEqual(r_cache.status_code, 200)
        print(">> Test 4 Passed: /api/data-providers/status & /api/data/cache respond 200")

    def test_05_scenario_presets_and_inspect(self):
        r_pres = client.get("/api/scenarios/presets")
        self.assertEqual(r_pres.status_code, 200)
        pdata = r_pres.json()
        self.assertTrue(isinstance(pdata, list))
        self.assertGreaterEqual(len(pdata), 1)

        inspect_payload = {
            "bbox": [13.01, 80.22, 13.06, 80.27],
            "preset_id": pdata[0]["id"],
            "cyclone_params": {"max_wind_speed_kmh": 120.0},
            "rainfall_params": {"rainfall_intensity_mm_hr": 60.0},
            "duration_hours": 24
        }
        r_insp = client.post("/api/scenarios/inspect-parameters", json=inspect_payload)
        self.assertEqual(r_insp.status_code, 200)
        idata = r_insp.json()
        self.assertTrue("effective_parameters" in idata or "parameters" in idata)
        print(">> Test 5 Passed: Scenario inspect-parameters endpoint functional")


if __name__ == "__main__":
    unittest.main()
