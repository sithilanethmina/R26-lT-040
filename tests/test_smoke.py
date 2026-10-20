"""
FairPriceLK Consolidated Smoke Test Suite
Verifies API Gateway health, in-memory downstream microservices,
metadata endpoints, and live ML inference across all 4 categories.
Can be run via:
    python tests/test_smoke.py
or:
    pytest tests/test_smoke.py
"""

import sys
import os
import unittest
from pathlib import Path

# Ensure project root and api-gateway are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GATEWAY_DIR = PROJECT_ROOT / "api-gateway"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(GATEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(GATEWAY_DIR))

# Force CONSOLIDATED_MODE for CI/CD smoke test
os.environ["CONSOLIDATED_MODE"] = "true"

from starlette.testclient import TestClient
from gateway import app

client = TestClient(app)


class TestFairPriceLKSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 60)
        print("Starting FairPriceLK Consolidated Smoke Tests")
        print("=" * 60)

    def test_01_gateway_health(self):
        """Gateway /api/health returns HTTP 200 and healthy status."""
        resp = client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("services", data)
        print("  [PASS] test_01_gateway_health")

    def test_02_developer_health(self):
        """Developer health inspects all 4 microservices."""
        resp = client.get("/api/developer/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        services = {s["id"]: s["status"] for s in data.get("services", [])}
        for svc in ["gpu", "mobile", "vehicle", "electronics"]:
            self.assertIn(svc, services, f"Service {svc} missing in developer health")
            self.assertEqual(services[svc], "healthy", f"Service {svc} is not healthy: {services[svc]}")
        print("  [PASS] test_02_developer_health (all 4 services healthy)")

    def test_03_developer_models(self):
        """Developer models endpoint lists ML artifact inventory."""
        resp = client.get("/api/developer/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        models = data.get("models", data) if isinstance(data, dict) else data
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 4)
        print("  [PASS] test_03_developer_models")

    def test_04_metadata_endpoints(self):
        """Category metadata endpoints return valid model catalogs."""
        # GPU metadata
        r_gpu = client.get("/api/gpu/metadata")
        self.assertEqual(r_gpu.status_code, 200)
        
        # Vehicle metadata
        r_veh = client.get("/api/vehicle/metadata")
        self.assertEqual(r_veh.status_code, 200)
        
        # Mobile metadata
        r_mob = client.get("/api/mobile/metadata")
        self.assertEqual(r_mob.status_code, 200)
        
        # Electronics metadata
        r_elec = client.get("/api/electronics/metadata")
        self.assertEqual(r_elec.status_code, 200)
        print("  [PASS] test_04_metadata_endpoints (gpu, vehicle, mobile, electronics)")

    def test_05_gpu_prediction(self):
        """GPU Price Prediction returns valid price estimation."""
        payload = {
            "model": "RTX 3060",
            "vram_gb": 12.0,
            "brand": "MSI",
            "listed_price": 95000
        }
        resp = client.post("/api/gpu/predict", json=payload)
        self.assertEqual(resp.status_code, 200, f"GPU predict failed: {resp.text}")
        data = resp.json()
        pred_price = data.get("predicted_price")
        self.assertIsNotNone(pred_price)
        self.assertGreater(pred_price, 0)
        print(f"  [PASS] test_05_gpu_prediction (predicted: Rs. {pred_price:,.2f})")

    def test_06_mobile_prediction(self):
        """Mobile Price Prediction returns valid valuation and range."""
        payload = {
            "brand": "Apple",
            "model": "iPhone 13",
            "storage_gb": 128.0
        }
        resp = client.post("/api/mobile/predict", json=payload)
        self.assertEqual(resp.status_code, 200, f"Mobile predict failed: {resp.text}")
        data = resp.json()
        pred_price = data.get("predicted_price")
        self.assertIsNotNone(pred_price)
        self.assertGreater(pred_price, 0)
        print(f"  [PASS] test_06_mobile_prediction (predicted: Rs. {pred_price:,.2f})")

    def test_07_vehicle_prediction(self):
        """Vehicle Price Prediction returns valid valuation for cars and SUVs."""
        # Car prediction
        car_payload = {
            "brand": "Toyota",
            "model": "Axio",
            "model_year": 2018,
            "fuel_type": "Petrol",
            "transmission": "Automatic"
        }
        resp_car = client.post("/api/vehicle/predict", json=car_payload)
        self.assertEqual(resp_car.status_code, 200, f"Car predict failed: {resp_car.text}")
        car_price = resp_car.json().get("predicted_price")
        self.assertGreater(car_price, 0)

        # SUV prediction
        suv_payload = {
            "brand": "Toyota",
            "model": "Prado",
            "model_year": 2017,
            "engine_cc": 2800,
            "fuel_type": "Diesel",
            "transmission": "Automatic"
        }
        resp_suv = client.post("/api/vehicle/predict/suv", json=suv_payload)
        self.assertEqual(resp_suv.status_code, 200, f"SUV predict failed: {resp_suv.text}")
        suv_price = resp_suv.json().get("predicted_price")
        self.assertGreater(suv_price, 0)
        print(f"  [PASS] test_07_vehicle_prediction (Car: Rs. {car_price:,.2f}, SUV: Rs. {suv_price:,.2f})")

    def test_08_electronics_prediction(self):
        """Electronics Price Prediction returns valid estimation for laptops."""
        laptop_payload = {
            "category": "laptop",
            "brand": "Dell",
            "model": "Latitude 5420",
            "cpu": "Intel Core i5",
            "generation": 11,
            "ram": 16,
            "storage": 512,
            "storage_type": "SSD",
            "gpu": "Integrated"
        }
        resp = client.post("/api/electronics/predict", json=laptop_payload)
        self.assertEqual(resp.status_code, 200, f"Electronics predict failed: {resp.text}")
        data = resp.json()
        pred_price = data.get("predicted_price") or data.get("price")
        self.assertIsNotNone(pred_price)
        self.assertGreater(pred_price, 0)
        print(f"  [PASS] test_08_electronics_prediction (predicted: Rs. {pred_price:,.2f})")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFairPriceLKSmoke)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
