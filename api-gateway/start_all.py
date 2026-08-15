import subprocess
import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SERVICES = [
    {
        "name": "GPU Microservice",
        "cwd": PROJECT_ROOT / "gpu-price-model",
        "cmd": [sys.executable, "-m", "gpu_price_predictor.api"]
    },
    {
        "name": "Mobile Microservice",
        "cwd": PROJECT_ROOT / "mobile-price-model",
        "cmd": [sys.executable, "api.py"]
    },
    {
        "name": "Vehicle Microservice",
        "cwd": PROJECT_ROOT / "vehicle-price-model",
        "cmd": [sys.executable, "app.py"]
    },
    {
        "name": "Electronics Microservice",
        "cwd": PROJECT_ROOT / "electronics-price-model",
        "cmd": [sys.executable, "app.py"]
    },
    {
        "name": "API Gateway",
        "cwd": PROJECT_ROOT / "api-gateway",
        "cmd": [sys.executable, "-m", "uvicorn", "gateway:app", "--port", "8000"]
    }
]

processes = []

def start_services():
    print("Starting all FairPriceLK services...")
    for svc in SERVICES:
        # Check if the file to execute exists (for the ones like api.py/app.py)
        # to avoid crashes if they aren't fully implemented yet
        if "api.py" in svc["cmd"] or "app.py" in svc["cmd"]:
            file_to_run = svc["cwd"] / svc["cmd"][-1]
            if not file_to_run.exists():
                print(f"Skipping {svc['name']} - {file_to_run} not found.")
                continue
                
        print(f"Starting {svc['name']}...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "gpu-price-model" / "src")
        
        p = subprocess.Popen(
            svc["cmd"], 
            cwd=str(svc["cwd"]),
            env=env
        )
        processes.append((svc['name'], p))
        time.sleep(1) # stagger startup

    print("\nAll services started. Press Ctrl+C to stop.")

if __name__ == "__main__":
    try:
        start_services()
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        for name, p in processes:
            print(f"Terminating {name}...")
            p.terminate()
        sys.exit(0)
