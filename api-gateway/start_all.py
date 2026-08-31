import subprocess
import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def find_venv_python() -> Path:
    """Find virtual environment python in workspace."""
    candidate_paths = [
        PROJECT_ROOT / "venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / "venv" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    return Path(sys.executable)

VENV_PYTHON = str(find_venv_python())

REQUIRED_MODULES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("flask", "flask"),
    ("flask_cors", "flask-cors"),
    ("xgboost", "xgboost"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("requests", "requests"),
]

def check_dependencies():
    """Verify that backend microservice dependencies are present."""
    missing = []
    for mod_name, pkg_name in REQUIRED_MODULES:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pkg_name)
    if missing:
        print(f"[!] Warning: Missing dependencies detected: {', '.join(missing)}")
        print(f"    Installing via pip using: {VENV_PYTHON} ...")
        subprocess.run([VENV_PYTHON, "-m", "pip", "install"] + missing)
        print("[+] Dependencies verified.\n")
    else:
        print("[+] All microservice dependencies verified.")

SERVICES = [
    {
        "name": "GPU Microservice",
        "cwd": PROJECT_ROOT / "gpu-price-model" / "src",
        "cmd": [VENV_PYTHON, "-m", "uvicorn", "gpu_price_predictor.api:app", "--host", "0.0.0.0", "--port", "8001"]
    },
    {
        "name": "Mobile Microservice",
        "cwd": PROJECT_ROOT / "mobile-price-model",
        "cmd": [VENV_PYTHON, "api.py"]
    },
    {
        "name": "Vehicle Microservice",
        "cwd": PROJECT_ROOT / "vehicle-price-model",
        "cmd": [VENV_PYTHON, "app.py"]
    },
    {
        "name": "Electronics Microservice",
        "cwd": PROJECT_ROOT / "electronics-price-model",
        "cmd": [VENV_PYTHON, "app.py"]
    },
    {
        "name": "API Gateway",
        "cwd": PROJECT_ROOT / "api-gateway",
        "cmd": [VENV_PYTHON, "-m", "uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", os.environ.get("PORT", "8000")]
    }
]

processes = []

def start_services():
    check_dependencies()
    print("\nStarting all FairPriceLK backend microservices...")
    for svc in SERVICES:
        # Check if the file to execute exists (for the ones like api.py/app.py)
        if "api.py" in svc["cmd"] or "app.py" in svc["cmd"]:
            file_to_run = svc["cwd"] / svc["cmd"][-1]
            if not file_to_run.exists():
                print(f"Skipping {svc['name']} - {file_to_run} not found.")
                continue
                
        print(f"Starting {svc['name']}...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        python_paths = [
            str(PROJECT_ROOT),
            str(PROJECT_ROOT / "api-gateway"),
            str(PROJECT_ROOT / "gpu-price-model" / "src"),
            str(PROJECT_ROOT / "mobile-price-model"),
            str(PROJECT_ROOT / "vehicle-price-model"),
            str(PROJECT_ROOT / "electronics-price-model")
        ]
        if "PYTHONPATH" in env:
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        
        p = subprocess.Popen(
            svc["cmd"], 
            cwd=str(svc["cwd"]),
            env=env
        )
        processes.append((svc['name'], p))
        time.sleep(1) # stagger startup

    print("\nAll 4 microservices + API Gateway are running on http://127.0.0.1:8000. Press Ctrl+C to stop.")

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
