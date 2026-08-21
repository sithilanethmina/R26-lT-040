from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]          # gpu-price-model
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]        # repo root
SRC_DIR = PROJECT_ROOT / "src"
APP_PY = SRC_DIR / "gpu_price_predictor" / "app.py"
REQ_FILE = PROJECT_ROOT / "requirements.txt"

# Critical dependencies required by the GPU app
REQUIRED_PACKAGES = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("optuna", "optuna"),
    ("rapidfuzz", "rapidfuzz"),
    ("plotly", "plotly"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]


def find_venv_python() -> Path | None:
    """Find virtual environment python in workspace or project root."""
    candidate_paths = [
        WORKSPACE_ROOT / "venv" / "Scripts" / "python.exe",     # Windows repo root
        WORKSPACE_ROOT / "venv" / "bin" / "python",             # Linux/macOS repo root
        WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe",
        WORKSPACE_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT / "venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / "venv" / "bin" / "python",
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    return None


def is_running_in_target_venv(target_py: Path) -> bool:
    """Check if current execution interpreter matches target venv python."""
    try:
        return Path(sys.executable).resolve() == target_py.resolve()
    except Exception:
        return False


def check_and_install_dependencies(py_executable: str | Path):
    """Verify that all required packages are importable; prompt/auto-install if missing."""
    print("Checking dependencies...")
    missing = []
    
    for import_name, pkg_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        print(f"\n[!] Missing packages detected: {', '.join(missing)}")
        print("Installing missing dependencies now...")
        
        if REQ_FILE.exists():
            cmd = [str(py_executable), "-m", "pip", "install", "-r", str(REQ_FILE)]
        else:
            cmd = [str(py_executable), "-m", "pip", "install"] + missing
            
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[ERROR] Failed to install dependencies. Please run:\n  pip install -r {REQ_FILE}")
            sys.exit(1)
        print("[+] Dependencies installed successfully.\n")
    else:
        print("[+] All required packages are installed.")


def main():
    venv_py = find_venv_python()

    # If a venv exists and we're not currently running inside it, re-launch using venv python
    if venv_py and not is_running_in_target_venv(venv_py):
        print(f"[i] Switching to virtual environment python: {venv_py}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        # Set PYTHONPATH so gpu_price_predictor package is always found
        python_paths = [str(SRC_DIR), str(PROJECT_ROOT), str(WORKSPACE_ROOT)]
        if "PYTHONPATH" in env:
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)

        # Re-execute this script inside the venv interpreter
        cmd = [str(venv_py), str(__file__)] + sys.argv[1:]
        sys.exit(subprocess.call(cmd, env=env))

    # Inside target environment:
    current_py = sys.executable
    check_and_install_dependencies(current_py)

    # Ensure src is in sys.path
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    # Launch Streamlit app
    print(f"\n[+] Launching Streamlit GPU Price Predictor ({APP_PY.name})...\n")
    cmd = [
        str(current_py),
        "-m",
        "streamlit",
        "run",
        str(APP_PY),
        "--server.headless=false",
    ] + sys.argv[1:]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    python_paths = [str(SRC_DIR), str(PROJECT_ROOT), str(WORKSPACE_ROOT)]
    if "PYTHONPATH" in env:
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()

