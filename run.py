from __future__ import annotations

import sys
from pathlib import Path

# Add api-gateway directory to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "api-gateway"))

from start_all import start_services, processes
import time

if __name__ == "__main__":
    try:
        start_services()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        for name, p in processes:
            print(f"Terminating {name}...")
            p.terminate()
        sys.exit(0)
