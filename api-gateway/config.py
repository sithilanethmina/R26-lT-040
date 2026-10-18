import os

CONSOLIDATED_MODE = os.getenv("CONSOLIDATED_MODE", "true").lower() in ("1", "true", "yes")

SERVICES = {
    "gpu": {
        "url": os.getenv("GPU_SERVICE_URL", "http://localhost:8001"),
        "name": "GPU Price Predictor"
    },
    "mobile": {
        "url": os.getenv("MOBILE_SERVICE_URL", "http://localhost:8002"),
        "name": "Mobile Price Predictor"
    },
    "vehicle": {
        "url": os.getenv("VEHICLE_SERVICE_URL", "http://localhost:8003"),
        "name": "Vehicle Price Predictor"
    },
    "electronics": {
        "url": os.getenv("ELECTRONICS_SERVICE_URL", "http://localhost:8004"),
        "name": "Electronics Price Predictor"
    },
}
