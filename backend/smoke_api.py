"""Send sample requests to the running Flask server (manual Stage 7).

Start the server first in another terminal:  python backend/app.py
Then run, from the repository root:          python backend/smoke_api.py

This file is named smoke_*.py (not test_*.py) because it needs a running server
and must not be collected by pytest.
"""
import json
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:5000"

with requests.Session() as session:
    session.trust_env = False  # do not route localhost through a proxy

    health = session.get(f"{URL}/health", timeout=30)
    print("GET /health ->", health.status_code, health.text.strip())
    health.raise_for_status()

    for filename in ("test_payload_normal.json", "test_payload.json"):
        payload = json.loads((BASE / filename).read_text(encoding="utf-8"))
        response = session.post(f"{URL}/predict", json=payload, timeout=60)
        print(f"POST /predict ({filename}) -> {response.status_code}")
        print("  ", response.text.strip().replace("\n", " "))
        response.raise_for_status()

print("PASS: API request")
