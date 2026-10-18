"""Direct inference check, without Flask (manual Stage 5).

Loads the saved model and runs the sample payloads through the same code the
API uses. Run from the repository root:  python backend/smoke_pipeline.py

This file is named smoke_*.py (not test_*.py) so pytest does not collect it.
"""
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from inference import feature_names, predict_record  # noqa: E402

features = feature_names()
checks = [("test_payload_normal.json", 0), ("test_payload.json", 1)]

for filename, expected_label in checks:
    payload = json.loads((BACKEND_DIR / filename).read_text(encoding="utf-8"))
    assert set(payload) == set(features), f"{filename}: keys must match features"
    result = predict_record(payload)
    print(f"{filename}: prediction={result['prediction']} "
          f"P(attack)={result['probabilities'][1]:.4f} "
          f"warnings={len(result['warnings'])}")
    assert result["label"] == expected_label, (
        f"{filename}: expected label {expected_label}, got {result['label']}")

print("PASS: local inference")
