"""Model access for the API: load the existing bundle, validate, predict.

The model bundle (model/nslkdd_model.pkl) is a dictionary, not a scikit-learn
Pipeline, so the preprocessing is repeated here exactly as in export_model.py
and analyze_attack_types.py:

  1. encode protocol_type and service with the bundle's label encoders
  2. fill the VIF-dropped columns with 0 (they are discarded after scaling and
     RobustScaler works column by column, so they cannot affect the kept ones)
  3. scale all 40 scaler columns with the bundle's RobustScaler
  4. keep the 34 model features in the bundle's order
  5. predict_proba; label 1 (attack) if P(attack) >= 0.5

The bundle is a trusted pickle created by this project. Never load pickles from
untrusted sources.
"""
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
MODEL_PATH = Path(os.environ.get("NSLKDD_MODEL_PATH",
                                 REPO_ROOT / "model" / "nslkdd_model.pkl"))
SCHEMA_PATH = BACKEND_DIR / "schema.json"
THRESHOLD = 0.5
CLASS_NAMES = ["normal", "attack"]


class ValidationError(ValueError):
    """The request does not match the input schema."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@lru_cache(maxsize=1)
def get_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_PATH}. Set NSLKDD_MODEL_PATH or "
            "run export_model.py.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def feature_names():
    return list(get_bundle()["kept_features"])


def validate_record(record):
    """Check one request against the schema.

    Returns a list of warnings (values outside the training range). Raises
    ValidationError with every problem found if the record is invalid.
    """
    if not isinstance(record, dict):
        raise ValidationError(["Expected a JSON object"])

    fields = {f["name"]: f for f in get_schema()["features"]}
    problems, warnings = [], []

    missing = [n for n in fields if n not in record]
    extra = [k for k in record if k not in fields]
    if missing:
        problems.append(f"missing features: {missing}")
    if extra:
        problems.append(f"unexpected features: {extra}")

    for name, spec in fields.items():
        if name not in record:
            continue
        value = record[name]

        if spec["kind"] == "category":
            if not isinstance(value, str) or value not in spec["options"]:
                problems.append(
                    f"{name}: must be one of {len(spec['options'])} allowed "
                    f"values, for example {spec['options'][:3]}")
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{name}: must be a number")
            continue
        if not math.isfinite(value):
            problems.append(f"{name}: must be a finite number")
            continue
        if spec["kind"] == "binary" and value not in (0, 1):
            problems.append(f"{name}: must be 0 or 1")
            continue
        if value < spec["min_allowed"]:
            problems.append(f"{name}: must be at least {spec['min_allowed']}")
            continue
        if "max_allowed" in spec and value > spec["max_allowed"]:
            problems.append(f"{name}: must be at most {spec['max_allowed']}")
            continue

        if value < spec["train_min"] or value > spec["train_max"]:
            warnings.append(
                f"{name}: {value:g} is outside the range seen in training "
                f"({spec['train_min']:g} to {spec['train_max']:g})")

    if problems:
        raise ValidationError(problems)
    return warnings


def predict_proba_frame(frame):
    """P(attack) for each row of a DataFrame holding the 34 model features.

    Categorical columns hold the raw strings (for example "tcp", "http").
    """
    bundle = get_bundle()
    kept = list(bundle["kept_features"])
    scaler_columns = list(bundle["feature_names"])
    categorical = set(bundle["categorical_columns"])

    full = pd.DataFrame(0.0, index=frame.index, columns=scaler_columns)
    for column in kept:
        if column in categorical:
            full[column] = bundle["encoders"][column].transform(
                frame[column].astype(str))
        else:
            full[column] = frame[column].astype(float).to_numpy()

    scaled = pd.DataFrame(bundle["scaler"].transform(full),
                          columns=scaler_columns, index=frame.index)
    features = scaled[kept].to_numpy(dtype=np.float32)
    return bundle["model"].predict_proba(features)[:, 1]


def predict_record(record):
    """Validate one record and return the prediction as a plain dictionary."""
    warnings = validate_record(record)
    kept = feature_names()
    frame = pd.DataFrame([[record[name] for name in kept]], columns=kept)
    p_attack = float(predict_proba_frame(frame)[0])
    label = int(p_attack >= THRESHOLD)
    return {
        "label": label,
        "prediction": CLASS_NAMES[label],
        "probabilities": [1.0 - p_attack, p_attack],
        "threshold": THRESHOLD,
        "warnings": warnings,
    }
