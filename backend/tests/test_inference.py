import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inference import (feature_names, get_bundle, get_schema, predict_proba_frame,
                       predict_record, validate_record)

BACKEND_DIR = Path(__file__).resolve().parents[1]
EXAMPLES = json.loads((BACKEND_DIR / "examples.json").read_text(encoding="utf-8"))


def test_schema_matches_bundle():
    bundle = get_bundle()
    schema = get_schema()
    assert [f["name"] for f in schema["features"]] == list(bundle["kept_features"])
    assert schema["n_features"] == len(bundle["kept_features"]) == 34
    for spec in schema["features"]:
        if spec["kind"] == "category":
            assert spec["options"] == list(
                bundle["encoders"][spec["name"]].classes_)


def test_feature_counts():
    kinds = [f["kind"] for f in get_schema()["features"]]
    assert kinds.count("category") == 2
    assert len(kinds) - kinds.count("category") == 32


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_examples_reproduce_stored_labels(example):
    result = predict_record(example["record"])
    assert result["label"] == example["model_label"]
    assert abs(result["probabilities"][1] - example["p_attack"]) < 1e-3


def _reference_proba(bundle, raw):
    """Independent re-implementation of export_model.py's test-time steps,
    using all 40 real scaler columns (nothing filled with zeros)."""
    frame = raw.copy()
    for column in bundle["categorical_columns"]:
        frame[column] = bundle["encoders"][column].transform(frame[column].astype(str))
    frame = frame.drop(columns=bundle["drop_always"])
    names = list(bundle["feature_names"])
    scaled = pd.DataFrame(bundle["scaler"].transform(frame[names]), columns=names)
    features = scaled[list(bundle["kept_features"])].to_numpy(dtype=np.float32)
    return bundle["model"].predict_proba(features)[:, 1]


@pytest.mark.slow
def test_validation_accepts_every_real_test_record(kddtest_frame):
    """The rules must never reject genuine NSL-KDD traffic."""
    kept = feature_names()
    rejected = 0
    for record in kddtest_frame[kept].to_dict(orient="records"):
        try:
            validate_record(record)
        except ValueError:
            rejected += 1
    assert rejected == 0


@pytest.mark.slow
def test_adapter_reproduces_official_confusion_matrix(kddtest_frame):
    """The API's code path must give the same probabilities as the training
    pipeline and the same TN/FP/FN/TP as model/metrics.json."""
    bundle = get_bundle()
    kept = feature_names()

    adapter = predict_proba_frame(kddtest_frame[kept])
    reference = _reference_proba(bundle, kddtest_frame)
    assert np.allclose(adapter, reference, atol=1e-9)

    truth = (kddtest_frame["class"].str.lower() != "normal").to_numpy().astype(int)
    pred = (adapter >= 0.5).astype(int)
    found = {
        "tn": int(((pred == 0) & (truth == 0)).sum()),
        "fp": int(((pred == 1) & (truth == 0)).sum()),
        "fn": int(((pred == 0) & (truth == 1)).sum()),
        "tp": int(((pred == 1) & (truth == 1)).sum()),
    }
    assert found == bundle["metrics"]["confusion_matrix"]
    assert found == {"tn": 9432, "fp": 279, "fn": 4601, "tp": 8232}
