import json
from pathlib import Path

import pytest

from app import FEATURES, app

BACKEND_DIR = Path(__file__).resolve().parents[1]
ATTACK = json.loads((BACKEND_DIR / "test_payload.json").read_text(encoding="utf-8"))
NORMAL = json.loads((BACKEND_DIR / "test_payload_normal.json").read_text(encoding="utf-8"))
EXAMPLES = json.loads((BACKEND_DIR / "examples.json").read_text(encoding="utf-8"))


@pytest.fixture()
def client():
    return app.test_client()


# --- the four tests required by the lab manual -------------------------------

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["features"] == len(FEATURES) == 34


def test_predict_valid(client):
    response = client.post("/predict", json=ATTACK)
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["prediction"] in ("normal", "attack")
    assert len(body["probabilities"]) == 2
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-9


def test_predict_missing_feature(client):
    incomplete = dict(ATTACK)
    incomplete.pop(FEATURES[0])
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 400
    assert FEATURES[0] in response.get_json()["problems"][0]


def test_predict_invalid_json(client):
    response = client.post("/predict", data="not-json",
                           content_type="application/json")
    assert response.status_code == 400


# --- behaviour on real records ------------------------------------------------

def test_normal_and_attack_payloads(client):
    normal = client.post("/predict", json=NORMAL).get_json()
    attack = client.post("/predict", json=ATTACK).get_json()
    assert normal["label"] == 0 and normal["prediction"] == "normal"
    assert attack["label"] == 1 and attack["prediction"] == "attack"


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_examples_match_stored_model_output(client, example):
    body = client.post("/predict", json=example["record"]).get_json()
    assert body["label"] == example["model_label"]
    assert abs(body["probabilities"][1] - example["p_attack"]) < 1e-3


# --- input validation ---------------------------------------------------------

def _with(field, value):
    payload = dict(ATTACK)
    payload[field] = value
    return payload


def test_extra_feature_rejected(client):
    payload = dict(ATTACK, surprise=1)
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "surprise" in response.get_json()["problems"][0]


def test_unknown_category_rejected(client):
    response = client.post("/predict", json=_with("service", "not_a_service"))
    assert response.status_code == 400


def test_wrong_type_rejected(client):
    assert client.post("/predict", json=_with("duration", "ten")).status_code == 400
    assert client.post("/predict", json=_with("duration", True)).status_code == 400
    assert client.post("/predict", json=_with("protocol_type", 5)).status_code == 400


def test_rate_above_one_rejected(client):
    assert client.post("/predict", json=_with("serror_rate", 1.5)).status_code == 400


def test_negative_count_rejected(client):
    assert client.post("/predict", json=_with("src_bytes", -1)).status_code == 400


def test_binary_flag_must_be_zero_or_one(client):
    assert client.post("/predict", json=_with("logged_in", 2)).status_code == 400


def test_nan_rejected(client):
    body = json.dumps(_with("duration", float("nan")))  # serialises as NaN
    response = client.post("/predict", data=body, content_type="application/json")
    assert response.status_code == 400


def test_json_array_rejected(client):
    response = client.post("/predict", json=[1, 2, 3])
    assert response.status_code == 400


def test_out_of_training_range_warns_but_succeeds(client):
    response = client.post("/predict", json=_with("src_bytes", 5_000_000_000))
    assert response.status_code == 200
    assert any("src_bytes" in w for w in response.get_json()["warnings"])


# --- other routes -------------------------------------------------------------

def test_schema_endpoint(client):
    body = client.get("/schema").get_json()
    assert [f["name"] for f in body["features"]] == FEATURES


def test_get_on_predict_is_405(client):
    response = client.get("/predict")
    assert response.status_code == 405
    assert response.get_json()["error"] == "Method not allowed"


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
