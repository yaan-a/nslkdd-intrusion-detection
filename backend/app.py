"""Flask API for the NSL-KDD intrusion detection model.

Endpoints
  GET  /health   status and model information
  POST /predict  JSON object with the 34 model features -> prediction

Run from the repository root:  python backend/app.py
"""
from flask import Flask, jsonify, request

from inference import (THRESHOLD, ValidationError, get_bundle,
                       get_schema, predict_record)

BUNDLE = get_bundle()
FEATURES = list(BUNDLE["kept_features"])
TASK = "classification"
MODEL_NAME = (f"{BUNDLE.get('model_name', 'model')} "
              f"({BUNDLE.get('protocol', 'unknown')} protocol)")
DISCLAIMER = ("Research and teaching prototype only. Trained on the NSL-KDD "
              "benchmark; not validated for operational use.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # a valid request is under 2 KB


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK,
                   model=MODEL_NAME, threshold=THRESHOLD)


@app.get("/schema")
def schema():
    """Input schema, so a client can build its form from the server's rules."""
    return jsonify(get_schema())


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object",
                       problems=["Send a JSON object with Content-Type: "
                                 "application/json"]), 400
    try:
        result = predict_record(data)
    except ValidationError as exc:
        return jsonify(error="Input schema mismatch",
                       problems=exc.problems), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500

    result.update(task=TASK, model=MODEL_NAME, disclaimer=DISCLAIMER)
    return jsonify(result)


@app.errorhandler(404)
def not_found(_error):
    return jsonify(error="Not found"), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify(error="Method not allowed"), 405


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Request too large"), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
