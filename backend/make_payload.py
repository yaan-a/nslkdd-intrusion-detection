#!/usr/bin/env python3
"""Create sample requests from real KDDTest+ records.

Writes
  backend/test_payload.json         one attack record (neptune), for curl tests
  backend/test_payload_normal.json  one normal record
  backend/examples.json             five labelled examples, including attacks the
                                    model misses (used by tests and, later, by
                                    "load example" buttons in the web form)
  data/sample_inputs.csv            the first two records as a CSV (no label)

Records are public benchmark data, not personal data.
Run from the repository root:  python backend/make_payload.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from inference import feature_names, predict_proba_frame  # noqa: E402

# (id, title, true class in KDDTest+, model should say attack?)
WANTED = [
    ("normal", "Normal web traffic", "normal", False),
    ("neptune", "SYN flood (neptune), detected", "neptune", True),
    ("satan", "Port scan (satan), detected", "satan", True),
    ("guess_passwd", "Password guessing (guess_passwd), missed by the model",
     "guess_passwd", False),
    ("mailbomb", "Mail bomb (mailbomb), attack type not in training, missed",
     "mailbomb", False),
]


def load_test():
    path = REPO_ROOT / "data" / "KDDTest+.csv"
    if path.exists():
        return pd.read_csv(path)
    merged = REPO_ROOT / "data" / "NSL-KDD_merged.csv"
    if merged.exists():
        frame = pd.read_csv(merged)
        if "source_file" in frame.columns:
            return frame[frame["source_file"] == "test"].reset_index(drop=True)
    sys.exit("ERROR: data/KDDTest+.csv not found.")


def to_python(value):
    return value.item() if hasattr(value, "item") else value


def main():
    kept = feature_names()
    test = load_test()
    proba = predict_proba_frame(test[kept])
    test = test.assign(p_attack=proba)

    examples = []
    for ident, title, true_class, expect_attack in WANTED:
        rows = test[(test["class"] == true_class)
                    & ((test["p_attack"] >= 0.5) == expect_attack)]
        if rows.empty:
            sys.exit(f"ERROR: no {true_class} record with the expected outcome")
        row = rows.iloc[0]
        record = {name: to_python(row[name]) for name in kept}
        examples.append({
            "id": ident, "title": title, "true_class": true_class,
            "model_label": int(row["p_attack"] >= 0.5),
            "p_attack": round(float(row["p_attack"]), 6),
            "record": record,
        })

    by_id = {e["id"]: e for e in examples}
    (BACKEND_DIR / "test_payload.json").write_text(
        json.dumps(by_id["neptune"]["record"], indent=2), encoding="utf-8")
    (BACKEND_DIR / "test_payload_normal.json").write_text(
        json.dumps(by_id["normal"]["record"], indent=2), encoding="utf-8")
    (BACKEND_DIR / "examples.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8")

    sample = pd.DataFrame([by_id["normal"]["record"], by_id["neptune"]["record"]],
                          columns=kept)
    out_csv = REPO_ROOT / "data" / "sample_inputs.csv"
    sample.to_csv(out_csv, index=False)

    print(f"Wrote test_payload.json, test_payload_normal.json, examples.json "
          f"and {out_csv.name}")
    for e in examples:
        print(f"  {e['id']:13s} true={e['true_class']:13s} "
              f"model_label={e['model_label']}  P(attack)={e['p_attack']:.4f}")


if __name__ == "__main__":
    main()
