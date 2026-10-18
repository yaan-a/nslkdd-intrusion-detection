#!/usr/bin/env python3
"""Generate backend/schema.json from the model bundle and the NSL-KDD data.

The schema lists the 34 inputs the model actually uses (bundle["kept_features"],
in order), with type, allowed values, ranges and a short description. The API
validates requests against it, and the Angular form is built from it, so the two
cannot drift apart.

Run from the repository root:  python backend/make_schema.py
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

GROUP_BASIC = "Basic connection"
GROUP_CONTENT = "Content"
GROUP_TIME = "Time-window traffic (past 2 seconds)"
GROUP_HOST = "Host-based traffic (past 100 connections)"
GROUP_ORDER = [GROUP_BASIC, GROUP_CONTENT, GROUP_TIME, GROUP_HOST]

# Feature groups follow the KDD'99 feature documentation.
GROUPS = {
    "duration": GROUP_BASIC, "protocol_type": GROUP_BASIC,
    "service": GROUP_BASIC, "src_bytes": GROUP_BASIC,
    "dst_bytes": GROUP_BASIC, "land": GROUP_BASIC,
    "wrong_fragment": GROUP_BASIC, "urgent": GROUP_BASIC,
    "hot": GROUP_CONTENT, "num_failed_logins": GROUP_CONTENT,
    "logged_in": GROUP_CONTENT, "num_compromised": GROUP_CONTENT,
    "root_shell": GROUP_CONTENT, "su_attempted": GROUP_CONTENT,
    "num_file_creations": GROUP_CONTENT, "num_shells": GROUP_CONTENT,
    "num_access_files": GROUP_CONTENT, "is_host_login": GROUP_CONTENT,
    "is_guest_login": GROUP_CONTENT,
    "count": GROUP_TIME, "srv_count": GROUP_TIME,
    "serror_rate": GROUP_TIME, "rerror_rate": GROUP_TIME,
    "same_srv_rate": GROUP_TIME, "diff_srv_rate": GROUP_TIME,
    "srv_diff_host_rate": GROUP_TIME,
    "dst_host_count": GROUP_HOST, "dst_host_srv_count": GROUP_HOST,
    "dst_host_diff_srv_rate": GROUP_HOST,
    "dst_host_same_src_port_rate": GROUP_HOST,
    "dst_host_srv_diff_host_rate": GROUP_HOST,
    "dst_host_serror_rate": GROUP_HOST, "dst_host_rerror_rate": GROUP_HOST,
    "dst_host_srv_rerror_rate": GROUP_HOST,
}

# Short descriptions, paraphrased from the KDD'99 feature documentation.
DESCRIPTIONS = {
    "duration": "Length of the connection in seconds.",
    "protocol_type": "Transport protocol of the connection.",
    "service": "Network service on the destination (for example http, ftp).",
    "src_bytes": "Bytes sent from source to destination.",
    "dst_bytes": "Bytes sent from destination to source.",
    "land": "1 if the connection is from and to the same host and port.",
    "wrong_fragment": "Number of wrong fragments.",
    "urgent": "Number of urgent packets.",
    "hot": "Number of 'hot' indicators (access to system directories, program execution).",
    "num_failed_logins": "Number of failed login attempts.",
    "logged_in": "1 if the login succeeded.",
    "num_compromised": "Number of compromised conditions.",
    "root_shell": "1 if a root shell was obtained.",
    "su_attempted": "Set if a 'su root' command was attempted.",
    "num_file_creations": "Number of file creation operations.",
    "num_shells": "Number of shell prompts.",
    "num_access_files": "Number of operations on access control files.",
    "is_host_login": "1 if the login belongs to the 'hot' list (root or admin).",
    "is_guest_login": "1 if the login is a guest login.",
    "count": "Connections to the same host as this one in the past 2 seconds.",
    "srv_count": "Connections to the same service as this one in the past 2 seconds.",
    "serror_rate": "Fraction of those connections with SYN errors.",
    "rerror_rate": "Fraction of those connections with REJ errors.",
    "same_srv_rate": "Fraction of those connections to the same service.",
    "diff_srv_rate": "Fraction of those connections to different services.",
    "srv_diff_host_rate": "Fraction of same-service connections going to different hosts.",
    "dst_host_count": "Connections to the same destination host (past 100 connections).",
    "dst_host_srv_count": "Connections to the same destination host and service (past 100).",
    "dst_host_diff_srv_rate": "Fraction of connections to the destination host using different services.",
    "dst_host_same_src_port_rate": "Fraction of connections to the destination host from the same source port.",
    "dst_host_srv_diff_host_rate": "Fraction of same-service connections going to different hosts.",
    "dst_host_serror_rate": "Fraction of connections to the destination host with SYN errors.",
    "dst_host_rerror_rate": "Fraction of connections to the destination host with REJ errors.",
    "dst_host_srv_rerror_rate": "Fraction of same-service connections to the destination host with REJ errors.",
}

# Flags documented as 0/1 and observed as exactly {0, 1} in the data.
# su_attempted is documented as binary but the data also contains 2, so it is
# treated as a plain number.
KNOWN_BINARY = {"land", "logged_in", "root_shell", "is_host_login",
                "is_guest_login"}

LABEL_WORDS = {"src": "source", "dst": "destination", "srv": "service",
               "num": "number of", "serror": "SYN-error", "rerror": "REJ-error",
               "diff": "different", "srv_": "service"}


def make_label(name):
    words = [LABEL_WORDS.get(w, w) for w in name.split("_")]
    text = " ".join(words)
    return text[0].upper() + text[1:]


def load_official(train_path, test_path, merged_path):
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)
    if merged_path.exists():
        merged = pd.read_csv(merged_path)
        if "source_file" in merged.columns:
            print(f"NOTE: using {merged_path} split by its 'source_file' column.")
            return (merged[merged["source_file"] == "train"],
                    merged[merged["source_file"] == "test"])
    sys.exit("ERROR: official data not found. Provide data/KDDTrain+.csv and "
             "data/KDDTest+.csv (or the merged file with a source_file column).")


def clean(value):
    """Round floats and convert numpy scalars to plain Python numbers."""
    value = float(value)
    value = round(value, 6)
    return int(value) if value.is_integer() else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(REPO_ROOT / "model/nslkdd_model.pkl"))
    parser.add_argument("--train", default=str(REPO_ROOT / "data/KDDTrain+.csv"))
    parser.add_argument("--test", default=str(REPO_ROOT / "data/KDDTest+.csv"))
    parser.add_argument("--merged-csv",
                        default=str(REPO_ROOT / "data/NSL-KDD_merged.csv"))
    parser.add_argument("--out", default=str(REPO_ROOT / "backend/schema.json"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    kept = list(bundle["kept_features"])
    categorical = set(bundle["categorical_columns"])
    train, test = load_official(Path(args.train), Path(args.test),
                                Path(args.merged_csv))
    both = pd.concat([train, test])

    missing_meta = [f for f in kept if f not in GROUPS or f not in DESCRIPTIONS]
    if missing_meta:
        sys.exit(f"ERROR: no group/description for: {missing_meta}")

    features = []
    problems = []
    for name in kept:
        entry = {"name": name, "label": make_label(name),
                 "group": GROUPS[name], "description": DESCRIPTIONS[name],
                 "required": True}
        if name in categorical:
            entry["kind"] = "category"
            entry["options"] = [str(c) for c in bundle["encoders"][name].classes_]
            unseen = set(both[name].astype(str)) - set(entry["options"])
            if unseen:
                problems.append(f"{name}: data values not in encoder: {sorted(unseen)}")
        else:
            observed = both[name].astype(float)
            tr = train[name].astype(float)
            integer = bool(np.all(np.mod(observed, 1) == 0))
            if name.endswith("_rate"):
                kind = "rate"
                if observed.min() < 0 or observed.max() > 1:
                    problems.append(f"{name}: rate outside [0, 1] in the data")
            elif name in KNOWN_BINARY:
                kind = "binary"
                if not set(np.unique(observed)) <= {0.0, 1.0}:
                    problems.append(f"{name}: not binary in the data")
            else:
                kind = "number"
            if observed.min() < 0:
                problems.append(f"{name}: negative values in the data")
            entry.update({
                "kind": kind,
                "integer": integer,
                "min_allowed": 0,
                "train_min": clean(tr.min()),
                "train_max": clean(tr.max()),
                "median": clean(tr.median()),
                "p01": clean(tr.quantile(0.01)),
                "p99": clean(tr.quantile(0.99)),
            })
            if kind in ("rate", "binary"):
                entry["max_allowed"] = 1
        features.append(entry)

    if problems:
        sys.exit("ERROR: data does not match the validation assumptions:\n  "
                 + "\n  ".join(problems))

    schema = {
        "schema_version": 1,
        "model": bundle.get("model_name", "model"),
        "protocol": bundle.get("protocol", "official"),
        "threshold": 0.5,
        "n_features": len(features),
        "dropped_features": list(bundle["dropped_features"]),
        "groups": GROUP_ORDER,
        "features": features,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    kinds = pd.Series([f["kind"] for f in features]).value_counts().to_dict()
    print(f"Wrote {out} with {len(features)} features: {kinds}")
    print("Hard rules: required keys, category membership, numbers >= 0, "
          "rates in [0, 1], binary flags in {0, 1}.")
    print("Soft rule: values outside the training min/max produce a warning.")


if __name__ == "__main__":
    main()
