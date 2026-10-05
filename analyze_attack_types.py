#!/usr/bin/env python3
"""Per-attack-type recall of the official-protocol stacking model (Issue #1).

Protocol: OFFICIAL (model trained on KDDTrain+, evaluated on KDDTest+).
This is NOT the random 80:20 split of the merged file.

The script LOADS model/nslkdd_model.pkl (it does not retrain) and applies the
same preprocessing as export_model.py to KDDTest+:

  1. binary label = 0 if class.lower() == "normal" else 1
  2. encode protocol_type, service, flag with bundle["encoders"]
  3. drop num_outbound_cmds (bundle["drop_always"])
  4. scale bundle["feature_names"] with bundle["scaler"]
  5. keep bundle["kept_features"], in that order
  6. prediction = 1 if predict_proba(X)[:, 1] >= 0.5

Outputs:
  results/per_attack_recall.csv
  results/per_family_recall.csv

Run from the repository root:  python analyze_attack_types.py
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

THRESHOLD = 0.5

# Standard NSL-KDD attack-to-family mapping: the 22 training-set attack types
# plus the additional types that appear only in KDDTest+. Families follow the
# usual convention in the NSL-KDD literature. A few names are classified
# slightly differently across papers (for example "worm"); this table is the
# one used for every result in this repository. Any attack name that is not
# listed is reported as "unmapped" and is never guessed.
ATTACK_FAMILY = {
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L",
    "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L",
    "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}


def load_official_frames(train_path, test_path, merged_path):
    """Return (train, test) DataFrames for the official protocol.

    Preferred source: the official KDDTrain+.csv and KDDTest+.csv. If either is
    missing, fall back to the merged file, whose "source_file" column records
    which official file each row came from ("train" or "test").
    """
    if train_path.exists() and test_path.exists():
        print(f"Train file: {train_path}\nTest file:  {test_path}")
        return pd.read_csv(train_path), pd.read_csv(test_path)

    if merged_path.exists():
        merged = pd.read_csv(merged_path)
        if "source_file" in merged.columns:
            print(f"NOTE: {train_path.name} / {test_path.name} not found. "
                  f"Using {merged_path} split by its 'source_file' column.")
            train = merged[merged["source_file"] == "train"].reset_index(drop=True)
            test = merged[merged["source_file"] == "test"].reset_index(drop=True)
            if len(train) and len(test):
                return train, test

    sys.exit(
        "ERROR: could not find the official data. Provide "
        f"{train_path} and {test_path}, or a merged file with a "
        f"'source_file' column at {merged_path}."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", default="data/KDDTrain+.csv")
    parser.add_argument("--test", default="data/KDDTest+.csv")
    parser.add_argument("--merged-csv", default="data/NSL-KDD_merged.csv",
                        help="fallback if the official files are missing")
    parser.add_argument("--model", default="model/nslkdd_model.pkl")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"ERROR: model bundle not found at {model_path}. "
                 "It is produced by export_model.py.")

    print("PROTOCOL: official (train on KDDTrain+, evaluate on KDDTest+), "
          f"threshold {THRESHOLD}")
    print(f"scikit-learn {sklearn.__version__}, pandas {pd.__version__}")

    train, test = load_official_frames(
        Path(args.train), Path(args.test), Path(args.merged_csv))
    print(f"Train rows: {len(train):,}   Test rows: {len(test):,}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    kept_features = bundle["kept_features"]
    categorical_cols = bundle["categorical_columns"]
    drop_always = bundle["drop_always"]

    # --- preprocessing, identical to export_model.py -----------------------
    test = test.copy()
    test["label"] = (test["class"].str.lower() != "normal").astype(int)
    for column in categorical_cols:
        test[column] = encoders[column].transform(test[column].astype(str))
    test = test.drop(columns=drop_always)
    X_scaled = pd.DataFrame(
        scaler.transform(test[feature_names]), columns=feature_names)
    X = X_scaled[kept_features].to_numpy(dtype=np.float32)
    y = test["label"].to_numpy()

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    overall_recall = tp / (tp + fn)

    print(f"\nKept features: {len(kept_features)}")
    print(f"TN = {tn}   FP = {fp}   FN = {fn}   TP = {tp}")
    print(f"Overall recall = {100 * overall_recall:.2f}%")

    # --- consistency check against the stored official-protocol metrics ----
    stored = bundle.get("metrics", {}).get("confusion_matrix")
    if stored is not None:
        found = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        if found != stored:
            print("\nERROR: confusion matrix does NOT match the stored metrics.")
            print(f"  stored: {stored}\n  found:  {found}")
            print("The preprocessing or library versions differ from "
                  "export_model.py. Do not commit these results. "
                  "Nothing was written.")
            sys.exit(2)
        print("Confusion matrix matches the stored metrics in the bundle.")

    # --- per attack type ----------------------------------------------------
    train_types = set(train["class"])
    attacks = test.loc[test["label"] == 1, ["class"]].copy()
    attacks["detected"] = pred[test["label"].to_numpy() == 1]

    per_type = (
        attacks.groupby("class")["detected"]
        .agg(n_records="size", n_detected="sum")
        .reset_index()
        .rename(columns={"class": "attack_type"})
    )
    per_type["n_detected"] = per_type["n_detected"].astype(int)
    per_type["n_missed"] = per_type["n_records"] - per_type["n_detected"]
    per_type["recall"] = (per_type["n_detected"] / per_type["n_records"]).round(4)
    per_type["family"] = per_type["attack_type"].str.lower().map(
        ATTACK_FAMILY).fillna("unmapped")
    per_type["in_train"] = per_type["attack_type"].isin(train_types)
    per_type = per_type[["attack_type", "family", "in_train", "n_records",
                         "n_detected", "n_missed", "recall"]]
    per_type = per_type.sort_values(
        ["n_missed", "attack_type"], ascending=[False, True]
    ).reset_index(drop=True)

    unmapped = per_type.loc[per_type["family"] == "unmapped", "attack_type"]
    if len(unmapped):
        print(f"\nWARNING: unmapped attack types: {sorted(unmapped)}")

    # --- aggregates ---------------------------------------------------------
    def aggregate(frame, level, group):
        n_records = int(frame["n_records"].sum())
        n_detected = int(frame["n_detected"].sum())
        return {
            "level": level, "group": group,
            "n_attack_types": len(frame),
            "n_records": n_records, "n_detected": n_detected,
            "n_missed": n_records - n_detected,
            "recall": round(n_detected / n_records, 4),
        }

    rows = [aggregate(per_type, "overall", "all attacks")]
    for family in sorted(per_type["family"].unique()):
        rows.append(aggregate(per_type[per_type["family"] == family],
                              "family", family))
    for flag in (True, False):
        subset = per_type[per_type["in_train"] == flag]
        if len(subset):
            rows.append(aggregate(subset, "in_train", str(flag)))
    per_family = pd.DataFrame(rows)

    # --- sanity checks on the sums -----------------------------------------
    assert per_type["n_records"].sum() == int((y == 1).sum())
    assert per_type["n_detected"].sum() == tp
    assert per_type["n_missed"].sum() == fn

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_type.to_csv(out_dir / "per_attack_recall.csv", index=False)
    per_family.to_csv(out_dir / "per_family_recall.csv", index=False)

    print(f"\nSum of n_records = {per_type['n_records'].sum():,}   "
          f"n_detected = {per_type['n_detected'].sum():,}   "
          f"n_missed = {per_type['n_missed'].sum():,}")
    print(f"Attack types in KDDTest+: {len(per_type)} "
          f"({int((~per_type['in_train']).sum())} not in KDDTrain+)")
    print("\nPer family / seen vs unseen:")
    print(per_family.to_string(index=False))
    print("\nTop 10 attack types by missed records:")
    print(per_type.head(10).to_string(index=False))
    print(f"\nWrote {out_dir / 'per_attack_recall.csv'} and "
          f"{out_dir / 'per_family_recall.csv'}")


if __name__ == "__main__":
    main()
