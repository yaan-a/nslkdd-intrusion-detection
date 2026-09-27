#!/usr/bin/env python3
"""Train the official NSL-KDD stacking model and export one inference bundle."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DROP_ALWAYS = ["num_outbound_cmds"]
VIF_DROPS = [
    "num_root", "srv_serror_rate", "srv_rerror_rate",
    "dst_host_srv_serror_rate", "flag", "dst_host_same_srv_rate",
]
NOT_FEATURES = ["class", "difficulty_level", "label"]


def build_stacking_model(feature_names, seed):
    categorical_indices = [
        feature_names.index(column)
        for column in CATEGORICAL_COLS
        if column in feature_names
    ]

    def make_nb_pipeline():
        return Pipeline([
            ("one_hot_categories", ColumnTransformer(
                transformers=[(
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_indices,
                )],
                remainder="passthrough",
            )),
            ("classifier", BernoulliNB()),
        ])

    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=100, random_state=seed, n_jobs=-1)),
            ("dt", DecisionTreeClassifier(random_state=seed)),
            ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            ("nb", make_nb_pipeline()),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )


def validate_columns(train, test):
    if list(train.columns) != list(test.columns):
        raise SystemExit(
            "KDDTrain+.csv and KDDTest+.csv must have identical columns in "
            "the same order."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/KDDTrain+.csv")
    parser.add_argument("--test", default="data/KDDTest+.csv")
    parser.add_argument("--out", default="model/nslkdd_model.pkl")
    parser.add_argument("--metrics-out", default="model/metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_columns(train, test)

    required = set(DROP_ALWAYS + ["class"] + CATEGORICAL_COLS)
    missing = required - set(train.columns)
    if missing:
        raise SystemExit(f"Training data is missing columns: {sorted(missing)}")

    for frame in (train, test):
        frame["label"] = (frame["class"].str.lower() != "normal").astype(int)

    encoders = {}
    for column in CATEGORICAL_COLS:
        encoder = LabelEncoder()
        encoder.fit(pd.concat([train[column], test[column]]).astype(str))
        train[column] = encoder.transform(train[column].astype(str))
        test[column] = encoder.transform(test[column].astype(str))
        encoders[column] = encoder

    train = train.drop(columns=DROP_ALWAYS)
    test = test.drop(columns=DROP_ALWAYS)
    feature_names = [column for column in train.columns if column not in NOT_FEATURES]
    kept_features = [column for column in feature_names if column not in VIF_DROPS]

    scaler = RobustScaler().fit(train[feature_names])
    X_train_scaled = pd.DataFrame(
        scaler.transform(train[feature_names]), columns=feature_names
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(test[feature_names]), columns=feature_names
    )
    X_train = X_train_scaled[kept_features].to_numpy(dtype=np.float32)
    X_test = X_test_scaled[kept_features].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    model = build_stacking_model(kept_features, args.seed)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    metrics = {
        "accuracy": 100 * accuracy_score(y_test, predictions),
        "precision": 100 * precision_score(y_test, predictions, zero_division=0),
        "recall": 100 * recall_score(y_test, predictions, zero_division=0),
        "f1": 100 * f1_score(y_test, predictions, zero_division=0),
        "auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # Raw-value defaults and ranges make the local Streamlit form usable.
    numeric_defaults = {}
    numeric_bounds = {}
    for column in kept_features:
        if column in CATEGORICAL_COLS:
            continue
        values = train[column].astype(float)
        numeric_defaults[column] = float(values.median())
        numeric_bounds[column] = {
            "min": float(values.quantile(0.01)),
            "max": float(values.quantile(0.99)),
        }

    bundle = {
        "model_name": "Stacking (proposed)",
        "protocol": "official",
        "model": model,
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": feature_names,
        "kept_features": kept_features,
        "dropped_features": VIF_DROPS,
        "categorical_columns": CATEGORICAL_COLS,
        "drop_always": DROP_ALWAYS,
        "numeric_defaults": numeric_defaults,
        "numeric_bounds": numeric_bounds,
        "metrics": metrics,
    }

    output_path = Path(args.out)
    metrics_path = Path(args.metrics_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Official train rows: {len(train):,}")
    print(f"Official test rows: {len(test):,}")
    print(f"Kept features: {len(kept_features)}")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {output_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
