#!/usr/bin/env python3
"""
Replication of:
  "Machine learning-based intrusion detection framework using NSL-KDD dataset"
  Antony, Thaseen, Alashetty, Veronica, Kumar, Khan, Geethanjali, Sharma, Sekhar
  Discover Artificial Intelligence (Springer Nature), 2026, DOI 10.1007/s44163-026-01930-9

Dataset: NSL-KDD (KDDTrain+.csv / KDDTest+.csv, converted from the raw .txt files by
convert_nslkdd.py). 41 features + class label; num_outbound_cmds is constant (always 0)
across the entire dataset and is dropped before modelling.

Protocol:
  paper    : merges KDDTrain+ and KDDTest+ into one pool and does our own random 80/20
             split, matching the paper's stated "80:20 split" methodology. This is the
             set of numbers to compare directly against the paper's reported results.

Usage:
  pip install numpy pandas scikit-learn matplotlib statsmodels
    python replicate_nslkdd.py --merged-csv data\\NSL-KDD_merged.csv
    python replicate_nslkdd.py --train data\\KDDTrain+.csv --test data\\KDDTest+.csv
    python replicate_nslkdd.py --train ... --test ... --models lr,knn,nb,dt,rf

Everything the paper does not fully specify (VIF threshold and how unseen 'service'
categories in the test set are encoded) uses a documented default and is marked
ASSUMPTION below.
"""
import argparse
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, RobustScaler)
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Numbers reported in the paper (percent accuracy, from the Results section text
# -- the paper's actual data tables did not extract as machine-readable text, only
# the prose summary of each model's headline accuracy).
# ----------------------------------------------------------------------------
PAPER_ACCURACY = {
    "Logistic Regression": 96.72,
    "K-NN": 99.0,          # paper states ">99%", exact figure not given in prose
    "Naive Bayes": 89.71,
    "Decision Tree": 97.0,  # paper states "~97%"
    "Random Forest": 99.87,
}
TOLERANCE_POINTS = 7.0  # task requirement: within +-7%

VIF_THRESHOLD = 10.0  # ASSUMPTION: paper says VIF is used but never states the cutoff;
                       # 10 is the standard textbook rule of thumb for "high multicollinearity"

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DROP_ALWAYS = ["num_outbound_cmds"]  # constant (always 0) across the entire dataset
NOT_A_FEATURE = ["class", "difficulty_level"]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_raw(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"KDDTrain+: {train.shape[0]:,} rows   KDDTest+: {test.shape[0]:,} rows")

    for df, name in [(train, "train"), (test, "test")]:
        missing = [c for c in DROP_ALWAYS + NOT_A_FEATURE + CATEGORICAL_COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"{name} file is missing expected columns: {missing}")

    # Binary label: normal=0, any attack=1 (paper: "binary intrusion detection,
    # normal vs. attack classification")
    for df in (train, test):
        df["label"] = (df["class"].str.lower() != "normal").astype(int)

    return train, test


def load_merged(merged_path):
    merged = pd.read_csv(merged_path)
    if "source_file" not in merged.columns:
        raise SystemExit("Merged file must contain a source_file column")
    merged = merged.drop(columns=["source_file"])
    for column in DROP_ALWAYS + NOT_A_FEATURE + CATEGORICAL_COLS:
        if column not in merged.columns:
            raise SystemExit(f"Merged file is missing expected column: {column}")
    merged["label"] = (merged["class"].str.lower() != "normal").astype(int)
    print(f"Merged NSL-KDD: {merged.shape[0]:,} rows")
    return merged


def encode_categoricals(train, test):
    """Label-encode protocol_type/service/flag. ASSUMPTION: fit the encoder on the
    UNION of train+test categories, not train alone. NSL-KDD's test set is known to
    contain 'service' values that never appear in training (part of the benchmark's
    design, to test generalization to unseen conditions) -- fitting on train alone
    would crash on those at test time. This only touches categorical *encoding*, not
    the numeric features or the label, so it is not the kind of leakage SMOTE-before-
    split would be; it is standard practice for this specific dataset."""
    train, test = train.copy(), test.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le
    print("Encoded categoricals:", ", ".join(CATEGORICAL_COLS))
    return train, test, encoders


def feature_columns(df):
    return [c for c in df.columns if c not in NOT_A_FEATURE + DROP_ALWAYS + ["label"]]


def drop_constant_and_get_features(df):
    df = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns])
    return df


def vif_drop(X_df, threshold=VIF_THRESHOLD, max_drop_fraction=0.5):
    """Iteratively drop the single highest-VIF feature until all remaining features
    are below `threshold`, or until we've dropped max_drop_fraction of the columns
    (a safety valve -- VIF on one-hot-ish encoded categoricals can occasionally chain
    upward and strip more than intended if left unchecked)."""
    cols = list(X_df.columns)
    dropped = []
    max_drops = int(len(cols) * max_drop_fraction)
    X = X_df.values.astype(np.float64)

    while True:
        if X.shape[1] <= 1 or len(dropped) >= max_drops:
            break
        vifs = []
        for i in range(X.shape[1]):
            try:
                v = variance_inflation_factor(X, i)
            except Exception:
                v = np.inf
            vifs.append(v)
        vifs = np.array(vifs)
        worst_idx = int(np.nanargmax(vifs))
        if not np.isfinite(vifs[worst_idx]) or vifs[worst_idx] > threshold:
            dropped.append(cols[worst_idx])
            cols.pop(worst_idx)
            X = np.delete(X, worst_idx, axis=1)
        else:
            break
    return cols, dropped


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def prepare_paper(train, test, sample, seed):
    merged = pd.concat([train, test], axis=0, ignore_index=True)
    if sample:
        merged = merged.sample(n=min(sample, len(merged)), random_state=seed)
    feat_cols = feature_columns(merged)
    X = merged[feat_cols]
    y = merged["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    return finish_prepare(X_train, y_train, X_test, y_test, feat_cols)


def finish_prepare(X_train, y_train, X_test, y_test, feat_cols):
    # RobustScaler fit on train only (paper: RobustScaler, "less sensitive to outliers")
    scaler = RobustScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=feat_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feat_cols)

    # VIF-based feature selection, fit on train only to avoid leakage
    keep_cols, dropped_cols = vif_drop(X_train_s)
    print(f"VIF (threshold={VIF_THRESHOLD}) dropped {len(dropped_cols)} feature(s): {dropped_cols}")
    X_train_s = X_train_s[keep_cols].values.astype(np.float32)
    X_test_s = X_test_s[keep_cols].values.astype(np.float32)

    return X_train_s, X_test_s, y_train, y_test, keep_cols, scaler, dropped_cols


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def build_models(selected, seed, feature_names):
    models = {}
    if "lr" in selected:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000)
    if "knn" in selected:
        models["K-NN"] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if "nb" in selected:
        categorical_indices = [
            feature_names.index(col) for col in CATEGORICAL_COLS if col in feature_names
        ]
        models["Naive Bayes"] = Pipeline([
            ("one_hot_categories", ColumnTransformer(
                transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"),
                               categorical_indices)],
                remainder="passthrough")),
            ("classifier", BernoulliNB()),
        ])
    if "dt" in selected:
        models["Decision Tree"] = DecisionTreeClassifier(random_state=seed)
    if "rf" in selected:
        models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if "stack" in selected:
        categorical_indices = [
            feature_names.index(col) for col in CATEGORICAL_COLS if col in feature_names
        ]
        nb_pipeline = Pipeline([
            ("one_hot_categories", ColumnTransformer(
                transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"),
                               categorical_indices)],
                remainder="passthrough")),
            ("classifier", BernoulliNB()),
        ])
        models["Stacking (proposed)"] = StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=100, random_state=seed, n_jobs=-1)),
                ("dt", DecisionTreeClassifier(random_state=seed)),
                ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
                ("nb", nb_pipeline),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method="predict_proba",
            cv=5,
            n_jobs=-1,
        )

    return models


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(name, model, X_tr, y_tr, X_te, y_te, do_cv):
    print(f"\n--- {name} ---")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    cv_mean = cv_std = np.nan
    if do_cv:
        n_cv_rows = min(20000, len(y_tr))
        rs = np.random.default_rng(0)
        sub = rs.choice(len(y_tr), size=n_cv_rows, replace=False)
        scores = cross_val_score(model, X_tr[sub], y_tr[sub], cv=5, scoring="accuracy", n_jobs=-1)
        cv_mean, cv_std = scores.mean(), scores.std()

    row = {
        "model": name,
        "accuracy": 100 * accuracy_score(y_te, pred),
        "precision": 100 * precision_score(y_te, pred),
        "recall": 100 * recall_score(y_te, pred),
        "f1": 100 * f1_score(y_te, pred),
        "auc": roc_auc_score(y_te, proba),
        "cv_accuracy_mean": 100 * cv_mean if not np.isnan(cv_mean) else np.nan,
        "cv_accuracy_std": 100 * cv_std if not np.isnan(cv_std) else np.nan,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fit_seconds": round(fit_s, 1),
    }
    print(f"acc={row['accuracy']:.2f}  P={row['precision']:.2f}  R={row['recall']:.2f}  "
          f"F1={row['f1']:.2f}  AUC={row['auc']:.4f}  (fit {fit_s:.0f}s)"
          + (f"  cv_acc={row['cv_accuracy_mean']:.2f}+-{row['cv_accuracy_std']:.2f}" if do_cv else ""))
    return row, proba, pred


def compare_with_paper(results):
    print("\n=== Comparison with the paper (tolerance +-%.0f points) ===" % TOLERANCE_POINTS)
    for _, r in results.iterrows():
        paper_acc = PAPER_ACCURACY.get(r["model"])
        if paper_acc is None:
            continue
        d = r["accuracy"] - paper_acc
        ok = "OK " if abs(d) <= TOLERANCE_POINTS else "OUT"
        print(f"{r['model']:<20} accuracy   paper {paper_acc:6.2f}  ours {r['accuracy']:6.2f}  "
              f"diff {d:+6.2f}  [{ok}]")

    stacking = results[results["model"] == "Stacking (proposed)"]
    baselines = results[results["model"] != "Stacking (proposed)"]
    if not stacking.empty and not baselines.empty:
        stacking_accuracy = stacking.iloc[0]["accuracy"]
        top_row = baselines.loc[baselines["accuracy"].idxmax()]
        difference = stacking_accuracy - top_row["accuracy"]
        result = "better" if difference > 0 else "lower" if difference < 0 else "tied"
        print("\n=== Stacking (proposed) vs. top baseline ===")
        print(f"Stacking (proposed) accuracy {stacking_accuracy:.2f}  "
              f"top model {top_row['model']} {top_row['accuracy']:.2f}  "
              f"diff {difference:+.2f}  [{result}]")


def save_plots(out_dir, preds, probas, y_te, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return
    names = list(preds)
    n = len(names)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        cm = confusion_matrix(y_te, preds[name])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["normal", "attack"])
        ax.set_yticks([0, 1], ["normal", "attack"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(name)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices ({tag} protocol)")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrices_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in names:
        fpr, tpr, _ = roc_curve(y_te, probas[name])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves ({tag} protocol)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{tag}.png", dpi=150)
    plt.close(fig)


def save_model_bundle(path, model, encoders, scaler, feature_names, kept_features,
                      dropped_features):
    bundle = {
        "model_name": "Stacking (proposed)",
        "model": model,
        "categorical_encoders": encoders,
        "scaler": scaler,
        "feature_names": feature_names,
        "kept_features": kept_features,
        "dropped_features": dropped_features,
        "categorical_columns": CATEGORICAL_COLS,
        "drop_always": DROP_ALWAYS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved model bundle to: {path.resolve()}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--merged-csv", help="merged CSV; runs the paper protocol")
    input_group.add_argument("--train", help="path to KDDTrain+.csv")
    ap.add_argument("--test", help="path to KDDTest+.csv; required with --train")
    ap.add_argument("--models", default="lr,knn,nb,dt,rf,stack",
                    help="comma list from: lr,knn,nb,dt,rf,stack")
    ap.add_argument("--sample", type=int, default=0, help="subsample N rows for a quick test")
    ap.add_argument("--no-cv", action="store_true", help="skip the 5-fold cross-validation step (faster)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--save-pkl", action="store_true",
                    help="save the fitted Stacking (proposed) model and preprocessing")
    ap.add_argument("--pkl-out", default="results/nslkdd_stacking_proposed.pkl",
                    help="output path used with --save-pkl")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.save_pkl and "stack" not in selected:
        ap.error("--save-pkl requires stack in --models")

    if args.merged_csv:
        if args.test:
            ap.error("--test cannot be used with --merged-csv")
        merged = load_merged(args.merged_csv)
        train = drop_constant_and_get_features(merged)
        test = train.iloc[0:0].copy()
        train, test, encoders = encode_categoricals(train, test)
    else:
        if not args.train or not args.test:
            ap.error("--train and --test are required together")
        train, test = load_raw(args.train, args.test)
        train = drop_constant_and_get_features(train)
        test = drop_constant_and_get_features(test)
        train, test, encoders = encode_categoricals(train, test)

    print("\n" + "=" * 78 + "\nPROTOCOL: paper\n" + "=" * 78)
    X_tr, X_te, y_tr, y_te, kept, scaler, dropped_features = prepare_paper(
        train, test, args.sample, args.seed)
    print(f"Train: {X_tr.shape}   Test: {X_te.shape}   "
          f"Train attack rate: {y_tr.mean():.1%}   Test attack rate: {y_te.mean():.1%}")

    rows, preds, probas = [], {}, {}
    fitted_models = {}
    models = build_models(selected, args.seed, kept)
    for name, model in models.items():
        row, proba, pred = evaluate(name, model, X_tr, y_tr, X_te, y_te, do_cv=not args.no_cv)
        rows.append(row)
        preds[name], probas[name] = pred, proba
        fitted_models[name] = model

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "metrics_paper.csv", index=False)
    save_plots(out_dir, preds, probas, y_te, "paper")

    show = ["model", "accuracy", "precision", "recall", "f1", "auc",
            "cv_accuracy_mean", "cv_accuracy_std", "tn", "fp", "fn", "tp"]
    print("\n=== Results (paper protocol) ===")
    print(results[show].round(4).to_string(index=False))

    compare_with_paper(results)

    if args.save_pkl:
        save_model_bundle(
            Path(args.pkl_out),
            fitted_models["Stacking (proposed)"],
            encoders=encoders,
            scaler=scaler,
            feature_names=feature_columns(train),
            kept_features=kept,
            dropped_features=dropped_features,
        )

    print(f"\nSaved metrics CSVs and plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
