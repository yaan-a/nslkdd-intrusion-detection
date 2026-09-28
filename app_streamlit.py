"""Streamlit single-connection NSL-KDD classifier."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Official protocol model": PROJECT_ROOT / "model" / "nslkdd_model.pkl",
    "Paper protocol model": PROJECT_ROOT / "model" / "nslkdd_stacking_proposed.pkl",
}
MERGED_DATA_PATH = PROJECT_ROOT / "data" / "NSL-KDD_merged.csv"
PAPER_METRICS_PATH = PROJECT_ROOT / "results" / "metrics_paper.csv"

FEATURE_GLOSSARY = {
    "duration": "connection duration in seconds",
    "protocol_type": "network protocol used by the connection",
    "service": "destination network service",
    "flag": "connection status or TCP flag",
    "src_bytes": "bytes sent from source to destination",
    "dst_bytes": "bytes sent from destination to source",
    "land": "whether source and destination host/port are identical",
    "wrong_fragment": "number of malformed fragments",
    "urgent": "number of urgent packets",
    "hot": "number of hot indicators",
    "num_failed_logins": "number of failed login attempts",
    "logged_in": "whether the session successfully logged in",
    "num_compromised": "number of compromised conditions",
    "root_shell": "whether a root shell was obtained",
    "su_attempted": "whether a superuser command was attempted",
    "num_file_creations": "number of file creation operations",
    "num_shells": "number of shell prompts invoked",
    "num_access_files": "number of operations on access-control files",
    "is_host_login": "whether the login is a host login",
    "is_guest_login": "whether the login is a guest login",
    "count": "connections to the same host in the recent window",
    "srv_count": "connections to the same service in the recent window",
    "serror_rate": "percentage of connections with SYN error responses",
    "srv_serror_rate": "SYN error percentage for the same service",
    "rerror_rate": "percentage of connections with REJ error responses",
    "srv_rerror_rate": "REJ error percentage for the same service",
    "same_srv_rate": "percentage of connections to the same service",
    "diff_srv_rate": "percentage of connections to different services",
    "srv_diff_host_rate": "different-host percentage for the same service",
    "dst_host_count": "connections to the destination host",
    "dst_host_srv_count": "connections to the destination host and service",
    "dst_host_same_srv_rate": "same-service percentage for the destination host",
    "dst_host_diff_srv_rate": "different-service percentage for the destination host",
    "dst_host_same_src_port_rate": "same-source-port percentage for the destination host",
    "dst_host_srv_diff_host_rate": "different-host percentage for the destination host and service",
    "dst_host_serror_rate": "SYN error percentage for the destination host",
    "dst_host_srv_serror_rate": "SYN error percentage for the destination host and service",
    "dst_host_rerror_rate": "REJ error percentage for the destination host",
    "dst_host_srv_rerror_rate": "REJ error percentage for the destination host and service",
}


def load_paper_metrics():
    if not PAPER_METRICS_PATH.exists():
        return None
    rows = pd.read_csv(PAPER_METRICS_PATH)
    row = rows.loc[rows["model"] == "Stacking (proposed)"]
    if row.empty:
        return None
    values = row.iloc[0]
    return {
        "accuracy": float(values["accuracy"]),
        "precision": float(values["precision"]),
        "recall": float(values["recall"]),
        "f1": float(values["f1"]),
        "auc": float(values["auc"]),
        "confusion_matrix": {
            "tn": int(values["tn"]),
            "fp": int(values["fp"]),
            "fn": int(values["fn"]),
            "tp": int(values["tp"]),
        },
    }


def add_paper_bundle_metadata(bundle):
    bundle = dict(bundle)
    bundle["encoders"] = bundle.pop("categorical_encoders")
    bundle["protocol"] = "paper"
    bundle["metrics"] = load_paper_metrics()

    if MERGED_DATA_PATH.exists():
        data = pd.read_csv(MERGED_DATA_PATH)
        numeric_defaults = {}
        numeric_bounds = {}
        for feature in bundle["kept_features"]:
            if feature in bundle["categorical_columns"]:
                continue
            values = data[feature].astype(float)
            numeric_defaults[feature] = float(values.median())
            numeric_bounds[feature] = {
                "min": float(values.quantile(0.01)),
                "max": float(values.quantile(0.99)),
            }
        bundle["numeric_defaults"] = numeric_defaults
        bundle["numeric_bounds"] = numeric_bounds
    else:
        bundle["numeric_defaults"] = {}
        bundle["numeric_bounds"] = {}
    return bundle


@st.cache_resource
def load_bundle(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Create the model bundle before starting Streamlit."
        )
    bundle = joblib.load(model_path)
    if "categorical_encoders" in bundle:
        return add_paper_bundle_metadata(bundle)
    return bundle


def display_model_info(bundle):
    metrics = bundle.get("metrics")
    if metrics is None:
        st.info("Evaluation metrics are not available for this model bundle.")
        return
    matrix = metrics["confusion_matrix"]
    with st.expander("Model info", expanded=True):
        st.caption(
            "Official-protocol evaluation: trained on KDDTrain+.csv and evaluated "
            "on KDDTest+.csv, including attack types not seen during training."
            if bundle.get("protocol") == "official" else
            "Paper-protocol evaluation: the merged NSL-KDD data was randomly split "
            "into training and test partitions."
        )
        st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
        columns = st.columns(4)
        columns[0].metric("Precision", f"{metrics['precision']:.2f}%")
        columns[1].metric("Recall", f"{metrics['recall']:.2f}%")
        columns[2].metric("F1", f"{metrics['f1']:.2f}%")
        columns[3].metric("ROC-AUC", f"{metrics['auc']:.4f}")
        st.write("Confusion matrix")
        st.dataframe(pd.DataFrame([
            [matrix["tn"], matrix["fp"]],
            [matrix["fn"], matrix["tp"]],
        ], index=["actual normal", "actual attack"],
           columns=["predicted normal", "predicted attack"]), use_container_width=True)


def initialize_defaults(bundle):
    for feature in bundle["kept_features"]:
        key = f"feature_{feature}"
        if key in st.session_state:
            continue
        if feature in bundle["categorical_columns"]:
            st.session_state[key] = bundle["encoders"][feature].classes_[0]
        else:
            st.session_state[key] = bundle["numeric_defaults"].get(feature, 0.0)


def randomize_inputs(bundle):
    rng = __import__("numpy").random.default_rng()
    for feature in bundle["kept_features"]:
        key = f"feature_{feature}"
        if feature in bundle["categorical_columns"]:
            choices = bundle["encoders"][feature].classes_
            st.session_state[key] = str(rng.choice(choices))
        else:
            bounds = bundle["numeric_bounds"].get(feature, {"min": 0.0, "max": 1.0})
            low, high = bounds["min"], bounds["max"]
            st.session_state[key] = float(rng.uniform(low, high))


def main():
    st.set_page_config(page_title="NSL-KDD Detector", page_icon="🛡️", layout="wide")
    st.title("NSL-KDD Intrusion Detector")
    selected_model = st.selectbox("Model", list(MODEL_OPTIONS))
    st.write("Classify one network connection with the selected stacking model.")

    try:
        bundle = load_bundle(str(MODEL_OPTIONS[selected_model]))
    except Exception as error:
        st.error(str(error))
        st.stop()

    display_model_info(bundle)
    initialize_defaults(bundle)

    if st.button("Randomize test connection"):
        randomize_inputs(bundle)
        st.rerun()

    st.subheader("Connection details")
    with st.form("connection_form"):
        input_values = {}
        left, right = st.columns(2)
        for index, feature in enumerate(bundle["kept_features"]):
            container = left if index % 2 == 0 else right
            label = f"{feature}: {FEATURE_GLOSSARY.get(feature, 'NSL-KDD feature value')}"
            key = f"feature_{feature}"
            with container:
                if feature in bundle["categorical_columns"]:
                    options = [str(value) for value in bundle["encoders"][feature].classes_]
                    input_values[feature] = st.selectbox(label, options, key=key)
                else:
                    bounds = bundle["numeric_bounds"].get(feature, {"min": 0.0, "max": 1.0})
                    minimum = min(0.0, bounds["min"])
                    maximum = max(minimum + 1.0, bounds["max"])
                    input_values[feature] = st.number_input(
                        label, min_value=float(minimum), max_value=float(maximum),
                        key=key,
                    )
        submitted = st.form_submit_button("Classify connection")

    if submitted:
        raw_row = {}
        for feature in bundle["feature_names"]:
            if feature in input_values:
                raw_row[feature] = input_values[feature]
            elif feature in bundle["categorical_columns"]:
                raw_row[feature] = bundle["encoders"][feature].classes_[0]
            else:
                raw_row[feature] = 0.0

        encoded_row = raw_row.copy()
        for feature in bundle["categorical_columns"]:
            encoded_row[feature] = bundle["encoders"][feature].transform([
                str(encoded_row[feature])
            ])[0]
        frame = pd.DataFrame([encoded_row], columns=bundle["feature_names"])
        scaled = pd.DataFrame(
            bundle["scaler"].transform(frame), columns=bundle["feature_names"]
        )
        model_input = scaled[bundle["kept_features"]].to_numpy(dtype="float32")
        attack_probability = float(bundle["model"].predict_proba(model_input)[0, 1])
        normal_probability = 1.0 - attack_probability

        st.divider()
        if attack_probability >= 0.5:
            st.error(f"Likely an ATTACK ({attack_probability:.1%} attack risk)")
        else:
            st.success(f"Likely NORMAL traffic ({normal_probability:.1%} normal confidence)")
        st.progress(attack_probability, text=f"Attack risk: {attack_probability:.1%}")

    st.caption(
        "Educational lab demo only. This model learned from a fixed set of historical "
        "attack examples and can miss attack types not represented in its training data. "
        "It is not a production intrusion detection system."
    )


if __name__ == "__main__":
    main()
