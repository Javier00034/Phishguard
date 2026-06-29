import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# ------------------------
# Paths
# This file lives at <repo_root>/PhishGuard/dashboard/app.py, so the repo root
# is three levels up. The actual data/models/src live at the repo root.
# ------------------------

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parent.parent.parent          # repo root
NESTED_DIR = THIS_FILE.parent.parent               # PhishGuard/


def _first_existing(*candidates):
    """Return the first path that exists, else the last candidate."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


# Prefer the repo-root src; fall back to the nested PhishGuard/src.
SRC_DIR = _first_existing(ROOT_DIR / "src", NESTED_DIR / "src")
sys.path.append(str(SRC_DIR))

from feature_extractor import FeatureExtractor

st.set_page_config(
    page_title="PhishGuard",
    layout="wide"
)

st.title("🛡️ PhishGuard")
st.subheader("Machine Learning Phishing URL Detection")


# ------------------------
# Resolve data/model/output file locations (root first, nested fallback)
# ------------------------

MODEL_PATH = _first_existing(
    ROOT_DIR / "data" / "models" / "logistic_regression.pkl",
    NESTED_DIR / "data" / "models" / "logistic_regression.pkl",
)
SCALER_PATH = _first_existing(
    ROOT_DIR / "data" / "models" / "scaler.pkl",
    NESTED_DIR / "data" / "models" / "scaler.pkl",
)
DATASET_PATH = _first_existing(
    ROOT_DIR / "data" / "processed" / "master_urls.csv",
    NESTED_DIR / "data" / "processed" / "master_urls.csv",
)
RESULTS_PATH = _first_existing(
    ROOT_DIR / "outputs" / "reports" / "model_results.csv",
    NESTED_DIR / "outputs" / "reports" / "model_results.csv",
)


# ------------------------
# Cached loaders with visible error handling
# ------------------------

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


try:
    model, scaler = load_model_and_scaler()
    extractor = FeatureExtractor()
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.info(f"Looked for model at: {MODEL_PATH}")
    st.stop()

# ------------------------
# Dataset Statistics
# ------------------------

st.header("Dataset Overview")

try:
    dataset = load_dataset()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total URLs", len(dataset))
    col2.metric("Phishing URLs", len(dataset[dataset["label"] == 1]))
    col3.metric("Legitimate URLs", len(dataset[dataset["label"] == 0]))
except Exception as e:
    st.warning(f"Could not load dataset overview: {e}")

# ------------------------
# Model Results
# ------------------------

st.header("Model Performance")

try:
    results = load_results()
    st.dataframe(results, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load model results: {e}")

# ------------------------
# Live Detection
# ------------------------

st.header("Live URL Detection")

url = st.text_input("Enter URL")

if st.button("Analyse URL"):

    if url:
        try:
            features = extractor.extract(url)
            feature_df = pd.DataFrame([features])

            scaled = scaler.transform(feature_df)
            prediction = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0][1]

            if prediction == 1:
                st.error("⚠️ Phishing Detected")
            else:
                st.success("✅ Legitimate URL")

            st.metric("Risk Score", f"{probability*100:.2f}%")

            st.subheader("Extracted Features")
            st.dataframe(feature_df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to analyse URL: {e}")
    else:
        st.info("Please enter a URL to analyse.")
