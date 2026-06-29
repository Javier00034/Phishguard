import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# ------------------------
# Paths (anchored to repo root, not the working directory)
# ------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.append(str(SRC_DIR))

from feature_extractor import FeatureExtractor

st.set_page_config(
    page_title="PhishGuard",
    layout="wide"
)

st.title("🛡️ PhishGuard")
st.subheader(
    "Machine Learning Phishing URL Detection"
)


# ------------------------
# Cached loaders with visible error handling
# ------------------------

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(BASE_DIR / "data" / "models" / "logistic_regression.pkl")
    scaler = joblib.load(BASE_DIR / "data" / "models" / "scaler.pkl")
    return model, scaler


@st.cache_data
def load_dataset():
    return pd.read_csv(BASE_DIR / "data" / "processed" / "master_urls.csv")


@st.cache_data
def load_results():
    return pd.read_csv(BASE_DIR / "outputs" / "reports" / "model_results.csv")


try:
    model, scaler = load_model_and_scaler()
    extractor = FeatureExtractor()
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.stop()

# ------------------------
# Dataset Statistics
# ------------------------

st.header("Dataset Overview")

try:
    dataset = load_dataset()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total URLs",
        len(dataset)
    )

    col2.metric(
        "Phishing URLs",
        len(
            dataset[
                dataset["label"] == 1
            ]
        )
    )

    col3.metric(
        "Legitimate URLs",
        len(
            dataset[
                dataset["label"] == 0
            ]
        )
    )
except Exception as e:
    st.warning(f"Could not load dataset overview: {e}")

# ------------------------
# Model Results
# ------------------------

st.header(
    "Model Performance"
)

try:
    results = load_results()

    st.dataframe(
        results,
        use_container_width=True
    )
except Exception as e:
    st.warning(f"Could not load model results: {e}")

# ------------------------
# Live Detection
# ------------------------

st.header(
    "Live URL Detection"
)

url = st.text_input(
    "Enter URL"
)

if st.button(
    "Analyse URL"
):

    if url:

        try:
            features = extractor.extract(
                url
            )

            feature_df = pd.DataFrame(
                [features]
            )

            scaled = scaler.transform(
                feature_df
            )

            prediction = model.predict(
                scaled
            )[0]

            probability = model.predict_proba(
                scaled
            )[0][1]

            if prediction == 1:

                st.error(
                    f"⚠️ Phishing Detected"
                )

            else:

                st.success(
                    f"✅ Legitimate URL"
                )

            st.metric(
                "Risk Score",
                f"{probability*100:.2f}%"
            )

            st.subheader(
                "Extracted Features"
            )

            st.dataframe(
                feature_df,
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Failed to analyse URL: {e}")
    else:
        st.info("Please enter a URL to analyse.")
