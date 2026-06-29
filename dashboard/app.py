import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent / "src"
    )
)

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
# Load Model
# ------------------------

model = joblib.load(
    "data/models/logistic_regression.pkl"
)

scaler = joblib.load(
    "data/models/scaler.pkl"
)

extractor = FeatureExtractor()

# ------------------------
# Dataset Statistics
# ------------------------

st.header("Dataset Overview")

dataset = pd.read_csv(
    "data/processed/master_urls.csv"
)

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

# ------------------------
# Model Results
# ------------------------

st.header(
    "Model Performance"
)

results = pd.read_csv(
    "outputs/reports/model_results.csv"
)

st.dataframe(
    results,
    use_container_width=True
)

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