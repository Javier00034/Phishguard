import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

FEATURE_FILE = "data/processed/features.csv"

print("Loading data...")

df = pd.read_csv(FEATURE_FILE)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Loading model...")

model = joblib.load(
    "data/models/logistic_regression.pkl"
)

scaler = joblib.load(
    "data/models/scaler.pkl"
)

X_test_scaled = scaler.transform(X_test)

print("Calculating SHAP values...")

explainer = shap.LinearExplainer(
    model,
    X_test_scaled
)

shap_values = explainer.shap_values(
    X_test_scaled
)

print("Generating summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=X.columns,
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/shap/shap_summary.png",
    bbox_inches="tight"
)

plt.close()

print("SHAP summary saved.")