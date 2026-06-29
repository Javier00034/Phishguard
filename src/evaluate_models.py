import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_FILE = "data/processed/features.csv"

OUTPUT_DIR = "outputs"

print("Loading dataset...")

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

scaler = joblib.load(
    "data/models/scaler.pkl"
)

X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression":
        joblib.load(
            "data/models/logistic_regression.pkl"
        ),

    "Decision Tree":
        joblib.load(
            "data/models/decision_tree.pkl"
        ),

    "Random Forest":
        joblib.load(
            "data/models/random_forest.pkl"
        )
}

results = []

for name, model in models.items():

    print(f"\nEvaluating {name}")

    if name == "Logistic Regression":

        predictions = model.predict(
            X_test_scaled
        )

        probabilities = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    cm = confusion_matrix(
        y_test,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(name)

    plt.savefig(
        f"{OUTPUT_DIR}/confusion_matrices/"
        f"{name.replace(' ','_')}.png"
    )

    plt.close()

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={roc_auc:.3f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"ROC Curve - {name}"
    )

    plt.legend()

    plt.savefig(
        f"{OUTPUT_DIR}/roc_curves/"
        f"{name.replace(' ','_')}.png"
    )

    plt.close()

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    f"{OUTPUT_DIR}/reports/model_results.csv",
    index=False
)

print("\nEvaluation Results")

print(results_df)

print(
    "\nSaved report to:"
)

print(
    "outputs/reports/model_results.csv"
)