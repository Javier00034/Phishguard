import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

FEATURE_FILE = "data/processed/features.csv"

MODEL_DIR = "data/models"

print("Loading feature dataset...")

df = pd.read_csv(FEATURE_FILE)

X = df.drop("label", axis=1)
y = df["label"]

print(f"Dataset Shape: {df.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    f"{MODEL_DIR}/scaler.pkl"
)

print("\nTraining Logistic Regression...")

start = time.time()

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(
    X_train_scaled,
    y_train
)

print(
    f"Completed in {time.time()-start:.2f}s"
)

joblib.dump(
    lr,
    f"{MODEL_DIR}/logistic_regression.pkl"
)

print("\nTraining Decision Tree...")

start = time.time()

dt = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

print(
    f"Completed in {time.time()-start:.2f}s"
)

joblib.dump(
    dt,
    f"{MODEL_DIR}/decision_tree.pkl"
)

print("\nTraining Random Forest...")

start = time.time()

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

print(
    f"Completed in {time.time()-start:.2f}s"
)

joblib.dump(
    rf,
    f"{MODEL_DIR}/random_forest.pkl"
)

print("\nTraining K-Means...")

start = time.time()

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans.fit(
    X_train_scaled
)

print(
    f"Completed in {time.time()-start:.2f}s"
)

joblib.dump(
    kmeans,
    f"{MODEL_DIR}/kmeans.pkl"
)

print("\nTraining Complete.")

print("\nSaved Models:")

print("- scaler.pkl")
print("- logistic_regression.pkl")
print("- decision_tree.pkl")
print("- random_forest.pkl")
print("- kmeans.pkl")