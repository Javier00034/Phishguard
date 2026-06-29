import pandas as pd
import joblib

from feature_extractor import FeatureExtractor

model = joblib.load(
    "data/models/logistic_regression.pkl"
)

scaler = joblib.load(
    "data/models/scaler.pkl"
)

url = "https://google.com"

extractor = FeatureExtractor()

features = extractor.extract(url)

df = pd.DataFrame([features])

scaled = scaler.transform(df)

print("Features:")
print(df)

print("\nClasses:")
print(model.classes_)

print("\nPrediction:")
print(model.predict(scaled))

print("\nProbabilities:")
print(model.predict_proba(scaled))