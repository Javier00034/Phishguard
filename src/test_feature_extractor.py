from feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

url = (
    "https://paypal-login.verify-account.com"
)

features = extractor.extract(url)

print(features)