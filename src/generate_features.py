import pandas as pd

from feature_extractor import FeatureExtractor

INPUT_FILE = "data/processed/master_urls.csv"
OUTPUT_FILE = "data/processed/features.csv"

extractor = FeatureExtractor()


def generate_features():

    print("Loading URLs...")

    df = pd.read_csv(INPUT_FILE)

    rows = []

    for index, row in df.iterrows():

        features = extractor.extract(
            row["url"]
        )

        features["label"] = row["label"]

        rows.append(features)

        if index % 100 == 0:

            print(
                f"Processed {index} URLs"
            )

    feature_df = pd.DataFrame(rows)

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature extraction complete.")
    print(feature_df.head())

    print(
        f"\nSaved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    generate_features()