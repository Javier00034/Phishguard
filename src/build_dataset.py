import requests
import pandas as pd
from pathlib import Path

OPENPHISH_URL = (
    "https://raw.githubusercontent.com/openphish/public_feed/"
    "refs/heads/main/feed.txt"
)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_phishing_urls():
    print("Downloading OpenPhish feed...")

    response = requests.get(OPENPHISH_URL, timeout=30)
    response.raise_for_status()

    urls = [
        url.strip()
        for url in response.text.splitlines()
        if url.strip()
    ]

    phishing_df = pd.DataFrame({
        "url": urls,
        "label": 1
    })

    return phishing_df


def save_dataset(df):
    output_file = PROCESSED_DIR / "master_urls.csv"

    df.to_csv(output_file, index=False)

    print(f"Saved dataset to: {output_file}")
    print(f"Rows: {len(df)}")


def load_legitimate_urls():

    legit_file = RAW_DIR / "legitimate_urls.csv"

    legit_df = pd.read_csv(legit_file)

    return legit_df


def main():

    phishing_df = download_phishing_urls()

    legit_df = load_legitimate_urls()

    combined_df = pd.concat(
        [phishing_df, legit_df],
        ignore_index=True
    )

    combined_df = combined_df.drop_duplicates(
        subset=["url"]
    )

    save_dataset(combined_df)

    print("\nDataset Statistics")
    print("-------------------")
    print(combined_df["label"].value_counts())


if __name__ == "__main__":
    main()