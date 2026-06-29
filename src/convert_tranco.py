import pandas as pd

domains = []

with open("data/raw/tranco.csv", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split(",")

        # Format: rank,domain
        if len(parts) >= 2:
            domain = parts[1].strip()
        else:
            domain = parts[0].strip()

        domains.append(domain)

legit_df = pd.DataFrame({
    "url": ["https://" + d for d in domains],
    "label": 0
})

legit_df.to_csv(
    "data/raw/legitimate_urls.csv",
    index=False
)

print(f"Created {len(legit_df)} legitimate URLs")
print(legit_df.head())