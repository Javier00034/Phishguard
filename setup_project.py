from pathlib import Path

PROJECT_NAME = "PhishGuard"

folders = [

    # Data
    "data",
    "data/raw",
    "data/processed",
    "data/models",

    # Outputs
    "outputs",
    "outputs/confusion_matrices",
    "outputs/roc_curves",
    "outputs/shap",
    "outputs/reports",

    # Source code
    "src",

    # Dashboard
    "dashboard",

    # Documentation
    "docs",

    # Tests
    "tests"
]

files = {

    # Root
    "README.md": "",
    ".gitignore": "",
    "requirements.txt": "",

    # Source
    "src/__init__.py": "",
    "src/config.py": "",
    "src/data_loader.py": "",
    "src/build_dataset.py": "",
    "src/feature_extractor.py": "",
    "src/generate_features.py": "",
    "src/preprocess.py": "",
    "src/train_models.py": "",
    "src/evaluate_models.py": "",
    "src/explainability.py": "",
    "src/realtime_detector.py": "",
    "src/utils.py": "",

    # Dashboard
    "dashboard/app.py": "",

    # Tests
    "tests/test_feature_extractor.py": "",
    "tests/test_models.py": ""
}


def create_project():

    root = Path(PROJECT_NAME)

    root.mkdir(exist_ok=True)

    print(f"Created project: {PROJECT_NAME}")

    for folder in folders:

        path = root / folder

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"Created folder: {path}")

    for file_path, content in files.items():

        full_path = root / file_path

        full_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        full_path.write_text(content)

        print(f"Created file: {full_path}")

    print("\nProject structure created successfully.")


if __name__ == "__main__":
    create_project()