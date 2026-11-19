
import json
from pathlib import Path

import pandas as pd
from PIL import Image

# =============================================================
# Configuration
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "data_preparation"
    / "prepared"
    / "val_prepared.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
IMAGE_DIR = PROJECT_ROOT / "data" / "images_val"
SAMPLE_SIZE = 100
SEED = 42

REQUIRED_COLUMNS = [
    "image",
    "question",
    "reference_answers",
    "answerable",
]


# =============================================================
# Stratified Sampling
# =============================================================

def stratified_sample(df, n, seed):
    """Select a reproducible sample while preserving class proportions."""

    if n > len(df):
        raise ValueError("Sample size cannot exceed dataset size.")

    groups = list(df.groupby("answerable", sort=True))

    exact_sizes = [
        n * len(group) / len(df)
        for _, group in groups
    ]

    sample_sizes = [int(size) for size in exact_sizes]

    remaining = n - sum(sample_sizes)

    order = sorted(
        range(len(groups)),
        key=lambda i: exact_sizes[i] - sample_sizes[i],
        reverse=True,
    )

    for i in order[:remaining]:
        sample_sizes[i] += 1

    # Check class sample sizes
    for (_, group), size in zip(groups, sample_sizes):
        if size > len(group):
            raise ValueError(
                "Not enough records in one of the classes."
            )

    # Select samples from each class
    parts = [
        group.sample(n=size, random_state=seed)
        for (_, group), size in zip(groups, sample_sizes)
    ]

    return pd.concat(parts).sort_index().reset_index(drop=True)
# =============================================================
# Image Validation
# =============================================================

def validate_image(image_name):
    """Check whether an image exists and can be opened."""

    if not isinstance(image_name, str) or not image_name.strip():
        return False, "missing"

    image_path = IMAGE_DIR / image_name

    if not image_path.exists():
        return False, "missing"

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            img.load()

        return True, "ok"

    except Exception:
        return False, "unreadable"

# =============================================================
# Main
# =============================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prepared validation file not found: {INPUT_FILE}"
        )

    df = pd.read_json(INPUT_FILE)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Validate labels
    if not df["answerable"].isin([0, 1]).all():
        raise ValueError("Invalid values found in answerable.")

    # Validate reference answers
    if not df["reference_answers"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    ).all():
        raise ValueError("Missing or invalid reference answers found.")
    # Validate image directory
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation image directory not found: {IMAGE_DIR}"
        )

    # Check all validation images
    image_results = df["image"].apply(validate_image)

    df["image_valid"] = image_results.apply(lambda x: x[0])
    df["image_status"] = image_results.apply(lambda x: x[1])

    print("\n=== IMAGE VALIDATION ===")
    print(f"Total records checked: {len(df)}")
    print(df["image_status"].value_counts())



    # Keep records with valid images only
    valid_df = df[df["image_valid"]].copy()
    if len(valid_df) < SAMPLE_SIZE:
        raise ValueError(
            f"Not enough valid records: {len(valid_df)}"
        )

    if valid_df["answerable"].nunique() < 2:
        raise ValueError(
            "Not enough valid records from both classes."
        )

    print(f"\nValid records: {len(valid_df)}")
    print(f"Excluded records: {len(df) - len(valid_df)}")

    # Select the evaluation sample
    sample = stratified_sample(valid_df, SAMPLE_SIZE, SEED)

    # Keep only the required fields
    sample = sample[REQUIRED_COLUMNS].copy()

    # Check for duplicate image-question pairs
    if sample.duplicated(subset=["image", "question"]).any():
        raise ValueError("Duplicate image-question pairs found.")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "evaluation_sample_100_checked.json"
    csv_path = OUTPUT_DIR / "evaluation_sample_100_checked.csv"

    sample.to_json(
        json_path,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    csv_sample = sample.copy()
    csv_sample["reference_answers"] = csv_sample[
        "reference_answers"
    ].apply(lambda x: json.dumps(x, ensure_ascii=False))

    csv_sample.to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
    )

    # Print summary
    print("\n=== EVALUATION SAMPLE ===")
    print(f"Total records: {len(sample)}")
    print(f"Seed: {SEED}")
    print("\nClass distribution:")
    print(sample["answerable"].value_counts().sort_index())

    print("\nSaved files:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
