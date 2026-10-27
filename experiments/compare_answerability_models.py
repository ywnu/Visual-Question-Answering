"""Supplementary text-only answerability experiment inspired by Lab 4 (SVM).

This does NOT perform Visual Question Answering. It compares Logistic
Regression and Linear SVM using the same TF-IDF question features and split.
Run from the repository after installing pandas, scikit-learn and matplotlib,
with VizWiz annotations available locally at data/train.json.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "train.json"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"Missing {DATA_FILE}. Download the VizWiz training annotations; "
        "do not upload the dataset to GitHub."
    )

with DATA_FILE.open(encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f))

X = df["question"].fillna("").astype(str)
y = df["answerable"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
}

results = []
for name, classifier in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("model", classifier),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(f"\n{name}")
    print(classification_report(
        y_test, predictions,
        target_names=["Unanswerable", "Answerable"],
        zero_division=0,
    ))
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "unanswerable_recall": (
            ((predictions == 0) & (y_test.to_numpy() == 0)).sum()
            / (y_test.to_numpy() == 0).sum()
        ),
    })
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Unanswerable", "Answerable"],
        cmap="RdPu", ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}")
    fig.savefig(
        OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

comparison = pd.DataFrame(results)
print("\nComparison:")
print(comparison.to_string(index=False))
comparison.to_csv(OUTPUT_DIR / "answerability_model_comparison.csv", index=False)

print("\nReminder: these models predict answerability, not actual VQA answers.")
