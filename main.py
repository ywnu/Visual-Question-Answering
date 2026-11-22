import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

data_file = BASE_DIR / "data" / "train.json"

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------
with open(data_file, "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(data)


# -----------------------------
# Basic information
# -----------------------------
print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df[["image", "question", "answerable"]].head())

print("\nAnswerable distribution:")
print(df["answerable"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicates based on image and question:")
print(df.duplicated(subset=["image", "question"]).sum())

print("\nAnswerable percentage:")
print(df["answerable"].value_counts(normalize=True) * 100)


# -----------------------------
# Class distribution graph
# -----------------------------
df["answerable"].value_counts().sort_index().plot(kind="bar")

plt.title("Answerable vs Unanswerable")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(
    [0, 1],
    ["Unanswerable", "Answerable"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    output_dir / "answerable_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# Question features
# -----------------------------
df["question_length"] = df["question"].str.len()
df["word_count"] = df["question"].str.split().str.len()

print("\nQuestion features:")
print(
    df[
        ["question", "question_length", "word_count", "answerable"]
    ].head()
)

print("\nAverage question length by class:")
print(
    df.groupby("answerable")[
        ["question_length", "word_count"]
    ].mean()
)


# -----------------------------
# Features and target
# -----------------------------
X = df["question"]
y = df["answerable"]


# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------
# Baseline model
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test_tfidf)


# -----------------------------
# Model evaluation
# -----------------------------
print("\nBaseline Model Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Unanswerable",
            "Answerable"
        ]
    )
)


# -----------------------------
# Confusion Matrix
# -----------------------------
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=[
        "Unanswerable",
        "Answerable"
    ],
    cmap="RdPu"
)

plt.title("Confusion Matrix - Baseline Model")
plt.tight_layout()

plt.savefig(
    output_dir / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
