import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


# =============================
# 0. Settings
# =============================
SHOW_PLOTS = True  # True = also open each plot in a window (the script waits until you close it)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_JSON = DATA_DIR / "train.json"
VAL_JSON = DATA_DIR / "val.json"
TRAIN_IMAGES = DATA_DIR / "images"
VAL_IMAGES = DATA_DIR / "images_val" if (DATA_DIR / "images_val").exists() else DATA_DIR / "images"

OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "eda_plots"
PREP_DIR = OUTPUT_DIR / "prepared"
for d in (OUTPUT_DIR, PLOT_DIR, PREP_DIR):
    d.mkdir(parents=True, exist_ok=True)

REQUIRED = ["image", "question", "answerable"]
CLASS_NAMES = {0: "Unanswerable", 1: "Answerable"}
COLORS = {0: "#c44e80", 1: "#4c72b0"}
WH_WORDS = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}

# NOTE: 'answers' and 'answer_type' are NOT used here. They reveal the label
# (answer_type == "unanswerable" always means answerable == 0), so they must
# never be used as model features.


def save_fig(name):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / name, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# =============================
# 1. Load, clean and check one split
# =============================
def load_split(path, split):
    with open(path, "r", encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{split}: missing columns {missing} in {path}")
    df.insert(0, "split", split)
    return df


def clean_split(df, image_dir):
    report = {"rows_raw": len(df)}

    # missing values in the required columns
    report["missing_values"] = int(df[REQUIRED].isna().any(axis=1).sum())
    df = df.dropna(subset=REQUIRED).copy()

    # tidy the question text (keep the original too)
    df["question_raw"] = df["question"]
    df["question"] = df["question"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["question_clean"] = df["question"].str.lower()

    empty = df["question_clean"].str.len() == 0
    report["empty_questions"] = int(empty.sum())
    df = df[~empty].copy()

    # labels must be 0 or 1
    labels = pd.to_numeric(df["answerable"], errors="coerce")
    bad = ~labels.isin([0, 1])
    report["invalid_labels"] = int(bad.sum())
    df = df[~bad].copy()
    df["answerable"] = labels[~bad].astype(int)

    # duplicates: same image + question, with the same or a different label
    key = df["image"].astype(str) + "||" + df["question_clean"]
    conflict = df.groupby(key)["answerable"].transform("nunique") > 1
    report["conflicting_labels"] = int(conflict.sum())
    df, key = df[~conflict], key[~conflict]

    dup = key.duplicated(keep="first")
    report["duplicates"] = int(dup.sum())
    df = df[~dup].copy()

    # does each image file exist? (one folder listing, much faster than 20k exists() calls)
    on_disk = {p.name for p in image_dir.iterdir()} if image_dir.exists() else set()
    df["image_exists"] = df["image"].astype(str).map(lambda n: Path(n).name in on_disk)
    report["images_missing"] = int((~df["image_exists"]).sum())

    report["rows_final"] = len(df)
    report["answerable_rate"] = round(float(df["answerable"].mean()), 3) if len(df) else None
    return df.reset_index(drop=True), report


# =============================
# 2. Question features
# =============================
def add_features(df):
    df = df.copy()
    df["char_count"] = df["question"].str.len()
    df["word_count"] = df["question"].str.split().str.len()
    df["avg_word_len"] = df["char_count"] / df["word_count"].clip(lower=1)
    df["first_word"] = df["question_clean"].str.extract(r"^(\w+)")[0].fillna("")
    df["starts_with_wh"] = df["first_word"].isin(WH_WORDS).astype(int)
    df["has_question_mark"] = df["question"].str.contains(r"\?", regex=True).astype(int)
    df["has_digit"] = df["question"].str.contains(r"\d", regex=True).astype(int)
    df["uppercase_ratio"] = df["question"].apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    return df


FEATURE_COLS = ["char_count", "word_count", "avg_word_len", "starts_with_wh",
                "has_question_mark", "has_digit", "uppercase_ratio"]


# =============================
# 3. Load both splits
# =============================
print("Loading data...")
frames, reports = {}, {}
for split, (jpath, ipath) in {"train": (TRAIN_JSON, TRAIN_IMAGES), "val": (VAL_JSON, VAL_IMAGES)}.items():
    print(f"  {split}: {jpath.name}  |  images: {ipath}")
    clean, rep = clean_split(load_split(jpath, split), ipath)
    frames[split] = add_features(clean)
    reports[split] = rep

train, val = frames["train"], frames["val"]
both = pd.concat([train, val], ignore_index=True)

print("\n=== DATA INTEGRITY COUNTS ===")
print(f"{'':22s}{'train':>10}{'val':>10}")
for key in reports["train"]:
    print(f"{key:22s}{reports['train'][key]!s:>10}{reports['val'][key]!s:>10}")

# class balance: counts and percentages, printed and saved
class_counts = pd.DataFrame(
    {s: f["answerable"].map(CLASS_NAMES).value_counts() for s, f in frames.items()}
).reindex([CLASS_NAMES[0], CLASS_NAMES[1]])
class_percent = (class_counts / class_counts.sum() * 100).round(1)
class_balance = pd.concat({"count": class_counts, "percent": class_percent}, axis=1)
class_balance.columns = [f"{split}_{kind}" for kind, split in class_balance.columns]
class_balance.index.name = "class"
print("\n=== CLASS BALANCE ===")
print(class_balance.to_string())
class_balance.to_csv(PREP_DIR / "class_balance.csv")

# leakage between train and validation
pairs = lambda d: set(zip(d["image"], d["question_clean"]))
print("\n=== LEAKAGE CHECK (train vs val) ===")
print("Images in both splits:         ", len(set(train["image"]) & set(val["image"])))
print("Image+question pairs in both:  ", len(pairs(train) & pairs(val)))
print("Questions per image (train):   ", round(len(train) / train["image"].nunique(), 2),
      "(1.0 means every image has exactly one question)")

print("\n=== FEATURE MEANS BY CLASS (train) ===")
print(train.groupby("answerable")[FEATURE_COLS].mean().round(3).T)


# =============================
# 4. EDA
# =============================

# 4.1 Class share, train vs validation
share = pd.DataFrame({s: f["answerable"].value_counts(normalize=True).sort_index() * 100
                      for s, f in frames.items()})
ax = share.T.plot(kind="bar", color=[COLORS[0], COLORS[1]], figsize=(7, 5), rot=0)
ax.legend([CLASS_NAMES[0], CLASS_NAMES[1]])
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.1f%%")
plt.title("Class Share: Train vs Validation")
plt.ylabel("% of questions")
plt.xlabel("")
save_fig("01_class_share_train_vs_val.png")


# 4.2 Words and characters per question by class (whole-number bins, long tail grouped)
def hist_by_class(df, col, bins, cap, title, xlabel, fname):
    plt.figure(figsize=(8, 5))
    for label in (0, 1):
        vals = df.loc[df["answerable"] == label, col].clip(upper=cap)
        plt.hist(vals, bins=bins, alpha=0.6, density=True,
                 color=COLORS[label], label=CLASS_NAMES[label])
    plt.title(title)
    plt.xlabel(f"{xlabel} (values above {cap} are grouped at {cap})")
    plt.ylabel("Share of questions")
    plt.legend()
    save_fig(fname)


hist_by_class(train, "word_count", np.arange(0.5, 31.5, 1), 30,
              "Words per Question by Class", "Words", "02_word_count_by_class.png")
hist_by_class(train, "char_count", np.arange(0, 205, 5), 200,
              "Characters per Question by Class", "Characters", "03_char_count_by_class.png")

# 4.3 Box plot of word count (extreme outliers hidden so the boxes are readable)
plt.figure(figsize=(6, 5))
plt.boxplot([train.loc[train["answerable"] == k, "word_count"] for k in (0, 1)], showfliers=False)
plt.xticks([1, 2], [CLASS_NAMES[0], CLASS_NAMES[1]])
medians = train.groupby("answerable")["word_count"].median()
plt.title(f"Word Count by Class (median: {medians[0]:.0f} vs {medians[1]:.0f}, outliers hidden)")
plt.ylabel("Words")
save_fig("04_word_count_boxplot.png")

# 4.4 First word of the question and its answerable rate
top_first = train["first_word"].value_counts().head(15).index
fw = (train[train["first_word"].isin(top_first)].groupby("first_word")["answerable"]
      .agg(["count", "mean"]).sort_values("count"))
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(fw.index, fw["count"], color="#8da0cb")
for i, (cnt, rate) in enumerate(zip(fw["count"], fw["mean"])):
    ax.text(cnt, i, f"  {rate:.0%} answerable", va="center")
ax.set_xlim(0, fw["count"].max() * 1.25)
ax.set_xlabel("Number of questions")
ax.set_title("Top 15 First Words and Their Answerable Rate")
save_fig("05_first_words.png")


# 4.5 Most common words and word pairs per class, as % of that class's questions
def top_terms(texts, ngram, n=15):
    vec = CountVectorizer(stop_words="english", ngram_range=(ngram, ngram), binary=True)
    mat = vec.fit_transform(texts)
    pct = pd.Series(mat.sum(axis=0).A1, index=vec.get_feature_names_out()) / mat.shape[0] * 100
    return pct.sort_values(ascending=False).head(n)


for ngram, name in [(1, "words"), (2, "bigrams")]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, label in zip(axes, (0, 1)):
        terms = top_terms(train.loc[train["answerable"] == label, "question_clean"], ngram)
        terms.sort_values().plot(kind="barh", ax=ax, color=COLORS[label])
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f%%", padding=3)
        ax.margins(x=0.15)
        ax.set_title(f"Top {name} - {CLASS_NAMES[label]}")
        ax.set_xlabel("% of this class's questions containing it")
    save_fig(f"06_top_{name}_by_class.png")

# 4.6 Words most linked to each class (smoothed log-odds, words seen at least 20 times)
vec = CountVectorizer(stop_words="english", min_df=20)
mat = vec.fit_transform(train["question_clean"])
vocab = np.array(vec.get_feature_names_out())
y = train["answerable"].values
c1 = mat[y == 1].sum(axis=0).A1 + 1
c0 = mat[y == 0].sum(axis=0).A1 + 1
log_odds = np.log(c1 / c1.sum()) - np.log(c0 / c0.sum())
order = np.argsort(log_odds)
k = min(12, len(vocab) // 2)
sel = np.concatenate([order[:k], order[-k:]])
print("\nWords most linked to UNANSWERABLE:", list(vocab[order[:10]]))
print("Words most linked to ANSWERABLE:  ", list(vocab[order[-10:][::-1]]))
plt.figure(figsize=(8, 7))
plt.barh(vocab[sel], log_odds[sel], color=[COLORS[0]] * k + [COLORS[1]] * k)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Words Most Linked to Each Class (log-odds)")
plt.xlabel("<- Unanswerable      Answerable ->")
save_fig("07_word_log_odds.png")

# 4.6b Answerable rate by question length
buckets = pd.cut(train["word_count"], [0, 3, 5, 8, 12, np.inf],
                 labels=["1-3", "4-5", "6-8", "9-12", "13+"])
by_len = train.groupby(buckets, observed=True)["answerable"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(by_len.index.astype(str), by_len["mean"] * 100, color="#8da0cb")
for i, (m, c) in enumerate(zip(by_len["mean"], by_len["count"])):
    ax.text(i, m * 100, f"{m:.0%}\n(n={c})", ha="center", va="bottom")
ax.axhline(train["answerable"].mean() * 100, color="black", linestyle="--", linewidth=0.8)
ax.set_ylim(0, 100)
ax.set_title("Answerable Rate by Question Length (dashed = overall rate)")
ax.set_xlabel("Words in the question")
ax.set_ylabel("% answerable")
save_fig("08_answerable_rate_by_length.png")

# 4.7 Correlation of each question feature with the label
corr = train[FEATURE_COLS + ["answerable"]].corr()["answerable"].drop("answerable").dropna()
plt.figure(figsize=(7, 5))
corr.sort_values().plot(kind="barh", color="#8da0cb")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Question Features vs 'answerable' (correlation)")
plt.xlabel("Pearson correlation")
save_fig("09_feature_correlation.png")

# 4.8 Train vs validation summary
vocab_train = set(train["question_clean"].str.findall(r"\w+").explode().dropna())
rows = {}
for name, d in frames.items():
    toks = d["question_clean"].str.findall(r"\w+").explode().dropna()
    rows[name] = {
        "questions": len(d),
        "answerable_rate": d["answerable"].mean(),
        "mean_words": d["word_count"].mean(),
        "mean_chars": d["char_count"].mean(),
        "share_with_question_mark": d["has_question_mark"].mean(),
        "share_starting_with_wh": d["starts_with_wh"].mean(),
        "words_not_seen_in_train": 0.0 if name == "train" else float((~toks.isin(vocab_train)).mean()),
    }
comparison = pd.DataFrame(rows)
print("\n=== TRAIN VS VALIDATION ===")
print(comparison.round(3).to_string())
comparison.to_csv(PREP_DIR / "split_comparison.csv")

print("\nExample questions (random, per class):")
for label in (0, 1):
    print(f"\n{CLASS_NAMES[label]}:")
    for q in train[train["answerable"] == label].sample(5, random_state=42)["question"]:
        print("  -", q[:120])


# =============================
# 5. Save the prepared data
# =============================
keep = ["image", "question", "question_clean", "answerable", "image_exists"] + FEATURE_COLS
train[keep].to_csv(PREP_DIR / "train.csv", index=False)
val[keep].to_csv(PREP_DIR / "val.csv", index=False)

print(f"\nSaved prepared files to: {PREP_DIR}")
print(f"Saved plots to:          {PLOT_DIR}")
for label in (0, 1):
    qs = train.loc[train["answerable"] == label, "question_clean"]
    print(CLASS_NAMES[label], round(qs.str.contains("cooking instructions").mean() * 100, 2), "%")