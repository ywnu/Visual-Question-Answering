"""
Data preparation for the answerable / unanswerable VQA task.

What it does
------------
1. Loads train.json and val.json (image, question, answers, answerable, ...).
2. Cleans the examples and reports data-integrity counts (missing values,
   empty questions, invalid labels, duplicates, conflicting labels).
3. Checks every image: exists, can be opened, size, file size, brightness, sharpness.
4. Keeps the reference answers (the original `answers` field is never modified)
   and adds answer-level features.
5. Writes prepared train / validation files and reproducible (seeded) samples.
6. Runs EDA that EXTENDS the baseline EDA: answers, image quality, question-text
   patterns (first words, top words / word pairs, words linked to each class)
   and a train-vs-validation comparison. Class balance and question-length
   plots from the baseline are not repeated.

Paths are configurable (command-line flags or environment variables), and no
image is copied or written anywhere, so nothing image-related needs to go to GitHub.

Usage
-----
    python src/data_preparation.py
    python src/data_preparation.py --data-dir D:/vizwiz --val-images D:/vizwiz/val
    python src/data_preparation.py --seed 7 --sample-size 200 --skip-image-stats

Environment variables (optional): VQA_DATA_DIR, VQA_OUTPUT_DIR
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_COLUMNS = ["image", "question", "answerable"]
EXPECTED_ANSWERS = 10          # VizWiz has 10 crowd-sourced answers per question
WH_WORDS = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}
CLASS_NAMES = {0: "Unanswerable", 1: "Answerable"}
COLORS = {0: "#c44e80", 1: "#4c72b0"}
TENTH_BINS = np.linspace(-0.05, 1.05, 12)   # one bin per value 0.0, 0.1, ..., 1.0


# =============================================================
# Configuration
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="VQA data preparation")
    p.add_argument("--data-dir", type=Path,
                   default=Path(os.environ.get("VQA_DATA_DIR", PROJECT_ROOT / "data")),
                   help="Folder that contains train.json, val.json and the images")
    p.add_argument("--train-json", type=Path, default=None)
    p.add_argument("--val-json", type=Path, default=None)
    p.add_argument("--train-images", type=Path, default=None,
                   help="Folder with training images (default: <data-dir>/images)")
    p.add_argument("--val-images", type=Path, default=None,
                   help="Folder with validation images "
                        "(default: <data-dir>/images_val if it exists, else <data-dir>/images)")
    p.add_argument("--output-dir", type=Path,
                   default=Path(os.environ.get("VQA_OUTPUT_DIR",
                                               PROJECT_ROOT / "outputs" / "data_preparation")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=500,
                   help="Rows per split in the reproducible sample")
    p.add_argument("--skip-image-stats", action="store_true",
                   help="Only check that images exist / open; skip brightness and sharpness")
    p.add_argument("--keep-missing-images", action="store_true",
                   help="Keep rows whose image is missing or unreadable (they stay flagged)")
    args = p.parse_args()

    args.train_json = args.train_json or args.data_dir / "train.json"
    args.val_json = args.val_json or args.data_dir / "val.json"
    args.train_images = args.train_images or args.data_dir / "images"
    if args.val_images is None:
        alt = args.data_dir / "images_val"
        args.val_images = alt if alt.exists() else args.data_dir / "images"
    return args


# =============================================================
# Loading and answers
# =============================================================
def load_split(json_path, split):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"{split}: annotation file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{split}: missing required columns {missing} in {json_path}")
    if "answers" not in df.columns:
        print(f"WARNING [{split}]: no 'answers' column, reference answers cannot be retained.")
    df.insert(0, "split", split)
    return df


def parse_answers(answers):
    """Turn the raw answers list into simple features. The raw list is left untouched."""
    empty = {"reference_answers": [], "num_answers": 0, "top_answer": None,
             "answer_agreement": np.nan, "unanswerable_share": np.nan, "conf_yes_share": np.nan}
    if not isinstance(answers, list) or not answers:
        return empty
    texts = [str(a.get("answer", "")).strip().lower() for a in answers if isinstance(a, dict)]
    confs = [str(a.get("answer_confidence", "")).strip().lower()
             for a in answers if isinstance(a, dict)]
    if not texts:
        return empty
    top, top_n = Counter(texts).most_common(1)[0]
    return {
        "reference_answers": texts,
        "num_answers": len(texts),
        "top_answer": top,
        "answer_agreement": top_n / len(texts),
        "unanswerable_share": sum(t == "unanswerable" for t in texts) / len(texts),
        "conf_yes_share": sum(c == "yes" for c in confs) / len(confs) if confs else np.nan,
    }


def add_answer_features(df):
    if "answers" in df.columns:
        feats = pd.DataFrame(list(df["answers"].apply(parse_answers)), index=df.index)
    else:
        feats = pd.DataFrame(
            [parse_answers(None)] * len(df), index=df.index)
    return pd.concat([df, feats], axis=1)


# =============================================================
# Cleaning + integrity counts
# =============================================================
def clean_split(df, split):
    report = {"split": split, "rows_raw": int(len(df))}
    df = df.copy()

    for c in REQUIRED_COLUMNS:
        report[f"missing_{c}"] = int(df[c].isna().sum())
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    df["question_raw"] = df["question"]
    df["question"] = (df["question"].astype(str).str.strip()
                      .str.replace(r"\s+", " ", regex=True))
    df["question_clean"] = df["question"].str.lower()

    empty = df["question_clean"].str.len() == 0
    report["empty_questions"] = int(empty.sum())
    df = df[~empty].copy()

    labels = pd.to_numeric(df["answerable"], errors="coerce")
    bad = ~labels.isin([0, 1])
    report["invalid_labels"] = int(bad.sum())
    df = df[~bad].copy()
    df["answerable"] = labels[~bad].astype(int)

    key = df["image"].astype(str) + "||" + df["question_clean"]
    conflict = df.groupby(key)["answerable"].transform("nunique") > 1
    report["conflicting_label_rows"] = int(conflict.sum())
    df, key = df[~conflict], key[~conflict]

    dup = key.duplicated(keep="first")
    report["exact_duplicate_rows"] = int(dup.sum())
    df = df[~dup].copy()

    df = add_answer_features(df)
    report["rows_without_answers"] = int((df["num_answers"] == 0).sum())
    report["rows_unexpected_answer_count"] = int(
        ((df["num_answers"] > 0) & (df["num_answers"] != EXPECTED_ANSWERS)).sum())

    report["rows_after_cleaning"] = int(len(df))
    return df.reset_index(drop=True), report


# =============================================================
# Image checks
# =============================================================
def inspect_image(path, compute_quality):
    if not path.exists():
        return {"image_status": "missing"}
    try:
        info = {"image_status": "ok", "file_kb": path.stat().st_size / 1024}
        with Image.open(path) as im:
            im.verify()                       # detects truncated / corrupt files
        with Image.open(path) as im:
            info.update({"width": im.size[0], "height": im.size[1],
                         "format": im.format, "mode": im.mode})
            if compute_quality:
                im.draft("L", (256, 256))     # fast JPEG decoding; ignored for other formats
                gray = im.convert("L")
                gray.thumbnail((256, 256))
                arr = np.asarray(gray, dtype=np.float32)
                info["brightness"] = float(arr.mean())
                if arr.shape[0] >= 3 and arr.shape[1] >= 3:
                    lap = (-4 * arr[1:-1, 1:-1] + arr[:-2, 1:-1] + arr[2:, 1:-1]
                           + arr[1:-1, :-2] + arr[1:-1, 2:])
                    info["sharpness"] = float(lap.var())   # low value = blurry
        return info
    except Exception as exc:
        return {"image_status": "unreadable", "error": type(exc).__name__}


def check_images(df, image_dir, compute_quality, split):
    names = df["image"].astype(str).unique()
    rows = []
    for i, name in enumerate(names, 1):
        info = inspect_image(Path(image_dir) / Path(name).name, compute_quality)
        info["image"] = name
        rows.append(info)
        if i % 2000 == 0 or i == len(names):
            print(f"  [{split}] checked {i}/{len(names)} images")
    img_df = pd.DataFrame(rows)
    if "image_status" not in img_df.columns:
        img_df["image_status"] = pd.Series(dtype=str)
    return img_df


# =============================================================
# Samples and exports
# =============================================================
def stratified_sample(df, n, seed):
    """Seeded sample that keeps the answerable / unanswerable ratio."""
    if len(df) <= n:
        return df.copy()
    frac = n / len(df)
    parts = [g.sample(n=max(1, round(len(g) * frac)), random_state=seed)
             for _, g in df.groupby("answerable")]
    return pd.concat(parts).sort_index()


def to_json(df, path):
    df.to_json(path, orient="records", force_ascii=False, indent=2)


def to_csv(df, path):
    out = df.copy()
    for c in ("answers", "reference_answers"):
        if c in out.columns:
            out[c] = out[c].apply(lambda v: json.dumps(v, ensure_ascii=False))
    out.to_csv(path, index=False, encoding="utf-8")


# =============================================================
# EDA that extends the baseline
# =============================================================
def hist_by_class(df, col, title, xlabel, path, bins=30):
    if col not in df.columns:
        return
    data = df.dropna(subset=[col])
    if data.empty:
        return
    plt.figure(figsize=(7, 4.5))
    for label in (0, 1):
        vals = data.loc[data["answerable"] == label, col]
        if len(vals):
            plt.hist(vals, bins=bins, alpha=0.6, density=True,
                     color=COLORS[label], label=CLASS_NAMES[label])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def run_eda(df, img_df, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- answers ----
    hist_by_class(df, "answer_agreement", "Agreement Between the 10 Answers",
                  "Share of answers equal to the most common answer",
                  plot_dir / "answers_agreement_by_class.png", bins=TENTH_BINS)
    hist_by_class(df, "unanswerable_share", "Share of Annotators Answering 'unanswerable'",
                  "Share of answers = 'unanswerable'",
                  plot_dir / "answers_unanswerable_share_by_class.png", bins=TENTH_BINS)
    hist_by_class(df, "conf_yes_share", "Annotator Confidence ('yes') per Question",
                  "Share of answers with confidence = yes",
                  plot_dir / "answers_confidence_by_class.png", bins=TENTH_BINS)

    if "answer_type" in df.columns:
        ct = pd.crosstab(df["answerable"].map(CLASS_NAMES), df["answer_type"], normalize="index")
        ct.plot(kind="bar", stacked=True, figsize=(7, 4.5), rot=0)
        plt.title("Answer Type by Class")
        plt.ylabel("Share")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(plot_dir / "answers_type_by_class.png", dpi=200, bbox_inches="tight")
        plt.close()

    top = df["top_answer"].value_counts().head(15).sort_values()
    if len(top):
        top.plot(kind="barh", figsize=(7, 5), color="#8da0cb")
        plt.title("15 Most Common Top Answers")
        plt.xlabel("Questions")
        plt.tight_layout()
        plt.savefig(plot_dir / "answers_top15.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ---- images ----
    ok = img_df[img_df["image_status"] == "ok"]
    if not ok.empty and "width" in ok.columns:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(ok["width"].dropna(), bins=30, color="#8da0cb")
        axes[0].set_title("Image Width (px)")
        axes[1].hist(ok["height"].dropna(), bins=30, color="#8da0cb")
        axes[1].set_title("Image Height (px)")
        axes[2].hist(ok["file_kb"].dropna(), bins=30, color="#8da0cb")
        axes[2].set_title("File Size (KB)")
        plt.tight_layout()
        plt.savefig(plot_dir / "images_size_overview.png", dpi=200, bbox_inches="tight")
        plt.close()

    hist_by_class(df, "brightness", "Image Brightness by Class",
                  "Mean grey level (0 = black, 255 = white)",
                  plot_dir / "images_brightness_by_class.png")
    if "sharpness" in df.columns:
        tmp = df.assign(log_sharpness=np.log1p(df["sharpness"]))
        hist_by_class(tmp, "log_sharpness", "Image Sharpness by Class (low = blurry)",
                      "log(1 + variance of Laplacian)",
                      plot_dir / "images_sharpness_by_class.png")

    # ---- question text (only plots the baseline does not already have) ----
    run_question_eda(df, plot_dir)


def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def run_question_eda(df, plot_dir):
    """Question-text patterns. Class balance and length plots are left to the baseline."""
    d = df.copy()
    d["first_word"] = d["question_clean"].str.extract(r"^(\w+)")[0].fillna("")

    # first word of the question + share answerable
    top = d["first_word"].value_counts().head(15).index
    stats = (d[d["first_word"].isin(top)].groupby("first_word")["answerable"]
             .agg(["count", "mean"]).sort_values("count"))
    if len(stats):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(stats.index, stats["count"], color="#8da0cb")
        for i, (cnt, rate) in enumerate(zip(stats["count"], stats["mean"])):
            ax.text(cnt, i, f"  {rate:.0%} answerable", va="center")
        ax.set_xlabel("Number of questions")
        ax.set_title("Top 15 First Words and Their Answerable Rate")
        _savefig(plot_dir / "questions_first_words.png")

    # most common words and word pairs per class
    for ngram, name in [(1, "words"), (2, "bigrams")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        drew = False
        for ax, label in zip(axes, (0, 1)):
            texts = d.loc[d["answerable"] == label, "question_clean"]
            try:
                vec = CountVectorizer(stop_words="english", ngram_range=(ngram, ngram))
                mat = vec.fit_transform(texts)
            except ValueError:          # empty vocabulary
                continue
            freq = pd.Series(mat.sum(axis=0).A1, index=vec.get_feature_names_out())
            freq.sort_values(ascending=False).head(15).sort_values().plot(
                kind="barh", ax=ax, color=COLORS[label])
            ax.set_title(f"Top {name} - {CLASS_NAMES[label]}")
            ax.set_xlabel("Frequency")
            drew = True
        if drew:
            _savefig(plot_dir / f"questions_top_{name}_by_class.png")
        else:
            plt.close()

    # words most linked to each class (smoothed log-odds)
    try:
        vec = CountVectorizer(stop_words="english", min_df=5)
        mat = vec.fit_transform(d["question_clean"])
    except ValueError:
        mat = None
    if mat is not None and mat.shape[1] >= 4:
        vocab = np.array(vec.get_feature_names_out())
        c1 = mat[d["answerable"].values == 1].sum(axis=0).A1 + 1
        c0 = mat[d["answerable"].values == 0].sum(axis=0).A1 + 1
        log_odds = np.log(c1 / c1.sum()) - np.log(c0 / c0.sum())
        order = np.argsort(log_odds)
        k = min(12, len(vocab) // 2)
        sel = np.concatenate([order[:k], order[-k:]])
        plt.figure(figsize=(8, 7))
        plt.barh(vocab[sel], log_odds[sel], color=[COLORS[0]] * k + [COLORS[1]] * k)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("Words Most Linked to Each Class (log-odds)")
        plt.xlabel("<- Unanswerable      Answerable ->")
        _savefig(plot_dir / "questions_word_log_odds.png")

    # simple question features vs label
    f = pd.DataFrame({
        "word_count": d["question"].str.split().str.len(),
        "starts_with_wh": d["first_word"].isin(WH_WORDS).astype(int),
        "has_question_mark": d["question"].str.contains(r"\?", regex=True).astype(int),
        "has_digit": d["question"].str.contains(r"\d", regex=True).astype(int),
        "uppercase_ratio": d["question"].apply(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)),
        "answerable": d["answerable"],
    })
    corr = f.corr()["answerable"].drop("answerable").dropna()
    if len(corr):
        corr.sort_values().plot(kind="barh", figsize=(7, 4), color="#8da0cb")
        plt.title("Question Features vs 'answerable' (correlation)")
        plt.xlabel("Pearson correlation")
        _savefig(plot_dir / "questions_feature_correlation.png")


def split_comparison(frames):
    train = frames.get("train")
    train_vocab = set(train["question_clean"].str.findall(r"\w+").explode().dropna()) \
        if train is not None and not train.empty else set()
    rows = []
    for name, d in frames.items():
        if d.empty:
            continue
        toks = d["question_clean"].str.findall(r"\w+").explode().dropna()
        row = {"split": name, "questions": len(d), "images": d["image"].nunique(),
               "answerable_rate": d["answerable"].mean(),
               "mean_words": d["question"].str.split().str.len().mean(),
               "mean_questions_per_image": len(d) / d["image"].nunique()}
        for c in ("answer_agreement", "unanswerable_share", "brightness", "sharpness"):
            if c in d.columns:
                row[f"mean_{c}"] = d[c].mean()
        if name != "train" and train_vocab:
            row["oov_token_rate_vs_train"] = float((~toks.isin(train_vocab)).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


# =============================================================
# Main
# =============================================================
def main():
    args = parse_args()
    out = args.output_dir
    dirs = {k: out / k for k in ("prepared", "samples", "reports", "plots")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    splits = {"train": (args.train_json, args.train_images),
              "val": (args.val_json, args.val_images)}

    frames, img_frames, reports = {}, {}, []
    for split, (json_path, image_dir) in splits.items():
        print(f"\n=== {split.upper()} ===")
        print(f"annotations: {json_path}\nimages:      {image_dir}")
        raw = load_split(json_path, split)
        df, report = clean_split(raw, split)

        img_df = check_images(df, image_dir, not args.skip_image_stats, split)
        df = df.merge(img_df, on="image", how="left")
        df["questions_per_image"] = df.groupby("image")["question"].transform("count")

        report["images_checked"] = int(len(img_df))
        report["images_missing"] = int((img_df["image_status"] == "missing").sum())
        report["images_unreadable"] = int((img_df["image_status"] == "unreadable").sum())
        report["rows_with_bad_image"] = int((df["image_status"] != "ok").sum())

        if not args.keep_missing_images:
            df = df[df["image_status"] == "ok"].copy()
        df = df.reset_index(drop=True)
        report["rows_final"] = int(len(df))
        report["answerable_rate_final"] = float(df["answerable"].mean()) if len(df) else None

        if df.empty:
            print(f"WARNING [{split}]: no usable rows. Check the image folder "
                  f"({image_dir}) or run with --keep-missing-images.")

        frames[split] = df
        img_frames[split] = img_df
        reports.append(report)

    # ---- leakage between splits ----
    leakage = {"images_in_both_splits": 0, "image_question_pairs_in_both_splits": 0}
    if not frames["train"].empty and not frames["val"].empty:
        leakage["images_in_both_splits"] = len(
            set(frames["train"]["image"]) & set(frames["val"]["image"]))
        pairs = lambda d: set(zip(d["image"], d["question_clean"]))
        leakage["image_question_pairs_in_both_splits"] = len(
            pairs(frames["train"]) & pairs(frames["val"]))

    # ---- exports ----
    for split, d in frames.items():
        to_json(d, dirs["prepared"] / f"{split}_prepared.json")
        to_csv(d, dirs["prepared"] / f"{split}_prepared.csv")
        sample = stratified_sample(d, args.sample_size, args.seed) if len(d) else d
        to_json(sample, dirs["samples"] / f"{split}_sample.json")

    all_images = pd.concat(
        [i.assign(split=s) for s, i in img_frames.items()], ignore_index=True)
    all_images.to_csv(dirs["reports"] / "image_stats.csv", index=False)

    comparison = split_comparison(frames)
    comparison.to_csv(dirs["reports"] / "split_comparison.csv")

    full = {
        "config": {"train_json": str(args.train_json), "val_json": str(args.val_json),
                   "train_images": str(args.train_images), "val_images": str(args.val_images),
                   "seed": args.seed, "sample_size": args.sample_size,
                   "expected_answers": EXPECTED_ANSWERS,
                   "versions": {"pandas": pd.__version__, "numpy": np.__version__,
                                "pillow": PIL.__version__}},
        "integrity": reports,
        "leakage": leakage,
    }
    with open(dirs["reports"] / "integrity_report.json", "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)

    # ---- EDA ----
    combined = pd.concat([d for d in frames.values() if not d.empty], ignore_index=True)
    if not combined.empty:
        run_eda(combined, all_images, dirs["plots"])

    # ---- console summary ----
    print("\n=== DATA INTEGRITY COUNTS ===")
    print(f"{'':32s}" + "".join(f"{r['split']:>12}" for r in reports))
    for key in [k for k in reports[0] if k != "split"]:
        vals = [r.get(key) for r in reports]
        vals = [round(v, 3) if isinstance(v, float) else v for v in vals]
        print(f"{key:32s}" + "".join(f"{v!s:>12}" for v in vals))
    print("\n=== LEAKAGE BETWEEN TRAIN AND VAL ===")
    print(leakage)
    print("\n=== TRAIN VS VAL ===")
    print(comparison.round(3).T.to_string())
    print(f"\nEverything saved to: {out}")


if __name__ == "__main__":
    main()