# Data preparation and EDA for VizWiz answerability

This part of the project prepares the **VizWiz VQA** data for the *answerable / unanswerable* task: given a photo and a question, can the question be answered from the photo?

It covers:

- loading and cleaning the train and validation examples (image, question, reference answers, label),
- data-integrity counts and image checks,
- reproducible samples,
- exploratory analysis (EDA) that extends, and does not repeat, the existing question-only baseline.

The deliverable is a single script: **`src/data_preparation.py`**. Sections 1 to 9 document it. Sections 10 to 12 document the surrounding work done locally (baseline, a first-pass analysis, helper scripts and a work log).

> **No data is stored in this repository.** The dataset and the generated outputs stay on your own machine (see [What must not go into Git](#what-must-not-go-into-git)).

---

## 1. Quick start

```bash
pip install pandas numpy matplotlib scikit-learn pillow
python src/data_preparation.py
```

Run it from the repository root. By default it expects the data in `data/` and writes everything to `outputs/data_preparation/`.

Developed and run on Windows with Anaconda (Python 3.13).

---

## 2. Data

The data comes from the VizWiz VQA page: <https://vizwiz.org/tasks-and-datasets/vqa/>. Use of the data is subject to the terms on that page.

In our case the downloads were:

| Download | What it contains | Used here |
|---|---|---|
| Annotations zip | `train.json`, `val.json`, `test.json` | `train.json`, `val.json` |
| `train.zip` | training photos (`VizWiz_train_*.jpg`) | yes |
| `val.zip` | validation photos (`VizWiz_val_*.jpg`) | yes |
| Test photos zip | test photos (`VizWiz_test_*.jpg`) | no |

Expected layout (nothing below `data/` is committed):

```
data/
├── train.json
├── val.json
├── images/         # training photos
└── images_val/     # validation photos
```

Each record in `train.json` and `val.json` has:

| Field | Meaning |
|---|---|
| `image` | file name of the photo |
| `question` | the question asked about the photo |
| `answers` | 10 crowd-sourced answers, each with `answer` and `answer_confidence` |
| `answer_type` | `unanswerable`, `yes/no`, `number` or `other` |
| `answerable` | the label: `1` = answerable, `0` = unanswerable |

Notes on the files we received:

- `test.json` has **8,000** records with only `image` and `question`. It has **no labels**, so it cannot be used to measure accuracy and this script does not read it.
- `train.zip` holds **23,954** photos, but `train.json` refers to only **20,523** of them. The remaining 3,431 are never read by the code, so they are harmless.

---

## 3. How to run

```bash
python src/data_preparation.py
python src/data_preparation.py --data-dir D:/vizwiz --val-images D:/vizwiz/val
python src/data_preparation.py --seed 7 --sample-size 200 --skip-image-stats
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | `data/` (or `VQA_DATA_DIR`) | folder with the JSON files and photos |
| `--train-json`, `--val-json` | `<data-dir>/train.json`, `<data-dir>/val.json` | annotation files |
| `--train-images` | `<data-dir>/images` | training photos |
| `--val-images` | `<data-dir>/images_val` if it exists, else `<data-dir>/images` | validation photos |
| `--output-dir` | `outputs/data_preparation` (or `VQA_OUTPUT_DIR`) | where results are written |
| `--seed` | `42` | seed for the samples |
| `--sample-size` | `500` | rows per split in the sample |
| `--skip-image-stats` | off | only check that photos exist and open; skip brightness and sharpness (faster) |
| `--keep-missing-images` | off | keep rows whose photo is missing or unreadable (they stay flagged) |

Environment variables `VQA_DATA_DIR` and `VQA_OUTPUT_DIR` set the two folder defaults. The script never copies or writes any photo.

---

## 4. What the script does

1. **Loads** `train.json` and `val.json` and checks that `image`, `question` and `answerable` exist.
2. **Cleans** each split and counts what it finds:
   - rows with missing required fields,
   - empty questions (after trimming extra spaces),
   - labels that are not 0 or 1,
   - the same image and question appearing twice with *different* labels (these rows are removed, because the label cannot be trusted),
   - exact duplicates (the first copy is kept).
3. **Keeps the reference answers.** The original `answers` field is never modified. The script adds:

   | Column | Meaning |
   |---|---|
   | `reference_answers` | the 10 answer texts, trimmed and lower-cased |
   | `num_answers` | how many answers the question has (10 expected) |
   | `top_answer` | the most common answer |
   | `answer_agreement` | share of answers equal to the top answer |
   | `unanswerable_share` | share of answers equal to "unanswerable" |
   | `conf_yes_share` | share of answers with confidence "yes" |

4. **Checks every photo:** it exists, it opens (corrupt files are detected), and its width, height, format and file size. Unless `--skip-image-stats` is used, it also measures:
   - **brightness**: mean grey level, 0 (black) to 255 (white),
   - **sharpness**: variance of the Laplacian, where a low value means blurry.

   Both are measured on a thumbnail of at most 256 pixels.
5. **Drops rows whose photo is missing or unreadable**, unless `--keep-missing-images` is used.
6. **Checks leakage** between train and validation: shared photos and shared image-question pairs.
7. **Writes** the prepared files, seeded samples, reports and plots.
8. **Prints** the integrity counts, the leakage result and a train-versus-validation comparison.

---

## 5. Outputs

Everything goes to `outputs/data_preparation/`:

| Path | Content |
|---|---|
| `prepared/train_prepared.json`, `.csv` | cleaned training set with all added columns |
| `prepared/val_prepared.json`, `.csv` | cleaned validation set |
| `samples/train_sample.json`, `val_sample.json` | seeded samples (500 rows each by default) that keep each split's answerable / unanswerable ratio |
| `reports/integrity_report.json` | settings used, integrity counts and the leakage result |
| `reports/split_comparison.csv` | train-versus-validation summary |
| `reports/image_stats.csv` | one row per photo (size, file size, brightness, sharpness, status) |
| `plots/` | 13 plots, listed below |

The prepared and sample files contain real dataset content, and the integrity report stores the local paths that were used. This is the reason `outputs/` must stay out of Git.

### Plots

| File | What it shows |
|---|---|
| `answers_agreement_by_class.png` | how much the 10 answers agree, per class |
| `answers_unanswerable_share_by_class.png` | share of annotators answering "unanswerable", per class |
| `answers_confidence_by_class.png` | share of "yes" confidence answers, per class |
| `answers_type_by_class.png` | answer type per class |
| `answers_top15.png` | the 15 most common top answers |
| `images_size_overview.png` | photo width, height and file size |
| `images_brightness_by_class.png` | brightness per class |
| `images_sharpness_by_class.png` | sharpness per class (log scale, low = blurry) |
| `questions_first_words.png` | the 15 most common first words and their answerable rate |
| `questions_top_words_by_class.png` | most common words per class |
| `questions_top_bigrams_by_class.png` | most common word pairs per class |
| `questions_word_log_odds.png` | words most linked to each class |
| `questions_feature_correlation.png` | simple question features versus the label |

The class-balance and question-length plots are left out on purpose, because the existing baseline already shows them.

---

## 6. Results on the VizWiz data

### Integrity

| | train | validation |
|---|---|---|
| rows in the file | 20,523 | 4,319 |
| missing values, empty questions, invalid labels | 0 | 0 |
| conflicting labels, duplicates | 0 | 0 |
| rows without answers, rows without exactly 10 answers | 0 | 0 |
| photos missing or unreadable | 0 | 0 |
| rows after cleaning | 20,523 | 4,319 |

- **Leakage:** 0 photos and 0 image-question pairs appear in both train and validation.
- **One question per photo:** every photo has exactly one question.

### Class balance

| | train | validation |
|---|---|---|
| answerable | 14,991 (73.0%) | 2,934 (67.9%) |
| unanswerable | 5,532 (27.0%) | 1,385 (32.1%) |

The classes are imbalanced. A model that always answered "answerable" would be right 73.0% of the time on train and 67.9% on validation, so any accuracy has to be judged against those numbers.

### Train versus validation

| | train | validation |
|---|---|---|
| mean words per question | 6.67 | 7.17 |
| mean answer agreement | 0.567 | 0.574 |
| mean share of "unanswerable" answers | 0.273 | 0.317 |
| mean brightness | 110.7 | 108.9 |
| words in validation never seen in train | n/a | 1.6% |

Validation has more unanswerable questions and slightly longer questions. The shift is small, but scores on validation can differ a little from scores on a random slice of train.

---

## 7. Findings

- **The question wording is a weak signal.** No simple question feature has a correlation with the label above about 0.14 in absolute value. Unanswerable questions are somewhat longer (mean 7.8 vs 6.3 words, median 6 vs 5). Question mark and a "wh" opening word go with answerable questions.
- **Words:**
  - Linked to *unanswerable*: cooking, sodium, expiration, serial, instructions, date, ounces, directions. These ask about small printed text, which a blurry photo cannot show.
  - Linked to *answerable*: clouds, sky, upside, design, color, shoes, hat, plant, black, stripes. These are easy to see.
  - For example, "cooking instructions" is in about 0.89% of unanswerable questions and 0.08% of answerable ones.
- **Image quality is a clearer signal than wording.**
  - *Sharpness:* unanswerable questions come with blurrier photos. Answerable photos peak around 6 to 6.5 on the log scale, while unanswerable ones sit lower and spread wider.
  - *Brightness:* the middle of the distribution is the same for both classes, but unanswerable photos have extra spikes at almost black (0) and almost white (255). The class averages hide this.
  - *Size:* most photos are portrait, roughly 1,000 to 1,250 px wide and 1,250 to 1,650 px tall (read from the histogram, so approximate).
- **Longer questions are less often answerable.** In train, the answerable rate falls from about 84% (1 to 3 words) to about 62% (13 or more words). Questions opening with "how" are the least answerable common opener (about 50%), and yes/no openers ("is", "are", "do") the most (about 78 to 87%).

---

## 8. Important cautions

**Several columns reveal the label, or are computed from the answers, and must never be model inputs:**
`answers`, `reference_answers`, `answer_type`, `top_answer`, `answer_agreement`, `unanswerable_share` and `conf_yes_share`.

For example, `answer_type == "unanswerable"` always means `answerable == 0`. The label is derived from the annotators' answers, so these columns describe how the labels were made. They also do not exist at prediction time, when a user only provides a photo and a question. They are kept in the files because the task asks for the reference answers to be retained.

Other points:

- **Sharpness is a proxy.** It is measured on a small thumbnail, and a plain, textureless photo can score low even if it is in focus.
- **"Most common answer" is a plurality.** A question with 3 "unanswerable" answers out of 10 different ones still has "unanswerable" on top, which is why that answer is the top answer for more questions than are labeled unanswerable.
- **Plots pool both splits.** The plots in `plots/` combine train and validation, so numbers can differ slightly from analyses on train alone.
- **Top-words plots show raw counts.** Because the classes have different sizes, compare words by their relative position and not by bar length. Percentages within each class would be a fairer view.
- **Testing status.** The full pipeline was run on the real data and passed every check above. The `--skip-image-stats` option was tested only on a small dummy dataset, and not on the full data.

---

## 9. What must not go into Git

The project rules forbid datasets, photos, model weights, secrets and virtual environments in the repository.

| Keep out | Why |
|---|---|
| `data/` | the dataset (already covered by the repository's `.gitignore`) |
| `outputs/` | prepared files hold full copies of the questions and answers, and the report stores local paths. **This folder is not yet in the `.gitignore`.** |

Suggested `.gitignore` addition:

```
# Generated results (prepared files contain dataset questions and answers)
outputs/
```

If the samples should be shared, ask the team first and check the VizWiz terms. A narrower rule can then keep only the samples folder:

```
outputs/*
!outputs/data_preparation/
outputs/data_preparation/*
!outputs/data_preparation/samples/
```

---

## 10. Local work around the deliverable (not part of the repository)

These files were used to build and check the work. They stay on the author's machine, and they contain local paths.

| File | Purpose |
|---|---|
| `baseline_model.py` | the existing question-only baseline, kept unchanged |
| `data_prep_eda.py` | a first-pass analysis of train and validation (described below) |
| `complete_task.py` | runs the baseline and then `data_prep_eda.py` with one command |
| `check_data.py` | prints rows, fields and answerable share of the three JSON files |
| `check_images_folder.py` | counts files in `data/images` and finds the ones not listed in `train.json` |
| `extract_images.py`, `extract_val.py` | unzip the train and validation photos into `data/images` and `data/images_val` |

### 10.1 Baseline (`baseline_model.py`)

It loads `train.json`, prints basic information, saves a class-distribution chart, and trains TF-IDF plus a logistic regression (balanced class weights) on the question text. It splits `train.json` 80/20 (stratified, seed 42) and tests on 4,105 questions.

| | precision | recall | F1 |
|---|---|---|---|
| unanswerable | 0.43 | 0.62 | 0.51 |
| answerable | 0.83 | 0.70 | 0.76 |

- Accuracy is **0.678** and the macro F1 is **0.63**.
- Confusion matrix (rows = true class): unanswerable 682 correct and 425 wrong, answerable 2,102 correct and 896 wrong.
- Accuracy is below the "always answer 'answerable'" rate of about 73%. This is expected, because the model sees only the question text, and answerability depends mostly on the photo.
- Its test split is its own random split of `train.json`. It is not the validation set in `val.json`.
- It saves `answerable_distribution.png` and `confusion_matrix.png` to `outputs/`.

### 10.2 First-pass analysis (`data_prep_eda.py`)

It works only with the question text and the labels, using the official `train.json` for train and `val.json` for validation (no re-splitting). It never uses `answers` or `answer_type`, because they reveal the label.

**What it does**

1. Loads and cleans both splits with the same checks as the deliverable (missing values, empty questions, invalid labels, conflicting labels, duplicates) and checks that each photo file exists.
2. Adds seven question features: `char_count`, `word_count`, `avg_word_len`, `starts_with_wh`, `has_question_mark`, `has_digit` and `uppercase_ratio`.
3. Prints the integrity counts, the class balance, the leakage check, the feature means by class, the words most linked to each class, a train-versus-validation table, and random example questions.
4. Saves 10 plots and four files.

**Settings:** `SHOW_PLOTS = False` saves the plots without opening windows. Set it to `True` to also open each plot, and the script then waits for you to close each window.

**Files it writes**

| Path | Content |
|---|---|
| `outputs/prepared/train.csv`, `val.csv` | image, question, cleaned question, label, whether the photo exists, and the seven features |
| `outputs/prepared/split_comparison.csv` | train-versus-validation summary |
| `outputs/prepared/class_balance.csv` | counts and percentages of both classes for train and validation |
| `outputs/eda_plots/` | the 10 plots below |

**Plots**

| File | What it shows |
|---|---|
| `01_class_share_train_vs_val.png` | class share in train and validation |
| `02_word_count_by_class.png` | words per question by class (long tail grouped at 30) |
| `03_char_count_by_class.png` | characters per question by class (long tail grouped at 200) |
| `04_word_count_boxplot.png` | word-count box plot with outliers hidden |
| `05_first_words.png` | the 15 most common first words and their answerable rate |
| `06_top_words_by_class.png` | most common words, as a percentage of each class's questions |
| `06_top_bigrams_by_class.png` | most common word pairs, as a percentage of each class's questions |
| `07_word_log_odds.png` | words most linked to each class (words seen at least 20 times) |
| `08_answerable_rate_by_length.png` | answerable rate for questions of 1-3, 4-5, 6-8, 9-12 and 13+ words |
| `09_feature_correlation.png` | correlation of each question feature with the label |

**What it showed** (on train, unless noted)

- Every correlation is small. The largest is `char_count` at about -0.14.
- Answerable rate by question length: 84% (1-3 words), 74% (4-5), 69% (6-8), 66% (9-12), 62% (13+).
- "what" opens about 70% of the questions and is 77% answerable; "how" is about 51% answerable.
- Class share: 73.0% answerable in train, 67.9% in validation.

### 10.3 Running everything (`complete_task.py`)

It runs `baseline_model.py` first (two plot windows open and must be closed), then `data_prep_eda.py`. The run finishes with "All done" and exit code 0.

### 10.4 Checks and setup helpers

- `check_data.py`: `train.json` has 20,523 labeled rows (73.0% answerable), `val.json` has 4,319 labeled rows (67.9%) and `test.json` has 8,000 rows without labels.
- `check_images_folder.py`: `data/images` holds 23,954 `.jpg` files, all 20,523 train photos are present, and the extra 3,431 are train-named photos that `train.json` does not use.
- `extract_images.py` and `extract_val.py`: unzip the photos so that they sit directly in `data/images` and `data/images_val`.

---

## 11. Possible next steps

- Evaluate the baseline on `val.json`, which has labels and shares no photos with train.
- Add photo features. Sharpness and brightness already show a difference between the classes.
- Show the top-words and word-pair plots as a share of each class's questions, and require a higher minimum word count in the log-odds plot (it currently includes some rare words).

---

## 12. Work log: what was done, in order

1. **Data setup.** `train.zip` turned out to hold only photos, so the annotations were downloaded separately. `train.json`, `val.json` and `test.json` went into `data/`, and the train photos into `data/images`. `test.json` has no labels and is not used.
2. **Baseline.** The existing question-only baseline was run and its results recorded (section 10.1).
3. **First-pass analysis.** `data_prep_eda.py` was written and run. After reviewing its figures it was rewritten:
   - it now uses the official `train.json` and `val.json` instead of re-splitting `train.json`,
   - the histograms use bins that match whole numbers, with the long tail grouped,
   - word and word-pair plots show percentages of each class,
   - a feature that was the same for every row (questions per photo) was dropped from the plots,
   - a class-balance table and percentage labels on the bars were added,
   - old plots and CSV files from the first version were deleted.
4. **The deliverable.** `src/data_preparation.py` was run on train and validation. The first run kept no validation rows, because the validation photos were not in the project yet. After `val.zip` was extracted into `data/images_val`, the run kept all 4,319 rows. Three small fixes were then made:
   - `--skip-image-stats` no longer crashes (the brightness and sharpness plots are skipped instead),
   - the three histograms of answer-based values use bins that match the steps of 0.1, which removed artificial gaps,
   - the progress messages now end on the full count (for example `20523/20523`).
5. **Review of the deliverable.** All 13 plots, the printed output and the sample files were checked. The samples have 500 rows each, keep the class ratio (0.73 and 0.68), and keep both the original `answers` and the cleaned `reference_answers`.
6. **Repository review.** The existing `.gitignore` already covers `data/` and the common Python and secret files, but not `outputs/` (section 9).
