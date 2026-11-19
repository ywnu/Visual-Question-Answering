# Week 12 -- Sample Preparation and Validation

## 1. Overview

This document describes the sample-preparation work completed for Week
12. It covers the two sampling stages, the scripts used, the main
functions and methods, generated files, and the actual verification
results.

The work uses **Stratified Sampling** to select records while keeping
the proportion of the `answerable` classes as close as possible to the
original dataset distribution.

> **Important distinction:** `Stratified Sampling` is the name of the
> sampling technique. `stratified_sample()` is the Python function that
> implements this technique. It is not a separate Python file in this
> workflow.

## 2. Workflow Summary

There are two separate sampling stages:

1.  `data_preparation.py` prepares the Train and Validation datasets and
    creates a 500-record sample from each split.
2.  `prepare_eval_sample.py` creates a separate 100-record evaluation
    sample directly from the full prepared Validation dataset
    (`val_prepared.json`), after validating the images.

The 100-record evaluation sample is **not** selected from the earlier
500-record `val_sample.json`.

``` text
Original Train and Validation JSON files
                 |
                 v
       data_preparation.py
       - Loads and prepares Train and Validation data
       - Cleans records and validates image files
       - Saves the prepared datasets
       - Generates a 500-record stratified sample for each split
                 |
        +--------+---------+
        |                  |
        v                  v
 train_sample.json   val_sample.json
        |                  |
        +--------+---------+
                 |
                 v
        chick_samplesjson.py
        Verifies 500 records in each file


val_prepared.json (4,319 records)
                 |
                 v
       prepare_eval_sample.py
       - Validate required fields
       - Validate image files
       - Stratified sample (100 records)
                 |
                 v
 evaluation_sample_100_checked.json
 evaluation_sample_100_checked.csv
```

## 3. Stage 1 -- Preparing the 500-Record Samples

### Script: `data_preparation.py`

This script prepares the Train and Validation data. As part of its
workflow, it creates prepared output files and uses the
`stratified_sample()` function to generate a reproducible sample for
each split.

**Sampling configuration:**

  Setting                                         Value
  ------------------------------- ---------------------
  Default sample size per split                     500
  Random seed                                        42
  Sampling technique                Stratified Sampling
  Stratification field                     `answerable`

The function groups the records by `answerable`, calculates a
proportional sample size for each class, distributes any remaining
records according to the largest fractional remainders, samples from
each group, and combines the selected records.

**Generated sample files:**

  ------------------------------------------------------------------------------------------
  Output                                                 Description
  ------------------------------------------------------ -----------------------------------
  `outputs/data_preparation/samples/train_sample.json`   500 sampled Train records

  `outputs/data_preparation/samples/val_sample.json`     500 sampled Validation records
  ------------------------------------------------------------------------------------------

The script also saves the prepared datasets under
`outputs/data_preparation/prepared/`, including `train_prepared.json`
and `val_prepared.json`. The prepared Validation dataset contains 4,319
records and is used as the input for the second sampling stage.

### Verifying the 500-record samples

#### Script: `chick_samplesjson.py`

This is a verification script. It opens `train_sample.json` and
`val_sample.json`, loads their JSON records, and prints the number of
records in each file.

It **does not** create samples and does **not** perform Stratified
Sampling.

**Actual output:**

``` text
train_sample.json : 500 records
val_sample.json : 500 records

Process finished with exit code 0
```

**Verification result:**

  File                    Expected records   Actual records Result
  --------------------- ------------------ ---------------- --------
  `train_sample.json`                  500              500 Passed
  `val_sample.json`                    500              500 Passed

## 4. Stage 2 -- Preparing the 100-Record Evaluation Sample

### Script: `prepare_eval_sample.py`

This script creates a separate evaluation sample from the prepared
Validation dataset.

**Input:**

`outputs/data_preparation/prepared/val_prepared.json`

The input contains 4,319 records. The script validates the required
columns and checks whether the referenced image files exist and can be
opened. It then selects 100 records from the valid records using
Stratified Sampling.

**Important:** The script reads from `val_prepared.json` directly. It
does not use `val_sample.json` as its input.

**Configuration:**

  Setting                              Value
  -------------------- ---------------------
  Sample size                            100
  Random seed                             42
  Sampling technique     Stratified Sampling
  Image directory          `data/images_val`

### Main checks performed

-   Confirms that the prepared Validation JSON file exists.
-   Checks that the required fields are present: `image`, `question`,
    `reference_answers`, and `answerable`.
-   Checks that `answerable` contains only the expected class values
    (`0` and `1`).
-   Checks that each `reference_answers` value is a non-empty list.
-   Checks that each referenced image exists and can be opened and
    loaded.
-   Selects 100 records while preserving the class proportions as
    closely as possible.
-   Checks for duplicate `(image, question)` pairs before saving.

### Actual execution results

``` text
=== IMAGE VALIDATION ===
Total records checked: 4319
image_status
ok    4319
Name: count, dtype: int64

Valid records: 4319
Excluded records: 0

=== EVALUATION SAMPLE ===
Total records: 100
Seed: 42

Class distribution:
answerable
0    32
1    68

Saved files:
outputs/evaluation/evaluation_sample_100_checked.json
outputs/evaluation/evaluation_sample_100_checked.csv

Process finished with exit code 0
```

**Results summary:**

  Check / Result                      Value
  --------------------------------- -------
  Validation records checked          4,319
  Images successfully validated       4,319
  Excluded records                        0
  Final evaluation sample               100
  Unanswerable (`answerable = 0`)        32
  Answerable (`answerable = 1`)          68
  Random seed                            42

### Generated evaluation files

  ---------------------------------------------------------------------------------------------
  Output                                                    Description
  --------------------------------------------------------- -----------------------------------
  `outputs/evaluation/evaluation_sample_100_checked.json`   Evaluation sample in JSON format

  `outputs/evaluation/evaluation_sample_100_checked.csv`    Evaluation sample in CSV format
  ---------------------------------------------------------------------------------------------

The `reference_answers` field remains a list in the JSON output. For the
CSV output, the list is serialized as a JSON string so it can be stored
in a CSV cell.

## 5. Important Functions and Methods

The following table summarizes the main custom functions and
Python/library methods used in the two sampling stages and the
sample-size verification script.

### Custom functions

  -------------------------------------------------------------------------------------
  Function                           Script                     Purpose
  ---------------------------------- -------------------------- -----------------------
  `stratified_sample(df, n, seed)`   `data_preparation.py`      Selects a reproducible
                                                                sample while preserving
                                                                the `answerable` class
                                                                proportions.

  `stratified_sample(df, n, seed)`   `prepare_eval_sample.py`   Selects the 100-record
                                                                evaluation sample while
                                                                preserving class
                                                                proportions.

  `validate_image(image_name)`       `prepare_eval_sample.py`   Checks whether an image
                                                                filename is valid,
                                                                whether the file
                                                                exists, and whether the
                                                                image can be opened and
                                                                loaded.

  `main()`                           Both preparation scripts   Runs the corresponding
                                                                preparation and
                                                                sampling workflow.

  `parse_args()`                     `data_preparation.py`      Reads command-line
                                                                configuration,
                                                                including the sample
                                                                size and random seed.
  -------------------------------------------------------------------------------------

### Common Python and library methods

  -----------------------------------------------------------------------
  Method                              Purpose
  ----------------------------------- -----------------------------------
  `groupby("answerable")`             Groups records by their
                                      answerability class before
                                      sampling.

  `sample(n=..., random_state=...)`   Randomly selects the required
                                      number of records from a group.

  `pd.concat()`                       Combines the sampled class groups
                                      into one DataFrame.

  `sort_index()`                      Restores index order after
                                      combining sampled records.

  `pd.read_json()`                    Loads prepared JSON data into a
                                      DataFrame.

  `Image.open()`                      Opens an image file for validation.

  `img.verify()`                      Checks image file integrity.

  `img.load()`                        Loads image data to confirm it can
                                      be read.

  `json.load()`                       Reads JSON files in
                                      `chick_samplesjson.py`.

  `len(data)`                         Counts the records in a loaded
                                      sample file.

  `to_json()` / `to_csv()`            Saves prepared or sampled data in
                                      JSON and CSV formats.

  `value_counts()`                    Counts records in each `answerable`
                                      class for the evaluation output.
  -----------------------------------------------------------------------

## 6. Final Output Summary

  ---------------------------------------------------------------------------------------------------------
  Stage            Script                               Output size Main output files
  ---------------- -------------------------- --------------------- ---------------------------------------
  Prepared Train   `data_preparation.py`                        500 `train_sample.json`
  sample                                                            

  Prepared         `data_preparation.py`                        500 `val_sample.json`
  Validation                                                        
  sample                                                            

  Evaluation       `prepare_eval_sample.py`                     100 `evaluation_sample_100_checked.json`,
  sample                                                            `evaluation_sample_100_checked.csv`

  Sample-size      `chick_samplesjson.py`               Checks both Console verification output
  verification                                     500-record files 
  ---------------------------------------------------------------------------------------------------------

## 7. Key Notes

-   Stratified Sampling is a sampling technique implemented through a
    Python function; it is not a standalone script in this workflow.
-   The first stage creates two separate 500-record samples: one from
    Train and one from Validation.
-   The second stage selects 100 evaluation records directly from the
    4,319 records in `val_prepared.json`, after image validation.
-   The 100-record evaluation sample is not a subset of the 500-record
    `val_sample.json`.
-   `chick_samplesjson.py` only verifies the record counts of the two
    500-record sample files.
-   The reported successful image checks confirm that the files could be
    found and read by the validation script. They do not represent a
    manual review of image content.
