# Week 12 — BLIP Evaluation Runbook

## Purpose

Run BLIP against the same fixed 100 image-question pairs prepared by Aseel in:

`week12/evaluation/evaluation_sample_100_checked.csv`

The sample contains 100 rows: 68 marked answerable and 32 marked unanswerable. Keep every row. Do not filter the unanswerable cases out of the evaluation.

## What the evaluator records

- Image filename and question
- Answerable / unanswerable label
- The ten reference answers supplied in the sample
- BLIP's raw answer
- Normalized Exact Match against any supplied reference answer
- Response time, HTTP status, model/device, and request errors

Exact Match uses lowercase + whitespace normalization, matching the earlier Week 11 comparison protocol. It is a simple string metric, not official VizWiz VQA accuracy and not a semantic correctness score. Review model answers against the actual images before making a final model decision.

## Files

- `experiments/evaluate_blip_week12.py` — fixed-sample BLIP evaluator and preflight checker.
- `outputs/blip_week12_100.csv` — per-question results.
- `outputs/blip_week12_100_summary.json` — aggregate and subgroup metrics.

## Setup

The checked CSV currently lives on Aseel's `feature/data` branch, while this evaluator is on `feature/evaluation`. From the repository root, while on `feature/evaluation`, bring the sample file into the working tree:

```bash
git checkout feature/data -- week12/evaluation/evaluation_sample_100_checked.csv
```

Set the local image folder if it differs from the previous Week 11 location:

**Windows PowerShell**
```powershell
$env:BASIR_IMAGE_DIR = "D:\ML_Datasets\val\val"
```

Install the request dependency if needed:

```bash
python -m pip install requests
```

## Step 1 — preflight (no model/API needed)

```bash
python experiments/evaluate_blip_week12.py --check-only
```

Expected sample counts: 100 total, 68 answerable, 32 unanswerable. The preflight must find all 100 image files before the full run.

## Step 2 — run BLIP

Start Hanan's FastAPI service and wait until `GET /health` reports that the model is loaded. Then run:

```bash
python experiments/evaluate_blip_week12.py
```

If the API is hosted elsewhere:

```bash
python experiments/evaluate_blip_week12.py --api-url "http://127.0.0.1:8000/ask"
```

The script checks the health endpoint first, writes one output row per sample item, and flushes results as it goes. Missing images and failed requests remain visible as failed rows; they are never silently dropped.

## Step 3 — compare with Qwen

Jumanah's Qwen run must use the same 100 image filenames, questions, reference answers, and answerable labels. Ask for a CSV with one row per sample item and at least these columns:

`image, question, answerable, qwen_answer, response_seconds, status, error`

Keep the raw answer text. Do not compare BLIP and Qwen using different samples or different reference sets.

## Final comparison

Once both runs exist, calculate:
- Completed requests and failures
- Exact Match overall
- Exact Match on answerable items
- Exact Match on unanswerable items
- Mean response time for completed requests
- Output-quality issues (empty, punctuation-only, mixed-language, unsupported details)
- Human-reviewed semantic correctness on a shared subset

Do not declare a winner from Exact Match or latency alone. The earlier Week 11 report already notes that the BLIP and Qwen timing setup used different hardware and execution paths; any speed comparison must state the hardware and inference settings.
