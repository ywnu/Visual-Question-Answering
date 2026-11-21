# Basir Week 12 evaluation: BLIP and Qwen

## Setup

Both models answered the same fixed VizWiz validation sample of 100 image–question pairs. The sample includes 68 items labeled answerable and 32 labeled unanswerable. The comparison script aligned the outputs by image filename and verified the question text against the shared sample.

- **BLIP:** `Salesforce/blip-vqa-base`, served by the local FastAPI backend on CPU.
- **Qwen:** `Qwen/Qwen2.5-VL-3B-Instruct`, run in Google Colab on an NVIDIA Tesla T4. The submitted output combines 57 predictions at the original processor settings and 43 retries with a lower maximum input image resolution following CUDA out-of-memory failures.
- **Exact Match:** The comparison script lowercases answers and normalizes whitespace, then checks whether the full prediction equals any one reference answer. This is a string-match metric. The four image-level observations below illustrate answer quality separately.

## Results

| Measure | BLIP | Qwen |
| --- | ---: | ---: |
| Completed requests | 100/100 | 100/100 |
| Missing results | 0 | 0 |
| Empty answers | 0 | 0 |
| Exact Match, all items | 31/100 (31%) | 5/100 (5%) |
| Exact Match, answerable items | 27/68 (39.71%) | 5/68 (7.35%) |
| Exact Match, unanswerable items | 4/32 (12.50%) | 0/32 (0.00%) |
| Mean recorded response time | 0.784 s | 3.085 s |

BLIP led by 26 percentage points on Exact Match. The timing values describe the recorded runs on different hardware and with different execution setups. A controlled latency comparison would use a consistent measurement procedure and one fixed configuration per model.

## Visual review of four selected cases

The four cases were selected to inspect different types of disagreement. They are qualitative examples, so the visual judgments do not estimate an accuracy rate for the 100-item sample.

| Image | Question focus | BLIP (Exact Match) | Qwen (Exact Match) | Visual finding |
| --- | --- | --- | --- | --- |
| `VizWiz_val_00000642.jpg` | Identify the object | `laptop` (1) | A full description of a laptop and keyboard (0) | Both identified the laptop. BLIP answered concisely. Qwen's mention of a possible logo is uncertain because the screen is overexposed. The longer description did not match a full reference string. |
| `VizWiz_val_00003670.jpg` | Identify the coffee variety | `starbucks` (0) | `Caffè Verona` (0) | The capsule shows the Starbucks logo and the Caffè Verona name. BLIP identified the brand. Qwen correctly identified the variety. The closest reference uses `caffe verona`, and the matcher preserves accents. |
| `VizWiz_val_00001598.jpg` | Identify the product in the can | `purple juice` (0) | `Dettol` (0) | The photo shows a purple aerosol can with part of the Lysol logo visible. All ten references identify Lysol. BLIP misidentified the product category and Qwen misidentified the brand. |
| `VizWiz_val_00000339.jpg` | Locate Safe Mode relative to the highlight and count the steps | `no` (1) | Above the current highlight, without a step count (0) | In the rotated view, the plain Safe Mode entry is eight selectable items above the highlight. The dataset labels this item unanswerable. BLIP matched a minority reference string, while Qwen gave the correct direction and omitted the count. |

The visual review shows why the report keeps Exact Match and image-level judgment separate. A correct description or accented spelling can score zero. A matching reference string can also earn a point without providing the information the user requested.

## Reproducibility and evidence

- [Shared validation sample](https://github.com/ywnu/Visual-Question-Answering/blob/50798b242a46a113bef22c5ad886c650349aefa3/week12/evaluation/evaluation_sample_100_checked.csv)
- [Per-image predictions, references, and scores](../outputs/blip_qwen_week12_comparison.csv)
- [Comparison summary](../outputs/blip_qwen_week12_summary.json)
- [BLIP predictions](../outputs/blip_week12_100.csv) and [BLIP summary](../outputs/blip_week12_100_summary.json)
- [Qwen predictions](https://github.com/ywnu/Visual-Question-Answering/blob/3c53b1d8361958bc520360841f7b8a59451f6a28/outputs/qwen_week12_100_FINAL.csv) and [Qwen run notes](https://github.com/ywnu/Visual-Question-Answering/blob/3c53b1d8361958bc520360841f7b8a59451f6a28/docs/WEEK12_QWEN_EVALUATION.md)
- [Evaluation runbook](WEEK12_EVALUATION_RUNBOOK.md) and [comparison script](compare_week12_models.py)
