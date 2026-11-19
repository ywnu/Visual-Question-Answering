# Basir — Week 12 Qwen VQA Evaluation

## Status and artifacts

- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Evaluation set:** Aseel's fixed VizWiz validation sample, `feature/data:week12/evaluation/evaluation_sample_100_checked.csv`.
- **Sample:** 100 image–question pairs; 68 labeled answerable and 32 labeled unanswerable.
- **Final predictions:** [`outputs/qwen_week12_100_FINAL.csv`](../outputs/qwen_week12_100_FINAL.csv), committed on `feature/vqa-model`.
- **Execution status:** 100/100 rows have `status=success`, distinct `id` values, and nonempty model answers.
- **Evaluation script:** [`evaluate_qwen.py`](../evaluate_qwen.py). It outputs `id,image,question,qwen_answer,inference_seconds,status`; the final merged handoff CSV also includes `reference_answers,answerable,response_seconds,error`.

**Important:** 100/100 success is an **execution-completion rate**, not model correctness or VizWiz accuracy. No final Qwen-vs-BLIP quality winner has been established from this output.

## Environment and model configuration

The evaluation was run in Google Colab using a **Tesla T4 GPU** (approximately 14.56 GiB GPU memory reported in CUDA OOM messages), CUDA, and FP16 model weights. The model's generation settings were `max_new_tokens=80` and `do_sample=False`. The earlier Colab tests used Transformers 4.57.6 and eager attention. Exact package versions in the final 100-question run were **not captured** and must be pinned before a reproducibility claim.

**Crucial distinction:** The committed `src/vqa_model.py` is still the CPU-oriented version (`dtype=torch.float32` without `device_map="cuda"`). The Colab checkout was patched *locally* for `dtype=torch.float16`, `device_map="cuda"`, `inputs.to("cuda")`, and the chat-template system-message compatibility adjustment. The Colab GPU patch is **not committed** as a production implementation, so checking out this branch and running the evaluator without those changes does **not** reproduce the T4 evaluation setup.

## How the 100 rows were obtained

1. Downloaded Aseel's fixed 100-question CSV from `feature/data`. Added a stable 16-character SHA-256-derived ID of `image + "\\n" + question` for each row; retained the original question, reference answers, answerability label, and row order.
2. Downloaded and extracted the VizWiz validation images into `/content/vizwiz_data/val`. Verified all 100 corresponding JPG files existed and passed PIL image verification.
3. **Initial run:** 57 successful predictions and 43 CUDA out-of-memory failures on the T4, with the processor's original/default image settings.
4. **Retry run:** in the *local Colab copy* of `src/vqa_model.py`, initialized the processor with `min_pixels=256 * 28 * 28` and `max_pixels=512 * 28 * 28`; retried the 43 failures. All 43 retries returned `status=success`.
5. Combined the 57 original successes and 43 retry successes, matched by stable `id` and restored Aseel's original row order. The final file preserves all 100 rows, their ten reference answers, answerability labels, raw predictions, response times, statuses, and errors.

**Mixed-settings caveat:** **57 rows used the original processor settings and 43 rows used reduced maximum image resolution.** The final CSV does not include a per-row processor-setting column. These runs must not be presented as one uniformly configured experiment. For a controlled comparison with BLIP, rerun all 100 rows using **one fixed Qwen configuration** and archive the uniform results separately.

## Suggested single-configuration rerun (Colab)

This is a procedural outline rather than an exact notebook snapshot:

1. Select a T4 GPU in a fresh Colab runtime; install and pin compatible versions of PyTorch, Transformers, Accelerate, Pillow, and other runtime requirements. Record `pip freeze` and the GPU model.
2. Clone `feature/vqa-model`; apply the documented local GPU and message-template compatibility patches. Use the **same processor setting for all 100 cases**, initially `min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28`.
3. Build an input CSV with columns `id,image,question` using the same stable IDs and the same 100 images; keep the reference-answer/answerability columns for later joining.
4. Do **not** import the model into the notebook's parent Python process and then run it again in a subprocess; this previously caused duplicate GPU memory allocation.
5. Run the evaluator **in its own process**:

```bash
python evaluate_qwen.py --input /content/basir_eval_100.csv --images-dir /content/vizwiz_data/val --output /content/qwen_week12_100_uniform.csv
```

6. Check **every row's** `status` and `qwen_answer`; the evaluator catches row-level exceptions, so a zero process exit code alone does not prove successful inference. Join predictions to the source sample by `id` and preserve all 100 rows.
7. Record the exact model revision, package versions, processor settings, inference hardware, timing method, and any failures.

## Evaluation caveats and next handoff

- Compare predictions with reference answers on **the identical 100 image–question pairs** used by BLIP; keep all 32 unanswerable examples.
- Naive normalized exact match is **not** the official VizWiz VQA scoring protocol, and neither metric alone captures semantic correctness. Human-review a representative subset and document unsupported details, incorrect object identifications, and mixed-language output.
- Some generated responses in the retry preview asserted specific CAPTCHA values. Treat such answers as **unverified** until checked against the actual image; the text-generation result is not proof that the answer is visually supported.
- `response_seconds` measures the per-question call reported by the evaluator, **not model startup or first weight download**. Memory failures and reruns introduce selection effects into aggregate latency.
- **Deem:** use the final CSV as a provisional complete-predictions handoff, with the mixed-processor-settings limitation clearly disclosed. Prefer a uniform rerun for formal Qwen-vs-BLIP comparison.
- **Hanan:** GPU model loading and FastAPI `/ask` integration remain to be verified; no integrated API success is asserted here.
