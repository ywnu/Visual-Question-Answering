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

## Verified Colab GPU test environment (20 November 2026)

The user reported these versions **after successfully running the freshly fetched `src/vqa_model_gpu.py` module** on a VizWiz image. This is a verified post-evaluation smoke-test environment; it does not establish that the initial 57 predictions were generated with the exact same library versions.

| Component | Version / detail |
| --- | --- |
| PyTorch | `2.14.1+cu130` |
| Transformers | `4.57.6` |
| Pillow | `11.3.0` |
| Accelerate | `1.15.0` |
| PyTorch CUDA runtime | `13.0` |
| CUDA available | `True` |
| GPU | NVIDIA Tesla T4 |

**GitHub GPU smoke-test result:** In a separate subprocess the module loaded on the T4, answered `VizWiz_val_00000642.jpg` (`What is in this picture?`) with a description of a laptop and keyboard, and exited with code 0. Startup emitted nonfatal warnings about an optional CUDA driver stub, fast image processor defaults, and an ignored generation flag. This shows successful model import and single-image inference, **not** full API integration or a measured accuracy score.

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
- **Hanan (status at Week 12):** GPU model loading and FastAPI `/ask` integration had not yet been verified at the time of this report. See the later Week 15 standalone API test below.

## Later update — standalone Qwen FastAPI validation (Week 15, 27 November 2026)

This section updates the integration status; it **does not change the historical Week 12 evaluation procedure** or the mixed-settings limitation above.

- The committed, CUDA-enabled GPU implementation is now available as [`src/vqa_model_gpu.py`](../src/vqa_model_gpu.py). It is separate from the original `src/vqa_model.py`. The Week 12 text referring to uncommitted local patches describes the **original evaluation run**, not the current GPU integration file.
- A standalone FastAPI application is committed as [`qwen_api.py`](../qwen_api.py). Its `lifespan` loads the real `src.vqa_model_gpu.answer_question` when CUDA is available. The application was exercised in a Google Colab Tesla T4 session with `with TestClient(app) as client:`, which invokes startup/lifespan. **No mock or stubbed model was used**.
- **Observed health response:** HTTP `200`; `{"status": "healthy", "model_loaded": true, "model": "Qwen/Qwen2.5-VL-3B-Instruct", "device": "cuda"}`.
- **Observed POST /ask response:** HTTP `200`; request used a **programmatically created solid-blue 448×448 PNG**, not a natural/photo image. The question was `ما لون الصورة؟ أجب بكلمة عربية واحدة.`; the model answer was `اللون الأزرق`. The returned JSON included `success: true`, `model_connected: true`, `device: "cuda"`, `image_filename: "test.png"`, and `image_info: {format: "PNG", width: 448, height: 448}`. The user reported these outputs from the Colab test; they have not been captured as an independent CI or committed test log.
- Separate **real-world screenshot** tests (chart and terminal/download screen) were run by directly calling the Qwen model, including Arabic questions, OCR, and synthetically blurred variants; see [Week 15 Arabic spot checks](WEEK15_QWEN_ARABIC_SPOT_CHECKS.md). They **were not posted through the committed FastAPI route**.
- **Unverified:** integration with Hanan's `feature/backend/vqa_api.py`, switching Qwen/BLIP in the existing backend, external HTTP serving, frontend integration, voice features, and end-to-end tests with uploaded real-world photos on the final integrated system.

For backend integration, run only one GPU-heavy model at a time on a T4, preserve the frontend's `/ask` request/response contract, and verify real image uploads and error paths on the merged backend before considering integration complete.
