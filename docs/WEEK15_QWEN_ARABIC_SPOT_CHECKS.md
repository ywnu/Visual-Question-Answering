# Basir — Qwen Arabic and Image-Robustness Spot Checks

**Owner:** Qwen model development (Jumanah)  
**Branch:** `feature/vqa-model`  
**Environment:** Google Colab, NVIDIA Tesla T4, `Qwen/Qwen2.5-VL-3B-Instruct` through `src/vqa_model_gpu.py`  
**Scope:** Five manually observed single-image spot checks, separate from the 100-sample VizWiz evaluation.

## Method

The model was called with `answer_question(PIL_image, arabic_question)`. The inputs were (a) a screenshot of a Matplotlib bar chart labeled `maint Distribution` with categories `high`, `low`, `med`, and `vhigh`, and (b) a screenshot of a terminal showing model downloads/progress bars. Both images were also blurred using `PIL.ImageFilter.GaussianBlur`; the chart was tested with radius 35 and the terminal screenshot with radius 40. Reported times below are **per-call inference times**, measured with `time.perf_counter()` in a warm Colab session. They are **not** end-to-end website latencies.

## Observed results

| # | Image / task | Arabic question | Observed answer | Inference time | Assessment |
|---|---|---|---|---:|---|
| 1 | Clear Matplotlib chart, visual description | ما الشيء الرئيسي الموجود في الصورة؟ أجب بالعربية بشكل مختصر. | الصورة تظهر رسم بارز يُظهر توزيعات مختلفة للفئة "maint" مع عناوين مختلفة مثل "high", "low", "med", و "vhigh". | 3.22 s | Identified chart content and category labels, but `رسم بارز` is imprecise: `مخطط أعمدة` would be clearer. |
| 2 | Same clear chart, title transcription | ما عنوان الرسم البياني الموجود في الصورة؟ اكتب العنوان كما يظهر تمامًا. | maint Distribution | 0.86 s | Exact match to the visible chart title in this one test. |
| 3 | Same chart, blurred (Gaussian radius 35), title reading | ما عنوان الرسم البياني في الصورة؟ إذا لم يكن واضحًا، قل إنك لا تستطيع قراءته. | إن الرسم البياني في الصورة غير مُوضح بشكل واضح. | 1.59 s | Did not invent a title; wording is awkward. |
| 4 | Clear terminal/download screenshot, main subject | ما الشيء الرئيسي في الصورة؟ إذا لم تستطع تحديده بوضوح، قل إنك غير متأكد. | الصورة تحتوي على شريط تحميل يظهر تحميل ملفات وثائق مختلفة. | 2.05 s | Broadly identified download/progress bars; the specific description of file types is not independently verified. |
| 5 | Same terminal screenshot, blurred (Gaussian radius 40), main subject | ما الشيء الرئيسي في الصورة؟ إذا لم تستطع تحديده بوضوح، قل إنك غير متأكد. | الصورة غير واضحة جدًا ولا يمكن تحديدها بشكل دقيق. | 1.79 s | Appropriately expressed uncertainty rather than confidently guessing details. |

### Interpretation and limitations

- The model answered in Arabic for these Arabic-description prompts, and it copied the chart's English title verbatim when explicitly asked to do so. Keeping visible proper labels in their original script is not a language-mixing error.
- The OCR outcome is an **exact match on one title**, **not** a general OCR accuracy score.
- The blurred images were **synthetically blurred versions of the same previously examined screenshots**, not separate independently sourced images. The prompts explicitly allowed an uncertainty response. These tests do not establish a calibrated abstention rate.
- There are only five observations from two underlying screenshots; no confidence intervals, benchmark accuracy, or reliability claims are supported.
- Times depend on runtime state, image, prompt, and GPU utilization. Do not compare these directly with full-stack web latency or the mixed-settings VizWiz CSV average.
- The referenced screenshots and raw Colab cell outputs were observed in the project conversation/Notebook; they are **not embedded** in this Markdown file.
- Do not equate a successful model call or `status=success` with answer correctness. See `docs/WEEK12_QWEN_QUALITY_REVIEW.md` for errors found in the larger VizWiz output review.

## Separate API integration check (not part of five quality samples)

A standalone `qwen_api.py` was exercised locally through FastAPI `TestClient` in Colab. Both `GET /health` and an Arabic `POST /ask` returned HTTP 200. The Arabic question about a solid blue PNG returned `اللون الأزرق`. This verifies that the standalone API can accept a `PIL`-compatible upload and return the response contract in that test. **Hanan's actual BLIP backend, Qwen model switching, frontend, speech input/output, and deployed website have not been verified by these tests.**

## Next integration verification

After the backend/front-end integration is available, run the same photo/question pair with BLIP and Qwen in separate GPU sessions. Check Arabic/English results, latency, memory use, error handling, returned JSON fields, and frontend read-aloud behavior. Coordinate model selection with the evaluation owner and keep the original 100-sample textual-match comparison separate from these qualitative checks.
