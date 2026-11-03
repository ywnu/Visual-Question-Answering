# Basir — Preliminary BLIP vs Qwen2.5-VL Evaluation

**Week:** 11 · **Date:** 4 Nov 2026 · **Responsible:** Deem Almaneea

## Objective
Compare image-question answering behavior and preliminary Exact Match outcomes of BLIP and Qwen on a shared VizWiz validation sample. This is a progress review, not a final model selection.

## Protocol
- 20 randomly selected (seed 42) **answerable** VizWiz validation image-question pairs; questions are English only.
- Both models received the same sample filenames and question texts, compared to the same ten reference answer strings per item.
- Lowercase + whitespace-normalized **exact string match** to *any* reference answer. No synonym-based, semantic or official VizWiz VQA scoring.
- BLIP: `Salesforce/blip-vqa-base` on **CPU** through FastAPI `POST /ask`; timing includes local request overhead.
- Qwen: `Qwen/Qwen2.5-VL-3B-Instruct` in Google Colab **Tesla T4**, FP16, SDPA attention; original image files were passed without manual resizing; 80 new tokens maximum. Timing is around generation with GPU synchronization.
- Qwen's GitHub source still uses CPU FP32/eager. The evaluation ran a **temporary Colab copy** with GPU/FP16/SDPA; the model-development branch was not edited.

## Quantitative results
| Measure | BLIP | Qwen2.5-VL |
|---|---:|---:|
| Predictions completed | 20 / 20 | 20 / 20 |
| Punctuation-only unusable output | 0 | 1 |
| Exact matches | **6 / 20** | **3 / 20** |
| Exact-match rate | **30.0%** | **15.0%** |
| Mean measured response time | **0.83 s** | **4.51 s** |
| Hardware | CPU | T4 GPU |

## Qualitative observations (reference-answer alignment, not visual verification)
- For `What medicine is this?`, BLIP gives `advil` while Qwen gives `Advil Cold & Sinus`. Both are reference-aligned although only BLIP earns exact-string credit.
- For the CD title, Qwen mentions `Jazz Collection` but gives a longer title. The reference answers all say `jazz collection`; Qwen exact-match remains zero.
- Qwen references a Starz premium promotion where BLIP answered `star trek`; several references mention `starz`.
- Qwen identifies the Apple logo on the iPad, matching reference concepts, but exact-match is zero.
- Both models identify `Sony` correctly in the stereo question.
- Qwen produced repeated exclamation points for the wine-bottle question, a clear output-quality failure; BLIP answered `wine`.
- Qwen's longer screen/object descriptions can contain unsupported details and require human checking; one soup subtype differs from the reference wording.

## Interpretation and limitations
1. **Do not conclude BLIP is generally more accurate.** The strict string metric strongly penalizes long, informative answers; it cannot determine semantic correctness.
2. **No controlled speed ranking.** BLIP ran on CPU via API; Qwen ran directly on a T4 GPU.
3. **Limited sample.** Only 20 answerable English examples; this excludes the unanswerable class, Arabic scoring and a larger validation evaluation.
4. **Reference text is not image verification.** Qualitative alignment must be checked by human review of images before assigning a semantic accuracy score.
5. **Model reliability.** One Qwen output contained only punctuation; separate Arabic smoke tests showed mixed-language output.

## Conclusion / next steps
BLIP is currently integrated with Basir's FastAPI/Frontend prototype. Qwen has now been evaluated successfully on Colab T4 using FP16 + SDPA, without manually resizing originals. A final selection requires expanded image-grounded review, Arabic tests, unanswerable examples, matching hardware/inference setups, and integration feasibility.

**Results:** [20-sample table](../outputs/BLIP_QWEN_COMPARISON_TABLE.md) · [raw comparison CSV](../outputs/qwen_evaluation.csv) · [BLIP baseline report](BLIP_EVALUATION_REPORT.md).
