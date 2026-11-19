# Basir — Qwen Week 12 Output Quality Review

## Scope and interpretation

This review examines the **100 stored Qwen responses** in [`outputs/qwen_week12_100_FINAL.csv`](../outputs/qwen_week12_100_FINAL.csv), against the corresponding question text and VizWiz reference-answer strings. It is **not a visual inspection of all 100 source photographs**, nor a formal accuracy score. Some apparent mismatches need checking against the original image. For the dataset, `answerable=0` means the answer was judged not reliably answerable, rather than that an answer is necessarily impossible.

The file combines **57 original-resolution successes** with **43 reduced-resolution retries**. Do not treat it as a controlled uniform-settings experiment. All 100 rows have `status=success`, but this is **execution success only**.

## Descriptive statistics (100 stored responses)

| Measure | Value | Interpretation |
| --- | ---: | --- |
| Rows with `status=success` | 100/100 | Successful execution; **not** correct-answer rate |
| Empty/NaN raw answers | 0 | Nonempty text does not guarantee useful output |
| Median response length | 14 whitespace-separated words | Based on raw answer strings |
| Mean response length | 23.15 words | Based on raw answer strings |
| Answers longer than 30 words | 30/100 | Potential verbosity for voice output |
| Answers longer than 50 words | 18/100 | Potentially lengthy for an accessibility-focused interface |
| Answers matching uncertainty phrase heuristic | 17/100 | Text contains terms such as *unsure*, *unclear*, *cannot*, *unable*, *too blurry*; not a calibrated uncertainty metric |
| Answers matching hedging phrase heuristic | 29/100 | Terms such as *appears*, *seems*, *likely*, *possibly*, *could be*; overlaps with uncertainty group |
| Responses containing Arabic characters | 0/100 | All evaluation questions are in English; this **does not evaluate Arabic-language handling** |
| Responses containing Chinese Han characters | 0/100 | Only describes these stored English-question results |
| Mean per-question inference time | 3.0846 seconds | Mixed runs; excludes model-load time |
| Median per-question inference time | 2.969 seconds | Mixed runs; excludes model-load time |
| Fastest / slowest inference time | 0.542 / 11.485 seconds | Mixed runs; not comparable to BLIP without hardware and configuration context |

All text categories above are heuristic review flags, **not a verified count of factual errors**. The uncertainty and hedging groups overlap and must not be added together.

## Specific cases requiring follow-up

| Image | User question / topic | Stored Qwen output | Reference-answer signal | Review finding |
| --- | --- | --- | --- | --- |
| `VizWiz_val_00002463.jpg` | Read text | 80 exclamation marks (`!!!!!!!!...`) | References include “gods gift”, “christmas card” | **Clearly unusable response** despite `status=success`. Need detect punctuation-only output. |
| `VizWiz_val_00001598.jpg` | Product name on can | `Dettol` | 10/10 references say `lysol` | **Strong answer/reference disagreement**; inspect image for confirmation. |
| `VizWiz_val_00003489.jpg` | Oven temperature and cooking time | `375°F (190°C) and 10 minutes` | Most references say **350°F and 40–45 minutes** | **High-impact disagreement** with reference instructions; inspect image before use. |
| `VizWiz_val_00001603.jpg` | “what kind of pee is this?” (likely transcribed question issue) | Describes a green cap, “Gel” and “Streptococcus” | References predominantly say **Sleepytime herbal tea** | **Strong mismatch**; could involve poor text reading or misleading question transcription. |
| `VizWiz_val_00003867.jpg` | Item identification | `pen` | References often say **white cane / walking cane / walking stick**; some say pen | **Potential object misidentification**; source references are not unanimous. |
| `VizWiz_val_00003715.jpg` | Type of coffee | `Van Houtte` | Majority says **Mexico**, one says `van houtte mexico mexique dark` | **Incomplete answer**: gives brand rather than requested variety. |
| `VizWiz_val_00000757.jpg` | Code to be typed | `123456` | Predominantly **unanswerable** | **Unsupported-specificity risk**; must inspect image. |
| `VizWiz_val_00000782.jpg` | CAPTCHA code | `12345` | Predominantly **unanswerable** | **Unsupported-specificity risk**; must inspect image. |
| `VizWiz_val_00004297.jpg` | CAPTCHA characters | `T38R` | Predominantly **lex** or similar | **Strong reference disagreement**; inspect image before assuming hallucination. |
| `VizWiz_val_00000671.jpg` | Mail addressee | `Enclosed Billing Statement` | 10/10 references **unanswerable** | **May confuse a document heading with an addressee**. |
| `VizWiz_val_00003319.jpg` | Location of volume buttons | Buttons below numeric keypad | References mostly locate them on **middle left** | **Navigation-description disagreement**; needs visual check. |

## Examples aligned with references

- `VizWiz_val_00000642.jpg`: recognized **laptop**.
- `VizWiz_val_00000803.jpg`: described **red and white stripes**.
- `VizWiz_val_00002353.jpg`: identified **barbecue-flavored Lay's chips**.
- `VizWiz_val_00003670.jpg`: answered **Caffè Verona**.
- `VizWiz_val_00003733.jpg`: answered **keyboard**.

These are examples of **textual agreement with references**, not a scored benchmark or systematic image verification.

## Practical implications for the next iteration

1. **Usable-response validation:** treat an answer made only of punctuation, repeated characters, or whitespace as invalid. A row-level `success` status alone is insufficient.
2. **Uncertainty over invented precision:** request cautious wording for unreadable text, ambiguous labels and codes; review cases marked `answerable=0`.
3. **Accessibility:** prefer brief direct answers; optionally support a separate detailed-description mode when the user explicitly asks for one.
4. **Targeted quality tests:** create regression cases for `VizWiz_val_00002463.jpg`, `VizWiz_val_00001598.jpg`, `VizWiz_val_00003489.jpg` and `VizWiz_val_00001603.jpg`. Avoid changing model parameters until actual image review supports a diagnosis.
5. **Separate Arabic evaluation:** all 100 fixed-sample questions are in English; Arabic script quality remains unresolved and needs its own controlled test.
6. **Deem handoff:** provide this qualitative-review document beside the raw CSV. Formal correctness assessment and BLIP comparison remain with evaluation.

## Reproducible heuristics

```python
import pandas as pd

df = pd.read_csv("outputs/qwen_week12_100_FINAL.csv")
a = df["qwen_answer"].fillna("")
words = a.str.split().str.len()

uncertainty = r"unsure|unclear|cannot|can.t|unable|not clear|difficult to|too blurry|not visible|impossible|not discern|hard to|don.t know|not enough|not certain|uncertain"
hedging = r"appears|seems|likely|might|possibly|probably|perhaps|could be|may be"

print("successful", (df["status"] == "success").sum())
print("longer than 30 words", (words > 30).sum())
print("longer than 50 words", (words > 50).sum())
print("uncertainty phrasing", a.str.contains(uncertainty, case=False, regex=True).sum())
print("hedging phrasing", a.str.contains(hedging, case=False, regex=True).sum())
print("punctuation-only", a.str.fullmatch(r"[^\\w\\s]+").sum())
```
