# BLIP vs Qwen2.5-VL — 20 Sample Results

**Week 11 · 4 November 2026 · Deem Almaneea**

Identical 20 answerable English VizWiz validation pairs. Exact Match compares normalized prediction strings to any of ten reference answer strings; this is not official VizWiz VQA accuracy.

| Measure | BLIP | Qwen2.5-VL |
|---|---:|---:|
| Predictions completed | 20/20 | 20/20 (1 unusable punctuation-only response) |
| Exact Match | 6/20 (30%) | 3/20 (15%) |
| Average time | 0.83 s (CPU, API) | 4.51 s (Tesla T4, FP16+SDPA) |

**Caution:** The latency figures use different hardware and execution paths, so they are not a controlled speed comparison. Qwen often generates longer, partially reference-aligned answers that exact-string comparison scores as incorrect.

## Per-question preview

| # | Image | Question | BLIP answer | Qwen answer (preview) | B-EM | Q-EM | Qwen time |
|---:|---|---|---|---|---:|---:|---:|
| 1 | `VizWiz_val_00003882.jpg` | What color is this? | gray | The image appears to be a light gray or beige color. | 0 | 0 | 4.71 s |
| 2 | `VizWiz_val_00000847.jpg` | What does this say? | login | The screen shows various app icons on a smartphone or tablet. The apps listed include: -… | 0 | 0 | 3.43 s |
| 3 | `VizWiz_val_00000195.jpg` | What does the screen on my TV say? | star trek | The screen says "Starrz premium" and "starz + ence." It also mentions "Channels 326 - 34… | 0 | 0 | 3.36 s |
| 4 | `VizWiz_val_00001836.jpg` | What is this? | wine | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! | 1 | 0 | 8.29 s |
| 5 | `VizWiz_val_00001663.jpg` | What is the title of this cd? | jazz | The title of the CD is "Jazz Collection 1: Grand Reunion." | 0 | 0 | 5.20 s |
| 6 | `VizWiz_val_00001539.jpg` | What is this? | vase | This is a vase. It has a long neck and a wide base, with a glossy finish. The vase appea… | 1 | 0 | 3.94 s |
| 7 | `VizWiz_val_00001033.jpg` | Is it a box? | no | No | 1 | 1 | 2.43 s |
| 8 | `VizWiz_val_00000775.jpg` | Where are you? | in bedroom | I am in a room with a desk, a computer monitor, and a wall sticker with two bears on a t… | 0 | 0 | 6.49 s |
| 9 | `VizWiz_val_00004096.jpg` | Tell me what the Ipad screen says. Thank you very much. | it ' s in black | The iPad screen shows the Apple logo, indicating that it is turned on but not displaying… | 0 | 0 | 3.48 s |
| 10 | `VizWiz_val_00003353.jpg` | what is on this shirt? | scissors | The shirt has a graphic design featuring a pair of glasses with a red nose and a black h… | 0 | 0 | 5.10 s |
| 11 | `VizWiz_val_00000644.jpg` | Do the clouds look like storm clouds? What I'm saying is d… | storm | The clouds in the picture do appear to be storm clouds. They are dark and heavy-looking,… | 0 | 0 | 7.09 s |
| 12 | `VizWiz_val_00003595.jpg` | What medicine is this? | advil | Advil Cold & Sinus | 1 | 0 | 2.81 s |
| 13 | `VizWiz_val_00002647.jpg` | What does this say? | new york times | Jergens New Soft Shimmer SKIN RADIANCE MOISTURIZER | 0 | 1 | 3.28 s |
| 14 | `VizWiz_val_00000258.jpg` | OK, lets try this again and see if this works. Do you see … | yes | The image shows a window from a remote control support tool called "Ammyy Admin v2.13." … | 1 | 0 | 5.98 s |
| 15 | `VizWiz_val_00000240.jpg` | Can you please tell me what's on screen? Thank you. | no | The image appears to be a screenshot of a computer command prompt (cmd) window. The prom… | 0 | 0 | 6.59 s |
| 16 | `VizWiz_val_00000708.jpg` | What is this? | craft supplies | This is a grocery store coupon book with various deals on food items. The coupons includ… | 0 | 0 | 4.63 s |
| 17 | `VizWiz_val_00001512.jpg` | What type of Doritos are these? | extra | These are Doritos Ranch flavored chips. | 0 | 0 | 0.69 s |
| 18 | `VizWiz_val_00001591.jpg` | What's that brand name say? I'd like to know what company … | sony | SONY | 1 | 1 | 4.34 s |
| 19 | `VizWiz_val_00003130.jpg` | What is this? | lunch box | This appears to be a box of cereal. The character on the box looks like it might be from… | 0 | 0 | 5.77 s |
| 20 | `VizWiz_val_00003662.jpg` | what container is this, please? | ice cream | Campbell's Vegetable and Pea Soup | 0 | 0 | 2.65 s |

## Observations from reference answers

- **Advil**: BLIP `advil` and Qwen `Advil Cold & Sinus` both align with references; Qwen fails exact match due to extra detail/punctuation.
- **Jazz Collection**: Qwen includes the reference phrase `jazz collection`, but a longer CD title causes an exact mismatch.
- **Starz TV**: Qwen identifies a Starz premium offer, whereas BLIP says `star trek`.
- **Sony**: both models match the reference.
- **Wine**: Qwen returns 80 exclamation points and is not usable; BLIP matches `wine`.
- **Mixed-language Arabic output** was observed in separate early Qwen tests; the 20-question run was English-only.

These qualitative observations were made using **reference text**, not by independent human review of the original pictures. No semantic accuracy percentage is claimed.

**Method:** BLIP `Salesforce/blip-vqa-base` ran on CPU via FastAPI. Qwen `Qwen/Qwen2.5-VL-3B-Instruct` used a temporary FP16+CUDA+SDPA Colab adaptation on T4 and original-size image files. Model branch `feature/vqa-model` was not modified. Source: `outputs/qwen_evaluation.csv`.
