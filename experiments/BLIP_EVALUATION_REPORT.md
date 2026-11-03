# Basir – Preliminary BLIP Evaluation Report

**Week:** 11
**Responsible:** Deem Almaneea
**Task:** Model Training & Evaluation

## Objective
Evaluate BLIP on real image-question pairs from the VizWiz validation dataset.

## Evaluation Setup
- Model: Salesforce/blip-vqa-base
- Dataset: VizWiz Validation
- Samples: 20 answerable image-question pairs
- Random seed: 42
- Hardware: CPU
- Metrics: Exact Match and response time

## Results
| Metric | Result |
|---|---|
| Successful requests | 20/20 |
| Exact matches | 6/20 |
| Exact-match rate | 30% |
| Average response time | 0.83 seconds |

## Observations
- BLIP successfully processed all 20 requests.
- Some answers matched the references exactly.
- Other answers were incomplete or did not match.
- Exact Match does not capture every semantically correct answer.

## Limitations
- Small sample of 20 answerable questions.
- No Arabic evaluation in this experiment.
- No completed visual review of all images.
- Exact Match is not the same as final VQA accuracy.

## Next Steps
1. Evaluate Qwen2.5-VL on the same image-question pairs.
2. Compare answer quality and response times.
3. Document errors and recommend the most suitable model.

## Conclusion
BLIP achieved a 30% exact-match rate on 20 VizWiz validation samples, with an average response time of 0.83 seconds. These are preliminary results, not final VQA accuracy.
