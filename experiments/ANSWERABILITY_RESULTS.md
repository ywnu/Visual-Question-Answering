# Answerability Model Comparison — Results (27 Oct 2026)

These are **measured results supplied from a local PyCharm run** of `experiments/compare_answerability_models.py` by the evaluation contributor. The script uses question text only (TF-IDF); **neither model analyzes images or generates VQA answers**.

## Evaluation setup
- VizWiz train annotations: `train.json`.
- Held-out split: 20% with `random_state=42`, stratified by `answerable`.
- Test examples: 4,105 (1,107 Unanswerable, 2,998 Answerable).
- TF-IDF: up to 5,000 features; English stop words.
- Both classifiers use class balancing.

| Model | Accuracy | Macro F1 | Unanswerable Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.678197 | **0.634456** | **0.616079** |
| Linear SVM | **0.679172** | 0.633436 | 0.604336 |

Linear SVM's accuracy advantage is about **0.0975 percentage points**. Logistic Regression has slightly higher macro F1 and higher recall for the minority Unanswerable class. The differences are small; neither method dominates all metrics. Because Answerable represents about 73% of training records, do not judge model quality from accuracy alone.

## Arabic summary
هذه مقارنة حقيقية من تشغيل محلي لنموذجين على **نص السؤال فقط**؛ لا يوجد تحليل للصورة ولا توليد إجابة VQA في هذه التجربة.

- دقة Logistic Regression = 67.82%؛ Macro F1 = 0.6345؛ Recall لفئة غير قابل للإجابة = 61.61%.
- دقة Linear SVM = 67.92%؛ Macro F1 = 0.6334؛ Recall لفئة غير قابل للإجابة = 60.43%.
- SVM أعلى قليلًا بالدقة، لكن Logistic Regression أعلى قليلًا في Macro F1 وRecall لفئة Unanswerable.
- ما فيه تفوق واضح وشامل لنموذج واحد، ونحتاج تقييم نموذج VQA النهائي منفصلًا باستخدام الصور والأسئلة والإجابات المرجعية.

## Reproduction
1. Download VizWiz training annotations from https://vizwiz.org/tasks-and-datasets/vqa/ into local `data/train.json` (not GitHub).
2. Run `python experiments/compare_answerability_models.py` from the project root.
3. The script writes a local metrics CSV and two confusion-matrix PNGs under `outputs/`.

**Next:** retain the locally generated plots for tomorrow's progress review; later evaluate actual image-question answers with Jumanah's model.
