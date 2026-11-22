# تقرير تقدم المشروع — مراجعة 28 أكتوبر 2026
# Project Progress Check — 28 October 2026

## الهدف / Goal
مساعد بصري للمكفوفين وضعاف البصر: المستخدم يرفع صورة، يسأل كتابة أو بصوته بالعربي أو الإنجليزي، ثم يتلقى **إجابة حقيقية من نموذج VQA** مكتوبة ومسموعة. عرض الكاميرا المباشر ميزة اختيارية بعد إنجاز الموقع والصوت.

The goal is an accessible, working bilingual Arabic/English visual question answering website with image upload, typed/spoken questions and genuine text/spoken answers. Live camera is stretch scope.

## إنجازات مؤكدة / Verified accomplishments
1. تحديد الفكرة والداتا: VizWiz VQA (https://vizwiz.org/tasks-and-datasets/vqa/).
2. تنزيل وقراءة بيانات التدريب محليًا: **20,523** عينة و5 أعمدة.
3. تحليل البيانات: Missing Values = 0، وتكرار (image, question) = 0.
4. توازن الفئات: **14,991 Answerable (73.04%)** و**5,532 Unanswerable (26.96%)**.
5. استخراج Question Length وWord Count وتحليل المتوسطات.
6. بناء **Question-only Answerability Baseline** باستخدام TF-IDF + Logistic Regression على تقسيم تدريب/اختبار 80/20 مع stratification؛ Accuracy **67.82%**، وMacro F1 ≈ **0.63** (reported from the team's prior local run). تم إنشاء Classification Report وConfusion Matrix.
7. رفع كود هذا التحليل إلى [main.py](../main.py) على GitHub.
8. تجهيز المستودع والفروع الخمسة وREADME بالعربي والإنجليزي.
9. صور التدريب فُكّ ضغطها محليًا على هاردسك خارجي وفق آخر تحديث للفريق، **لكن لم تُربط بالكود بعد**.
10. كود تجربة SVM مقابل Logistic Regression موجود داخل `experiments/compare_answerability_models.py` بفرع `feature/evaluation`، **لكن لا توجد نتائج تشغيل مؤكدة له حتى الآن**.

## لم يُنجز بعد / Not yet completed
- Data preparation that links images, questions and human reference answers.
- A vision-language model that **answers** visual questions (the current Logistic Regression predicts answerability only).
- Arabic VQA evaluation and voice interaction.
- Working frontend, backend integration and deployment.
- Optional camera preview/live questioning.

## المهام المقترحة للمراجعة المقبلة / Immediate working goals
| الشخص | المهمة | الدليل المطلوب |
| --- | --- | --- |
| أسيل / Aseel | قراءة أول صور تدريب مع الأسئلة والإجابات والتحقق من المسارات | 3 linked examples + integrity counts |
| جمانه / Jumanah | اختبار سؤال واحد على صورة بنموذج VQA حقيقي | image + question + actual generated answer |
| ديم / Deem | تشغيل مقارنة SVM وLogistic Regression | CSV metrics + confusion matrix files |
| حنان / Hanan | API يستقبل صورة وسؤالاً ويعيد mock مؤقتًا | verified POST request/response (clearly marked mock) |
| بيادر / Bayader | صفحة رفع صورة وسؤال نصي مع نتائج mock | local usable page/screenshot |

## طريقة تشغيل مقارنة ديم / Deem's experiment
- استخدمي نسخة GitHub من `experiments/compare_answerability_models.py` في جذر المشروع الذي يحتوي `data/train.json`، ضمن مجلد `experiments/`.
- ثبتي `pandas scikit-learn matplotlib` إذا لم تكن متاحة.
- شغّلي من جذر المشروع: `python experiments/compare_answerability_models.py`.
- المخرجات المتوقعة **بعد التشغيل**: `outputs/answerability_model_comparison.csv`، `outputs/logistic_regression_confusion.png`، `outputs/linear_svm_confusion.png`.
- **لا تضيفي نتائج رقمية لـSVM قبل التشغيل الفعلي**. احتفظي بها منفصلة عن تقييم VQA النهائي.

## نقطة توضيح للدكتورة / Important academic distinction
**Current completed ML experiment:** a text-only binary classification baseline for answerability, evaluated on a held-out split.

**Final project under development:** image+question → real text answer, bilingual written/spoken interface and accessible website. The current classifier does not yet fulfil the final VQA objective.

**Data note:** original VizWiz images are large and stored locally, not uploaded to GitHub; use local configured paths and comply with dataset terms.
