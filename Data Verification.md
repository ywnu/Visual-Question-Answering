# Data Verification

[العربية](#arabic) | [English](#english)

<a id="arabic"></a>

# مشروع VizWiz VQA: تجهيز البيانات وعينات التقييم

يوثّق هذا الملف عمل أسيل على فرع `feature/data`، من تجهيز بيانات VizWiz الإنجليزية إلى إعداد عينة التقييم. يبدأ العمل بسؤال تصنيف ثنائي: هل يمكن الإجابة عن السؤال من الصورة؟ يتضمن الفرع نموذجًا مرجعيًا يعتمد على نص السؤال فقط، وتحليلًا استكشافيًا، وبيانات مجهزة، وعينة جاهزة لتقييم النموذج. لا توجد حتى الآن نتيجة موثّقة لتقييم نموذج يولّد الإجابات.

## البيانات والإعداد

نزّل ملفات البيانات والصور وفق [شروط استخدام VizWiz VQA](https://vizwiz.org/tasks-and-datasets/vqa/). ضع ملفَي `train.json` و`val.json` في `data/`، وصور التدريب في `data/images/`، وصور التحقق في `data/images_val/`.

يحتوي كل سجل مصنّف على اسم ملف الصورة، والسؤال، والإجابات المرجعية، ونوع الإجابة، وتصنيف قابلية الإجابة (`1` يعني قابل للإجابة). بيانات الاختبار لا تتضمن هذه التصنيفات. يعتمد التجهيز على التقسيمين الرسميين للتدريب والتحقق.

ثبّت المكتبات المطلوبة من المجلد الرئيسي للمستودع:

```bash
python -m pip install pandas numpy matplotlib scikit-learn pillow
```

## 1. النموذج المرجعي

يشغّل الأمر `python baseline_model.py` نموذج انحدار لوجستي بأوزان متوازنة للفئات، ويحوّل نص السؤال إلى خصائص رقمية باستخدام `TF-IDF`. بلغت الدقة المسجّلة **67.8%**، وبلغ متوسط `F1` عبر الفئتين نحو **0.63**. حُسبت النتيجتان على **4,105 أسئلة** من جزء محجوز يمثّل **20%** من بيانات التدريب، مع الحفاظ على توزيع الفئتين واستخدام البذرة `42`.

للمقارنة، اختيار «قابل للإجابة» لكل سؤال يعطي دقة تقارب **73%** على الجزء نفسه. هذه الأرقام تخص تصنيف قابلية الإجابة من النص، ولا تقيس جودة الإجابات المولّدة أو الأداء على تقسيم التحقق الرسمي.

## 2. تحليل الأسئلة وتجهيز البيانات

يشغّل الأمر `python data_prep_eda.py` التحليل الاستكشافي الأولي. رسم الارتباط `outputs/eda_plots/09_feature_correlation.png` يعرض **7 خصائص نصية من بيانات التدريب فقط**. كان عدد الأحرف أقواها ارتباطًا بقابلية الإجابة من حيث القيمة المطلقة، بمعامل بيرسون يقارب **−0.14**، وهو ارتباط ضعيف. ارتبطت الأسئلة الأطول بانخفاض قابلية الإجابة. ويشغّل `complete_task.py` النموذج المرجعي وهذا التحليل الأولي فقط.

لتشغيل التجهيز الأصلي:

```bash
python src/data_preparation.py
```

يفحص السكربت الحقول المطلوبة والتصنيفات غير الصالحة، ويعالج أزواج الصورة والسؤال المكررة أو المتعارضة. كما يحتفظ بالإجابات الأصلية والإجابات المرجعية بعد تنظيفها، ويفحص إمكانية قراءة الصور والتداخل بين التقسيمين. يحفظ الملفات المجهزة بصيغتَي CSV وJSON، والعينات، والتقارير، والرسوم في `outputs/data_preparation/`.

يمكن تغيير المسارات باستخدام `--data-dir` و`--train-images` و`--val-images` و`--output-dir`. نسخة الأسبوع 12 هي `week12/data_preparation.py`. تستخدم النسختان مجلد الإخراج الافتراضي نفسه، لذلك استخدم مجلدين مختلفين عند مقارنة تشغيلاتهما حتى لا تُستبدل المخرجات.

| النتيجة المسجّلة | التدريب | التحقق |
|---|---:|---:|
| السجلات المحتفظ بها | 20,523 | 4,319 |
| السجلات القابلة للإجابة | 14,991 (73.0%) | 2,934 (67.9%) |
| متوسط الكلمات لكل سؤال | 6.67 | 7.17 |
| الصور المفقودة أو غير القابلة للقراءة | 0 | 0 |
| السجلات التي لا تحتوي على إجابات | 0 | 0 |
| السجلات التي لا تحتوي على 10 إجابات: التشغيل الأصلي / الأسبوع 12 | 0 / 2 | 0 / 0 |

لم يسجّل التقريران أي صورة أو زوج صورة وسؤال مشترك بين التقسيمين. ولم يُحدّد سبب ظهور الحالتين في تدريب الأسبوع 12.

أما رسم التجهيز `plots/questions_feature_correlation.png` فيجمع **التدريب والتحقق**، ويعرض **5 خصائص** لا تشمل عدد الأحرف أو متوسط طول الكلمة. وظهرت فروق وصفية بين الفئتين في السطوع والحدة. الحدة المحسوبة من صورة مصغّرة مؤشر تقريبي، وهذه الفروق لا تثبت أن الصور تتنبأ بقابلية الإجابة أفضل من النص. لا تستخدم الحقول المستمدة من الإجابات كمدخلات لنموذج التصنيف، ومنها الإجابات المرجعية ونوع الإجابة واتفاق المعلّقين ودرجة ثقتهم.

## 3. تجهيز عينة التقييم في الأسبوع 12

ينتج التجهيز عينتين، كل واحدة **500 سجل**، مع الحفاظ على نسب الفئتين تقريبًا واستخدام البذرة `42`. أما عينة التقييم ذات **100 سجل** فتُسحب من **كامل بيانات التحقق المجهزة**، وليست جزءًا من عينة الـ500.

يقرأ سكربت التقييم `outputs/data_preparation/prepared/val_prepared.json` ويستخدم الصور من `data/images_val/`. بعد توفير الملف والصور، شغّله من المجلد الرئيسي للمستودع:

```bash
python week12/prepare_eval_sample.py
```

يفحص السكربت الحقول والتصنيفات وقوائم الإجابات المرجعية، ويتحقق من إمكانية فتح الصور، ثم يأخذ عينة تحافظ على توزيع الفئتين ويفحص تكرار أزواج الصورة والسؤال. في التشغيل المسجّل اجتازت **4,319 سجلات** من التحقق هذه الفحوص. فتح الصور فحص تقني، ولا يعني مراجعة محتواها يدويًا أو تقييم إجابات النموذج.

| العينة المحفوظة | قابل للإجابة | غير قابل للإجابة | المجموع |
|---|---:|---:|---:|
| التدريب | 365 | 135 | 500 |
| التحقق | 340 | 160 | 500 |
| التقييم، من كامل بيانات التحقق المجهزة | 68 | 32 | 100 |

يكتب السكربت `evaluation_sample_100_checked.json` و`evaluation_sample_100_checked.csv` في `outputs/evaluation/`. توجد النسخ المحفوظة في Git داخل `week12/evaluation/`. يحتفظ ملف JSON بالإجابات المرجعية كقوائم، ويخزّنها ملف CSV كنص بصيغة JSON. أداة `week12/chick_samplesjson.py` تعدّ سجلات عينتَي الـ500 فقط، وتستخدم مسارًا خاصًا بجهاز واحد. أداة فحص مجلد الصور في المجلد الرئيسي اسمها `check_image_folder.py`.

## 4. محتويات المستودع وإعادة التشغيل

سياسة المشروع هي إبقاء البيانات الخام والصور وأوزان النماذج وبيانات الدخول والبيئات الافتراضية خارج Git. لكن الفرع يتتبّع بالفعل ملفات بيانات مشتقة وعينات ورسومًا وتقارير تحت `outputs/` و`week12/`، وبعض التقارير يتضمن مسارات محلية. قواعد التجاهل الحالية تشمل `data/` ولا تستبعد مجلدات المخرجات هذه. راجع شروط استخدام البيانات قبل إعادة توزيع الملفات المشتقة.

طوبقت بنية الفرع ومسارات السكربتات والأوامر والتقارير المحفوظة وأعداد العينات مع [نسخة `50798b2`](https://github.com/ywnu/Visual-Question-Answering/tree/50798b242a46a113bef22c5ad886c650349aefa3). لم يُعَد تشغيل مراحل التجهيز أو النموذج أو فحص الصور لإعداد هذه المسودة.

## 5. المنتج المخطّط بالعربية والإنجليزية

المطلوب موقع يتيح للمستخدم رفع صورة وطرح سؤال بالعربية أو الإنجليزية، ويعرض الإجابة التي يولّدها النموذج داخل المحادثة. واجهة الربط **مقترحة ولم تُشغّل في هذا الفرع**:

- `POST /api/answer` يستقبل الحقلين `image` و`question` بصيغة `multipart`، ويعيد `{"answer":"..."}` أو `{"error":"..."}`.
- `GET /api/health` يعيد `{"status":"ok"}`.

إدخال السؤال بالصوت وقراءة الإجابة صوتيًا متطلبان باللغتين. يُختبر دعم المتصفحات مبكرًا، مع توفير بديل عند الحاجة، وطلب إذن الميكروفون، واستخدام `HTTPS`، ودعم قارئ الشاشة ورسائل الخطأ الواضحة. معاينة الكاميرا والتقاط صورة عند السؤال خياران بعد اكتمال الموقع والصوت. تحليل الفيديو المستمر مرحلة لاحقة.

لا يحتوي هذا الفرع على موقع عامل أو خدمة صوت أو واجهة مستضافة. يعتمد الفريق على فروع مستقلة وطلبات دمج تُراجع قبل قبولها.

---

<a id="english"></a>

# VizWiz VQA — Data preparation and evaluation samples

Aseel's work on `feature/data` prepares English VizWiz image–question records for a binary task: can the question be answered from the photo? The branch contains a question-only baseline, exploratory analysis, prepared splits and a checked evaluation sample. The sample is ready for model evaluation. No generative VQA score is reported.

## Data and setup

Obtain the annotations and photos under the [VizWiz VQA terms](https://vizwiz.org/tasks-and-datasets/vqa/). Place `train.json` and `val.json` in `data/`, training photos in `data/images/`, and validation photos in `data/images_val/`. Each labeled record includes an image filename, question, reference answers, answer type and binary label (1 = answerable). The test annotations lack labels. Use the official training and validation splits for preparation.

From the repository root, install the dependencies:

```bash
python -m pip install pandas numpy matplotlib scikit-learn pillow
```

## 1. Establish the baseline

`python baseline_model.py` fits TF-IDF and class-balanced logistic regression using question text. The reported accuracy was 0.678 and macro-F1 was 0.63 on a stratified 20% holdout of the training annotations (4,105 questions, seed 42). Always predicting “answerable” scores about 73% there. These metrics measure binary classification on that holdout. They do not measure generated-answer quality or performance on the official validation split.

## 2. Explore questions, then prepare the full records

`python data_prep_eda.py` compares the official splits. Its training-only correlation plot, `outputs/eda_plots/09_feature_correlation.png`, covers seven text features. Character count had the largest reported absolute Pearson correlation (about −0.14). Longer questions were less often answerable. The optional wrapper `complete_task.py` runs the baseline and this first analysis only.

For the original preparation workflow, run:

```bash
python src/data_preparation.py
```

The script cleans required fields, invalid labels and duplicate or conflicting image–question pairs. It retains original answers and normalized references, checks image readability and split overlap, and writes prepared CSV/JSON files, samples, reports and plots under `outputs/data_preparation/`. Use `--data-dir`, `--train-images`, `--val-images` and `--output-dir` to override paths. The Week 12 revision is `week12/data_preparation.py`. Both versions default to the same output directory. Use separate output directories when comparing runs.

| Recorded result | Train | Validation |
|---|---:|---:|
| Retained records | 20,523 | 4,319 |
| Answerable | 14,991 (73.0%) | 2,934 (67.9%) |
| Mean words per question | 6.67 | 7.17 |
| Missing or unreadable images | 0 | 0 |
| Records without answers | 0 | 0 |
| Records with an unexpected answer count, original / Week 12 | 0 / 2 | 0 / 0 |

Both reports record zero shared images and zero shared image–question pairs across splits. The cause of the two training exceptions in the Week 12 report is unresolved. The preparation plot `plots/questions_feature_correlation.png` pools both splits and uses five features, excluding character count and average word length. Image sharpness and brightness distributions also differed by class. Thumbnail sharpness is a proxy, and these observations do not establish predictive superiority. Keep answer-derived fields, including reference answers, answer type, agreement and annotator confidence, out of answerability model inputs.

## 3. Prepare the Week 12 evaluation sample

Preparation produces two stratified samples of 500 using seed 42. The separate evaluation script reads the **full** `outputs/data_preparation/prepared/val_prepared.json` and images from `data/images_val/`. With those inputs in place, run from the repository root:

```bash
python week12/prepare_eval_sample.py
```

The script checks required fields, labels and reference lists, opens the images, samples by class and checks duplicate image–question pairs. The recorded run accepted all 4,319 validation records.

| Saved sample | Answerable | Unanswerable | Total |
|---|---:|---:|---:|
| Training | 365 | 135 | 500 |
| Validation | 340 | 160 | 500 |
| Evaluation from full validation | 68 | 32 | 100 |

The evaluation script writes `evaluation_sample_100_checked.json` and its CSV counterpart to `outputs/evaluation/`. The committed copies are in `week12/evaluation/`. JSON preserves reference answers as lists. CSV stores them as JSON strings. The helper `week12/chick_samplesjson.py` only counts the two 500-record files and contains a machine-specific path. The root image-folder helper is named `check_image_folder.py`. Image validation here means the files could be read. It does not establish that someone assessed their content.

## 4. Repository contents and reproducibility

The project policy keeps raw datasets, photos, weights, secrets and environments out of Git. The reviewed branch nevertheless tracks derived records, samples, plots and reports under `outputs/` and `week12/`. Some reports contain local paths. The current ignore rules cover `data/` but do not exclude those output trees. Check the dataset terms before redistributing derived records.

This review checked the branch tree, script paths, saved reports and sample counts at [commit 50798b2](https://github.com/ywnu/Visual-Question-Answering/tree/50798b242a46a113bef22c5ad886c650349aefa3). The documented commands were checked against source. The pipeline, model and image checks were not rerun for this draft.

## 5. Planned bilingual product

The planned website accepts a photo and an Arabic or English question and returns a model-generated answer in the conversation. The proposed API accepts multipart `image` and `question` fields at `POST /api/answer`, returning `{"answer":"..."}` or `{"error":"..."}`. The proposed `GET /api/health` returns `{"status":"ok"}`.

Bilingual speech input and spoken answers are required. Test Arabic and English speech support early, provide a fallback if browser support is insufficient, request microphone consent, use HTTPS, and provide screen-reader support and clear error states. Camera preview and frame capture are optional after the website and voice flows work. Continuous video understanding is a later extension. The branch contains no working website, speech service or hosted API. The team's integration plan uses feature branches and reviewed pull requests.
