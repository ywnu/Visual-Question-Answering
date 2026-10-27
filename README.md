# Visual Question Answering for People with Visual Impairments
**Machine Learning course team project — planning and integration in progress.**

[English](#english) | [العربية](#العربية)

## English
### What we are building
A fully functioning, accessible bilingual **Arabic/English** AI website for people who are blind or have low vision. Users upload an image, type or speak their question, and receive a real AI-generated textual answer, with optional spoken playback (voice interaction is required).

**Priority 0 — Required:** bilingual chat, image upload, questions and genuine VQA-generated text answers through the website.

**Priority 1 — Required:** bilingual Arabic/English spoken questions (speech-to-text) and spoken answers (text-to-speech), with permissions and accessible error handling. Test actual Arabic browser compatibility early; use an alternative service if necessary.

**Priority 2 — Optional/stretch:** live camera preview and frame capture while asking. Fully continuous video conversation is beyond the minimum target. Do not block priorities 0–1.

### Actual progress (Oct 2026)
**Done:** problem and VizWiz dataset selection; basic EDA with 20,523 annotated training samples (5 fields), no null values, no duplicate image-question records in the checked dataset; class distribution (14,991 answerable, 5,532 unanswerable); question length/word-count features; question-only TF-IDF + Logistic Regression answerability baseline (about 67.8% accuracy on a held-out 20% split); classification report and confusion-matrix code. Original analysis is in [main.py](main.py). GitHub feature branches and bilingual task instructions have been created.

**Locally prepared, not integrated:** training photos have been extracted to an external drive. They are NOT linked in code yet.

**In progress / NOT built:** actual model that generates VQA answers; image-question-answer loader; bilingual capability verification; functioning frontend/backend; microphone input, speech output, hosted site, and live camera. The supplemental SVM comparison code is on Deem's evaluation branch, but results are NOT yet measured.

### Team responsibilities
| Member | Role | Branch | English instructions | Arabic instructions |
| --- | --- | --- | --- | --- |
| Aseel | Data Preparation & EDA | `feature/data` | [English](../../tree/feature/data) | [العربية](../../blob/feature/data/README_AR.md) |
| Jumanah | VQA Model Development | `feature/vqa-model` | [English](../../tree/feature/vqa-model) | [العربية](../../blob/feature/vqa-model/README_AR.md) |
| Deem | Model Training & Evaluation | `feature/evaluation` | [English](../../tree/feature/evaluation) | [العربية](../../blob/feature/evaluation/README_AR.md) |
| Hanan | Backend & Integration | `feature/backend` | [English](../../tree/feature/backend) | [العربية](../../blob/feature/backend/README_AR.md) |
| Bayader | Frontend & Accessibility | `feature/frontend` | [English](../../tree/feature/frontend) | [العربية](../../blob/feature/frontend/README_AR.md) |

### Integration contract (planned)
- `GET /api/health` -> `{"status":"ok"}`.
- `POST /api/answer` accepts multipart form-data fields `image` (JPG/PNG) + `question` (Arabic/English text); returns `{"answer":"..."}` or appropriate `{"error":"..."}`.
- `answer_question(image, question) -> str` is the intended Python VQA interface.
- Browser voice transcription and playback may be used, with fallback if Arabic support fails.

Use pull requests from feature branches to main. Keep huge VizWiz JPGs, model weights, API secrets and virtual environments OUT of Git. Get data from https://vizwiz.org/tasks-and-datasets/vqa/ following provider terms.

### Next milestones
1. Aseel implements reproducible image/question/reference answer preparation; Jumanah proves bilingual VQA inference on a real image.
2. Hanan and Bayader deliver a **clearly marked mock** end-to-end website request, later replaced by real model outputs.
3. Add/test Arabic and English microphone transcription + speech output.
4. Deem evaluates real answer quality and documents measured metrics.
5. Deploy HTTPS and verify the live website; only then attempt optional camera mode.

## العربية
### فكرة المشروع
بناء **موقع فعّال ومتوافق مع متطلبات إمكانية الوصول** لمساعدة المكفوفين وضعاف البصر. المستخدم يرفع صورة، ويسأل كتابة أو بصوته **بالعربي أو الإنجليزي**، ويحصل على **إجابة فعلية من نموذج يفهم الصورة والسؤال** تظهر في المحادثة، مع القدرة على سماع الإجابة صوتيًا.

**الأولوية P0 — أساسية:** موقع محادثة يدعم رفع الصور والسؤال الكتابي بالعربي والإنجليزي وإجابات VQA حقيقية.

**الأولوية P1 — أساسية:** سؤال صوتي وتحويل الكلام لنص، وقراءة الإجابة صوتيًا بالعربي والإنجليزي. لازم نختبر دعم العربية والميكروفون وإذن المستخدم ونجهز بديلًا إذا لم يدعم المتصفح بعض الوظائف.

**الأولوية P2 — اختيارية:** تشغيل معاينة الكاميرا والتقاط صورة وقت السؤال. تحليل الفيديو باستمرار مثل مكالمة مباشرة ميزة إضافية أصعب، ولا تؤخر الأساسيات.

### وش خلصنا فعلًا؟
- اخترنا VizWiz وفكرة المشروع.
- حللنا 20,523 سجل تدريب، وفحصنا القيم الناقصة والتكرارات وتوزيع الفئات؛ عندنا 14,991 قابل للإجابة و5,532 غير قابل للإجابة.
- حسبنا طول السؤال وعدد الكلمات، ونفّذنا TF-IDF وLogistic Regression لتصنيف **قابلية الإجابة فقط**؛ دقته نحو 67.8% في تقسيم اختبار 20%، مع Classification Report وConfusion Matrix.
- رفعنا `main.py` على GitHub، وأنشأنا خمسة Branches وREADME بالعربي والإنجليزي لكل بنت.
- فكّينا ضغط صور التدريب على هاردسك خارجي، **لكن ما ربطناها بالكود**.

### وش باقي؟
**لم ننفذ حتى الآن** نموذج VQA يعطي الإجابة نفسها، ولا ربط الصور بالأسئلة والإجابات، ولا موقع يعمل، ولا مدخلات صوتية أو رد صوتي أو دعم عربي مؤكد، ولا استضافة. تجربة SVM موجود كودها في فرع ديم لكن **لم نشغلها ونقيس نتائجها**.

### تقسيم الفريق
- **أسيل — `feature/data`:** Data Preparation & EDA. تكمل التحليل السابق وتجهز الصور والأسئلة والإجابات.
- **جمانه — `feature/vqa-model`:** تطوير نموذج VQA بالعربي والإنجليزي.
- **ديم — `feature/evaluation`:** تدريب/تحسين النموذج وتقييم الإجابات والمقارنات.
- **حنان — `feature/backend`:** API وربط النموذج والواجهة والصوت عند الحاجة.
- **بيادر — `feature/frontend`:** المحادثة والموقع والتصميم وإمكانية الوصول وواجهة الصوت.

كل بنت تراجع `README.md` و`README_AR.md` داخل فرعها. لا نرفع الصور الكبيرة أو الأوزان ولا نعدّل `main.py` القديم عشوائيًا. نستخدم Pull Requests بعد المراجعة.

### خطة العمل القادمة
1. تجهيز بيانات الصور والأسئلة والإجابات، وبالتوازي تشغيل نموذج VQA حقيقي بالعربي والإنجليزي على صورة.
2. بناء الموقع وAPI مع رد تجريبي **موسوم بوضوح** ثم استبداله بالنموذج الحقيقي.
3. تشغيل الصوت: سؤال مسموع وجواب مسموع باللغتين.
4. تقييم دقة الإجابات ثم نشر الموقع واختباره.
5. إذا خلصت الأساسيات، نضيف وضع الكاميرا الاختياري.
