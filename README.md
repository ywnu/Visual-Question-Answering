# Basir | Visual Question Answering

Basir is a Machine Learning course project for people who are blind or have low vision. Upload an image, ask a question about it, and get a text answer with optional spoken playback.

[Quick start](#quick-start) · [English demo](#english-demo) · [Evaluation](#evaluation) · [العربية](#العربية)

## Current status

- **Verified locally:** Upload a JPG, submit a typed English question, view the BLIP answer, and hear the answer with **Read Answer**. The interface includes a keyboard skip link, labeled inputs, and live status updates for screen readers.
- **Voice input:** **Ask by Voice** returned “Speech recognition is unavailable” during the local check with microphone access allowed. Typed input remains available. The team is also reviewing the playback voice.
- **Pending verification:** Arabic answer quality, Arabic speech input and playback, camera capture, and the final integrated release.

[Week 13 integration check](https://github.com/ywnu/Visual-Question-Answering/blob/feature/evaluation/experiments/WEEK13_INTEGRATION_CHECKS.md)

## Quick start

You need Git, Python 3.11, a browser, and internet access on the first run to download the BLIP weights. The frontend and backend are currently on separate feature branches.

### 1. Start the backend

From a folder where you want to keep both worktrees:

```sh
git clone https://github.com/ywnu/Visual-Question-Answering.git basir
cd basir
git worktree add --detach ../basir-backend origin/feature/backend
git worktree add --detach ../basir-frontend origin/feature/frontend
cd ../basir-backend
python -m venv .venv
```

Activate the virtual environment with `source .venv/bin/activate` on macOS or Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. If your platform uses `python3` for Python, substitute that command before activation.

```sh
python -m pip install -r requirements.txt
python -m uvicorn vqa_api:app --host 127.0.0.1 --port 8000
```

Leave this terminal running. Open [backend health](http://127.0.0.1:8000/health) and check that `model_loaded` is `true` before submitting a question. The backend chooses CUDA when available and otherwise uses the CPU.

### 2. Start the website

In a second terminal, enter `basir-frontend` from the same parent folder:

```sh
cd basir-frontend
python -m http.server 8080 --bind 127.0.0.1 --directory frontend
```

Open [Basir locally](http://127.0.0.1:8080/). Keep the backend on port 8000 because the current frontend targets `http://127.0.0.1:8000/ask`.

## English demo

| Step | What to do |
| --- | --- |
| 1 | Upload `VizWiz_val_00000642.jpg` from the VizWiz validation image set. |
| 2 | Type `what is in this picture?` and select **Submit Question**. |
| 3 | The verified local run displayed **laptop**. Select **Read Answer** to hear it. |

The [Week 14 demo draft (MP4)](docs/demo/Basir_English_Demo.mp4) replays this documented run using the original image and result screenshot. The short answer audio was recreated for the replay. The final demo will be recorded after the integrated flow is retested.

Image credit: `VizWiz_val_00000642.jpg` by the [VizWiz team](https://vizwiz.org/tasks-and-datasets/vqa/), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The image was resized for display in the video.

[VizWiz VQA dataset](https://vizwiz.org/tasks-and-datasets/vqa/) · [Documented test](https://github.com/ywnu/Visual-Question-Answering/blob/feature/evaluation/experiments/WEEK13_INTEGRATION_CHECKS.md)

## API

| Route | Request | Response fields used by the demo |
| --- | --- | --- |
| `GET /health` | No body | `status`, `model_loaded`, `device` |
| `POST /ask` | Multipart form with `image` and `question` | `success`, `answer`, `model`, `device` |

[Interactive API docs (local)](http://127.0.0.1:8000/docs) · [Backend source](https://github.com/ywnu/Visual-Question-Answering/blob/feature/backend/vqa_api.py)

## Evaluation

### Visual question answering

BLIP and Qwen ran on the same 100 VizWiz validation examples, including 68 answerable and 32 unanswerable questions. For Exact Match, the evaluator lowercases and normalizes whitespace in the output and checks the full answer against the reference strings.

| Measure | Salesforce/blip-vqa-base | Qwen2.5-VL-3B-Instruct |
| --- | ---: | ---: |
| Completed predictions | 100/100 | 100/100 |
| Exact Match, all questions | 31/100 (31%) | 5/100 (5%) |
| Exact Match, answerable | 27/68 | 5/68 |
| Exact Match, unanswerable | 4/32 | 0/32 |
| Mean recorded response time | 0.784 s (CPU) | 3.085 s (Tesla T4) |

These timings reflect different hardware and operating conditions. Qwen used two image-resolution settings across the run. Exact Match can also miss visually correct answers. For example, Qwen read **Caffè Verona**, while the matcher preserved accents and the closest reference was `caffe verona`. The current English demo uses BLIP. Final model selection is pending team validation of the integrated flow.

[Week 12 evaluation report and visual examples](https://github.com/ywnu/Visual-Question-Answering/blob/feature/evaluation/experiments/WEEK12_EVALUATION_REPORT.md) · [Reproduction runbook](https://github.com/ywnu/Visual-Question-Answering/blob/feature/evaluation/experiments/WEEK12_EVALUATION_RUNBOOK.md)

### Question-only answerability baseline

TF-IDF features from the question text were used to classify whether a question was answerable. On a stratified held-out split of 4,105 records, Logistic Regression reached **67.82% accuracy** and **0.6345 macro F1**. Linear SVM reached **67.92% accuracy** and **0.6334 macro F1**.

[Answerability results](https://github.com/ywnu/Visual-Question-Answering/blob/feature/evaluation/experiments/ANSWERABILITY_RESULTS.md)

## Team

| Member | Responsibility | Work branch |
| --- | --- | --- |
| Aseel | Data preparation and exploratory analysis | [feature/data](https://github.com/ywnu/Visual-Question-Answering/tree/feature/data) |
| Jumanah | VQA model development | [feature/vqa-model](https://github.com/ywnu/Visual-Question-Answering/tree/feature/vqa-model) |
| Diyam | Model training and evaluation | [feature/evaluation](https://github.com/ywnu/Visual-Question-Answering/tree/feature/evaluation) |
| Hanan | Backend and integration | [feature/backend](https://github.com/ywnu/Visual-Question-Answering/tree/feature/backend) |
| Bayader | Frontend and accessibility | [feature/frontend](https://github.com/ywnu/Visual-Question-Answering/tree/feature/frontend) |

Keep model weights, dataset images, virtual environments, and secrets outside Git. Obtain the dataset from [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/) and follow its terms.

## العربية

**بصير** مشروع لمساعدة المكفوفين وضعاف البصر على طرح أسئلة عن الصور. المسار اللي تأكدنا منه محليًا هو رفع صورة وكتابة سؤال بالإنجليزي، ثم عرض الإجابة وسماعها بزر **Read Answer**. إدخال السؤال بالصوت يحتاج إعادة اختبار، ومسار الإجابة بالعربي ينتظر التحقق من الجودة.

للتشغيل، افتحي طرفيتين حسب [خطوات التشغيل](#quick-start). شغّلي الـAPI على المنفذ `8000` والواجهة على المنفذ `8080`. بعد ما يجهز النموذج، افتحي [واجهة بصير المحلية](http://127.0.0.1:8080/) وجربي صورة اللابتوب. اكتبي `what is in this picture?`، اضغطي **Submit Question**، وبعد ما تظهر إجابة **laptop** اضغطي **Read Answer**.
