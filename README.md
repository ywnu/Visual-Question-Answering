# Jumanah — VQA Model Development

## Your task
Build the actual VQA engine: **image + English question -> generated answer**. Start by running one real end-to-end inference example on a JPG; select a VQA-capable vision-language model and processor appropriate for the laptop hardware, license, and instructor's lab/rubric requirements. Create `src/vqa_model.py` exposing `answer_question(image: PIL.Image.Image, question: str) -> str` for Hanan to call. Record model name/version, preprocessing, runtime and failures. Discuss training or fine-tuning with Deem; do not claim training if using only pretrained inference.

## Data and software
Use Aseel's processed VizWiz records and locally downloaded train/validation annotations and JPG images from https://vizwiz.org/tasks-and-datasets/vqa/. Your local model cache/weights must stay outside GitHub. Start with Pillow and the appropriate PyTorch/Transformers versions *after* choosing the model; never download large weights blindly.

## Deliverables and done criteria
Reusable inference function, installation/setup notes, a real image + question demonstration showing the predicted answer, latency/resource notes, and integration contract. This is complete when inference returns an actual answer (not just Answerable/Unanswerable).

## Project context
Our goal is a **working Visual Question Answering (VQA) website** for people with visual impairments: **image + question -> textual answer**. The previously completed Logistic Regression baseline in `main.py` predicts **answerability only**, not the answer. It must not be presented as the final VQA model.

**Team:** Aseel (data), Jumanah (VQA model), Deem (evaluation), Hanan (backend), Bayader (frontend). Keep code on your own feature branch; raise a pull request for review before merging. The original `main.py` is already on `main`; do not overwrite it. Do not upload dataset images, archives, model weights, secrets or `.venv`.
