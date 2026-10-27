# Deem — Model Training & Evaluation

## Your task
Collaborate with Jumanah to evaluate and improve genuine VQA answers. Implement reproducible validation evaluation on a fixed sample and full validation if practical. Compare predicted answers with **all human reference answers** using an appropriate VQA metric; examine failure cases and latency. If course requirements and hardware allow, train/fine-tune or compare VQA candidates and document exact settings. Clearly separate pretrained inference from training.

## Data you need
VizWiz `val.json`, validation JPG images and the prepared loader from Aseel; Jumanah's VQA function/predictions. Use the official validation split, never train on it. See https://vizwiz.org/tasks-and-datasets/vqa/ for files and terms.

## Lab 4 experiment already on this branch
`experiments/compare_answerability_models.py` compares question-only Logistic Regression and Linear SVM (TF-IDF) using `data/train.json`. Run it locally before reporting numbers; it evaluates **answerability**, not VQA, and must remain supplementary.

## Deliverables and done criteria
Repeatable evaluation script, measured metrics and model comparison table, a few failure examples, plots and brief conclusions, with clear commands to reproduce. Do not invent results.

## Project context
Our goal is a **working Visual Question Answering (VQA) website** for people with visual impairments: **image + question -> textual answer**. The previously completed Logistic Regression baseline in `main.py` predicts **answerability only**, not the answer. It must not be presented as the final VQA model.

**Team:** Aseel (data), Jumanah (VQA model), Deem (evaluation), Hanan (backend), Bayader (frontend). Keep code on your own feature branch; raise a pull request for review before merging. The original `main.py` is already on `main`; do not overwrite it. Do not upload dataset images, archives, model weights, secrets or `.venv`.
