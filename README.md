# Aseel — Data Preparation & EDA

## Your task
Continue the EDA done in `main.py` rather than repeat it. Implement a new reusable data-preparation module (`src/data_preparation.py`) that maps each image filename to the image, question, human reference answers and binary answerability label. Check missing/corrupt images, document the target-answer selection rule and preserve all reference answers for evaluation. Keep official training and validation sets separate to avoid leakage. Print counts and display a few correct image-question-answer examples.

## Data you need
From https://vizwiz.org/tasks-and-datasets/vqa/ download **train annotations (`train.json`)**, **validation annotations (`val.json`)**, **training images**, and **validation images**. Annotation fields include `image`, `question`, `answers`, `answer_type`, `answerable`. Images are large: store locally/external drive, not in GitHub. Support configurable paths instead of hardcoding `D:`.

## Interface to agree with Jumanah
Provide a loader returning examples containing `image_path`, `question`, `reference_answers`, `target_answer`, `answerable`. Document any excluded or ambiguous samples.

## Deliverables and done criteria
A tested reusable loader, local setup instructions, integrity counts, a small sample check, and a README explanation. Done when Jumanah can load both splits using her own local image paths without changing `main.py`.

## Project context
Our goal is a **working Visual Question Answering (VQA) website** for people with visual impairments: **image + question -> textual answer**. The previously completed Logistic Regression baseline in `main.py` predicts **answerability only**, not the answer. It must not be presented as the final VQA model.

**Team:** Aseel (data), Jumanah (VQA model), Deem (evaluation), Hanan (backend), Bayader (frontend). Keep code on your own feature branch; raise a pull request for review before merging. The original `main.py` is already on `main`; do not overwrite it. Do not upload dataset images, archives, model weights, secrets or `.venv`.
