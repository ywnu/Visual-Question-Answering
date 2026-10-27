# Hanan — Backend & Integration

## Your task
Implement the Python backend for the website and connect it to Jumanah's VQA function. Use Flask unless the team explicitly agrees on another framework. Begin with clearly labelled mock responses so Bayader can develop the UI; replace the mock with real inference before completion.

## Agreed API contract (not implemented yet)
- `GET /api/health` -> JSON `{"status":"ok"}`.
- `POST /api/answer` -> `multipart/form-data` fields `image` (JPG/PNG file) and `question` (text).
- Successful response -> `{"answer":"..."}`.
- Invalid input -> HTTP 400 and `{"error":"..."}`; inference/server errors -> appropriate HTTP 5xx.

## Inputs and setup
Need `src/vqa_model.py` with `answer_question(image, question) -> str` once available, Flask, Pillow and a small test image. **You do not need the full VizWiz dataset for endpoint development.** Request size/type validation, useful errors, loading models once, configuration and eventual hosting should be documented. No hardcoded private credentials.

## Deliverables and done criteria
`backend/` with runnable API, install/run commands, automated request/error tests, documented example requests, and evidence the real model is connected. Deployment is a separate verified final step, not assumed.

## Project context
Our goal is a **working Visual Question Answering (VQA) website** for people with visual impairments: **image + question -> textual answer**. The previously completed Logistic Regression baseline in `main.py` predicts **answerability only**, not the answer. It must not be presented as the final VQA model.

**Team:** Aseel (data), Jumanah (VQA model), Deem (evaluation), Hanan (backend), Bayader (frontend). Keep code on your own feature branch; raise a pull request for review before merging. The original `main.py` is already on `main`; do not overwrite it. Do not upload dataset images, archives, model weights, secrets or `.venv`.
