# Hanan — Backend, API & Integration

## Your responsibilities
Build real inference API, integrate with frontend and voice mechanisms as needed, validate inputs and errors.

## Required data / dependencies
Flask, Pillow, Jumanah answer_question module, test image; no full dataset required.

## Expected deliverables
GET /api/health and POST /api/answer multipart image/question; JSON answer/error; documented deployment needs.

## Shared scope and priorities
**P0 — Mandatory: Bilingual VQA website.** A user uploads a photo, asks in Arabic or English, and receives a genuine model-generated text answer in the conversation. The website must function end to end.

**P1 — Mandatory: Voice interaction.** Users can ask by voice (speech-to-text) and listen to the AI answer (text-to-speech), in Arabic and English. Choose/test a speech stack early: browser Web Speech support is variable, so an alternative provider or service may be necessary. Verify mic consent, accessibility, secure hosting (HTTPS), and meaningful failure states.

**P2 — Optional, only after P0/P1 work: Live camera.** Camera preview with frame capture during questioning is the first optional implementation; continuous real-time video understanding is a harder extension. Do not block mandatory deliverables on P2.

**Architecture contract:** `POST /api/answer` receives multipart `image` + `question`, returns `{"answer":"..."}` or `{"error":"..."}`; `GET /api/health` returns `{"status":"ok"}`. This is a *planned interface*, not a running hosted app. Speech transcription/synthesis can initially run in the browser; confirm reliability in Arabic before deciding whether to move it server-side.

**Collaboration:** feature branches + reviewed PRs into `main`; preserve existing `main.py` (question-only answerability baseline). No datasets/images, weights, secrets, or virtual environments in Git. Train/validation images and annotations come from https://vizwiz.org/tasks-and-datasets/vqa/ under its terms. The repository currently contains scaffolding/analysis; **there is no functional VQA website or speech support yet**.
