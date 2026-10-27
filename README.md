# Bayader — Frontend, Voice & Accessibility

## Your responsibilities
Build a bilingual accessible chat website with image upload, typed/spoken question and spoken/written answer.

## Required data / dependencies
HTML/CSS/JS or agreed stack; browser microphone/speech capability or agreed speech service; backend API; sample JPG only.

## Expected deliverables
Working accessible chat UI, recording/mic permission, speech-to-text and text-to-speech flow tested, error/loading handling.

## Shared scope and priorities
**P0 — Mandatory: Bilingual VQA website.** A user uploads a photo, asks in Arabic or English, and receives a genuine model-generated text answer in the conversation. The website must function end to end.

**P1 — Mandatory: Voice interaction.** Users can ask by voice (speech-to-text) and listen to the AI answer (text-to-speech), in Arabic and English. Choose/test a speech stack early: browser Web Speech support is variable, so an alternative provider or service may be necessary. Verify mic consent, accessibility, secure hosting (HTTPS), and meaningful failure states.

**P2 — Optional, only after P0/P1 work: Live camera.** Camera preview with frame capture during questioning is the first optional implementation; continuous real-time video understanding is a harder extension. Do not block mandatory deliverables on P2.

**Architecture contract:** `POST /api/answer` receives multipart `image` + `question`, returns `{"answer":"..."}` or `{"error":"..."}`; `GET /api/health` returns `{"status":"ok"}`. This is a *planned interface*, not a running hosted app. Speech transcription/synthesis can initially run in the browser; confirm reliability in Arabic before deciding whether to move it server-side.

**Collaboration:** feature branches + reviewed PRs into `main`; preserve existing `main.py` (question-only answerability baseline). No datasets/images, weights, secrets, or virtual environments in Git. Train/validation images and annotations come from https://vizwiz.org/tasks-and-datasets/vqa/ under its terms. The repository currently contains scaffolding/analysis; **there is no functional VQA website or speech support yet**.
