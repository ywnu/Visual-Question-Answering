# Bayader — Frontend & Accessibility

## Your task
Develop the functioning web interface for blind and low-vision users. It must allow **image upload, question entry, submit, loading/error feedback, and real answer display** from Hanan's backend. Use semantic labels, keyboard-only navigation, visible focus indicators, accessible error messages, `aria-live` for announced results, sufficient contrast and responsive layout. Test with screen-reader and keyboard. Speech playback is optional after the text flow works.

## Backend contract
Send `FormData` with fields `image` and `question` to `POST /api/answer`. Read `{"answer":"..."}`; display `{"error":"..."}` for failures. Agree with Hanan on API base URL, CORS and deployment. Mock responses are allowed only for early development and must be explicitly identified.

## Data and setup
You **do not need to download train/validation images or annotations**. Use one small sample JPG to test upload, HTML/CSS/JavaScript (or agreed frontend stack), and the backend test endpoint.

## Deliverables and done criteria
Working `frontend/` code, setup/run instructions, accessibility checklist, desktop/mobile checks, and proof that a real uploaded image + question gets the true model answer through the site. Hosting is not complete until tested at the published URL.

## Project context
Our goal is a **working Visual Question Answering (VQA) website** for people with visual impairments: **image + question -> textual answer**. The previously completed Logistic Regression baseline in `main.py` predicts **answerability only**, not the answer. It must not be presented as the final VQA model.

**Team:** Aseel (data), Jumanah (VQA model), Deem (evaluation), Hanan (backend), Bayader (frontend). Keep code on your own feature branch; raise a pull request for review before merging. The original `main.py` is already on `main`; do not overwrite it. Do not upload dataset images, archives, model weights, secrets or `.venv`.
