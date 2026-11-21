# Basir Week 13 integration checks

## Local setup

The frontend from `feature/frontend` was served locally and sent requests to the FastAPI `/ask` endpoint from `feature/backend`. The backend used BLIP on CPU.

## English image question and answer playback

| Step | Outcome | Evidence |
| --- | --- | --- |
| Upload `VizWiz_val_00000642.jpg` | Pass | The selected filename appeared above the question field. |
| Type and submit `what is in this picture?` | Pass | The answer panel displayed `laptop`. |
| Select **Read Answer** | Pass | The tester confirmed hearing the word “laptop.” |
| Assess playback quality | Follow-up | The tester requested a voice adjustment after hearing the answer. |

This check covers one English image, a typed question, the displayed answer, and audible playback. English microphone input and Arabic answer generation are separate checks that remain to be recorded.

## Playback follow-up

The [frontend playback handler](https://github.com/ywnu/Visual-Question-Answering/blob/feature/frontend/frontend/script.js#L574-L617) sets the utterance language from the answer, uses a rate of `0.9` and a pitch of `1`, and leaves voice selection to the browser. To assess the requested improvement, replay a short sentence using available English voices, keeping the speech rate fixed while comparing clarity. Check the preferred voice at the default and adjusted rates before selecting a setting for the demo.
