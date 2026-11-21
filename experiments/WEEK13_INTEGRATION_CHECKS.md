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

This check covers one English image, a typed question, the displayed answer, and audible playback. Arabic answer generation remains to be checked.

## English microphone input

| Step | Outcome | Evidence |
| --- | --- | --- |
| Select **Ask by Voice** and speak a question | Fail in this environment | Microphone permission was set to Allow. The page displayed `Speech recognition is unavailable.` and no transcript appeared in the question field. |

The [frontend recognition handler](https://github.com/ywnu/Visual-Question-Answering/blob/feature/frontend/frontend/script.js#L679-L690) maps the observed message to the browser recognition error `network`. The browser's underlying failure remains undiagnosed. The typed-question and answer-playback checks passed separately.

## Playback follow-up

The [frontend playback handler](https://github.com/ywnu/Visual-Question-Answering/blob/feature/frontend/frontend/script.js#L574-L617) sets the utterance language from the answer, uses a rate of `0.9` and a pitch of `1`, and leaves voice selection to the browser. To assess the requested improvement, replay a short sentence using available English voices, keeping the speech rate fixed while comparing clarity. Check the preferred voice at the default and adjusted rates before selecting a setting for the demo.
