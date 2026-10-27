# Visual Question Answering for People with Visual Impairments

Course machine-learning project using the VizWiz VQA dataset. The intended system accepts an image and a natural-language question and returns a textual answer.

**Status:** Project setup in progress. The actual VQA model, website, and deployment are not built yet.

## Current progress
- On Jumanah's laptop (not yet uploaded): basic EDA of VizWiz training annotations, a question-only answerability classifier (Logistic Regression), classification report, and confusion matrix.
- Training images extracted to an external drive but not yet linked by code.
- Next: image-question-answer preparation, real VQA inference/model, API, accessible website, evaluation, deployment.

## Team roles (confirmed)
- **Aseel** — Data Preparation & EDA.
- **Jumanah** — VQA Model Development.
- **Deem, Hanan, Bayader** — roles pending team discussion.

## Branches
- `main`: shared stable branch.
- `feature/data`: Aseel's work.
- `feature/vqa-model`: Jumanah's work.

Submit work as pull requests; don't commit to `main` directly.

## Data and local development
Download VizWiz data from https://vizwiz.org/tasks-and-datasets/vqa/ and follow its usage terms. Keep the large image datasets outside GitHub and never commit API keys, virtual environments, or model checkpoints. Configure dataset paths per machine rather than assuming a fixed drive letter.

The existing `main.py` is still on Jumanah's device: it has **not** been imported into this repo, and nothing on her laptop is changed by this setup.

## Model objective
`(image, question) -> generated answer`

The answerability classifier is an optional supporting experiment, not the final VQA model.
