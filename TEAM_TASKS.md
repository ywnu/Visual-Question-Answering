# Team tasks

## Confirmed assignments

### Aseel — Data Preparation & EDA
- Review Jumanah's existing EDA when shared; don't redo completed analysis.
- Map image filenames to questions and reference answers.
- Check local image existence/readability and report missing/corrupt entries.
- Build reproducible training and validation preparation separately; avoid data leakage.
- Implement preprocessing in separate files without modifying Jumanah's original `main.py`.
- Document how to configure local dataset paths. Do not upload image archives.

### Jumanah — VQA Model Development
- Select and run a model that produces actual answers from image + question.
- Test model inference and measure outputs on VizWiz examples.
- Fine-tune or compare models if justified by lab requirements and available hardware.
- Keep the previous text-only answerability model as a supplemental baseline.

## Roles not yet assigned
- Deem
- Hanan
- Bayader

## Working rules
- Each owner works on their feature branch.
- Use pull requests and review before merging into `main`.
- Agree on image/question/answer interfaces before integration.
