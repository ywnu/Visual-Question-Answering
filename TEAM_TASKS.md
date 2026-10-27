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

### Deem — Model Training & Evaluation
- Collaborate with Jumanah to test and improve model performance.
- Measure answer quality on held-out VizWiz validation examples.
- Compare experiments, document metrics and failure cases, and prepare evaluation charts.

### Hanan — Backend & Integration
- Build a backend API for image + question input and answer output.
- Integrate the actual VQA inference module when available.
- Validate inputs, handle errors, and coordinate deployment/testing with Bayader.

### Bayader — Frontend & Accessibility
- Build the functional website interface: image upload, question input, answer display.
- Support accessible labels, keyboard navigation, and screen-reader use.
- Integrate with Hanan's API, and test the full user workflow.

## Working rules
- Each owner works on her feature branch.
- Use pull requests and review before merging into `main`.
- Agree on image/question/answer interfaces before integration.
- Large VizWiz images and trained model artifacts stay outside GitHub.
