# Basir — VQA Model Guide

## 1. Overview

Basir is a Visual Question Answering (VQA) project designed to assist blind and visually impaired users.

The model receives an image and a question, then generates a short answer based on the visible content.

**Selected model:** Qwen2.5-VL-3B-Instruct
**Model source:** `Qwen/Qwen2.5-VL-3B-Instruct`
**Supported question languages:** Arabic and English

## 2. Project Files

* `src/vqa_model.py` — Main VQA implementation.
* `src/vqa_model_qwen3.py` — Previous model retained for comparison.
* `test_vqa.py` — Local testing script.
* `requirements.txt` — Required Python packages.

## 3. Installation

Use Python 3.11 in a virtual environment.

Install the required packages from the project root:

```bash
python -m pip install -r requirements.txt
```

The first execution requires an internet connection to download the model from Hugging Face.

The model weights are not included in the GitHub repository.

## 4. Using the Model

Import the `answer_question()` function:

```python
from src.vqa_model import answer_question

image_path = "path/to/image.jpg"
question = "ما الشيء الموجود في الصورة؟"

answer = answer_question(image_path, question)

print(answer)
```

For English questions:

```python
answer = answer_question(
    "path/to/image.jpg",
    "What is in this image?"
)

print(answer)
```

The function accepts an image path or a PIL Image object.

**Input:** Image and question.
**Output:** Answer as a Python string.

## 5. Initial Testing

The model was tested locally with Arabic and English questions.

Observed results included:

* Correct identification of a keyboard in English.
* Correct identification of a keyboard in Arabic.
* Correct identification of a Hello Kitty character in Arabic.
* Description of multiple objects in an image, with some minor inaccuracies.

These are preliminary functional tests, not a full accuracy evaluation.

## 6. GPU Setup and Performance

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Tested GPU: NVIDIA Tesla T4 with 14.56 GiB available in Colab
- GPU configuration: CUDA, FP16, `attn_implementation="eager"`
- Tested Colab versions: PyTorch `2.11.0+cu130`, Transformers `4.57.6`
- GPU memory allocated after loading: approximately 7.0 GiB

### Reproduce the Colab test

1. Open the [Basir evaluation notebook](https://colab.research.google.com/drive/1gUBXoOK_eIM1ak04f6B5LdawOCe94NOe?usp=sharing).
2. Select **Runtime → Change runtime type** and choose **T4 GPU**.
3. Select **Runtime → Run all**. The first run downloads the model weights.
4. Check the final cell for the Arabic and English answers and timings.

A fresh Colab session measured 11.59 seconds for one English question and
4.83 seconds for one Arabic question on the same image. Each timing covers
one call to `answer_question()` after model loading. The Arabic answer mixed
scripts and ended mid-sentence.

For backend integration, use `answer_question(image, question) -> str`.
The Colab notebook applies the GPU and chat-template compatibility changes
to its checkout. FastAPI integration with Qwen is pending verification.
## 7. Backend Integration

The backend can import the main function:

```python
from src.vqa_model import answer_question
```

Example:

```python
result = answer_question(image, question)
```

The returned value is a string that can be included in an API response.

The model currently loads when its Python module is imported. Backend initialization and GPU deployment require further integration testing.

## 8. Current Status

The model is ready for initial integration and further evaluation.

Remaining work includes:

* Testing integration with FastAPI.
* Measuring inference speed on a GPU.
* Evaluating accuracy on a larger image dataset.
* Testing the complete frontend-to-backend workflow.
