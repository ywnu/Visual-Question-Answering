
from fastapi.testclient import TestClient
from unittest.mock import patch
from io import BytesIO

from PIL import Image

from vqa_api import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "VQA Backend is running!"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_empty_question():
    response = client.post(
        "/ask",
        data={"question": "   "},
        files={"image": ("test.png", b"fake image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please enter a question."


def test_invalid_image():
    response = client.post(
        "/ask",
        data={"question": "What is in this picture?"},
        files={"image": ("test.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is not a valid image."


def test_successful_prediction():
    image = Image.new("RGB", (100, 100), color="white")
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    with patch("vqa_api.model", object()), \
         patch("vqa_api.processor", object()), \
         patch("vqa_api.generate_answer", return_value="cat"):

        response = client.post(
            "/ask",
            data={"question": "What is in this picture?"},
            files={"image": ("test.png", image_bytes, "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["answer"] == "cat"
    assert response.json()["model_connected"] is True
