"""Standalone Qwen FastAPI integration for Basir.

Run this API INSTEAD of the BLIP API on a T4 GPU, not alongside it.
Hanan's feature/backend/vqa_api.py remains unchanged.
"""
from io import BytesIO
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
answer_question = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global answer_question
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen GPU backend requires an NVIDIA CUDA GPU.")
    from src.vqa_model_gpu import answer_question as gpu_answer_question
    answer_question = gpu_answer_question
    yield
    answer_question = None


app = FastAPI(
    title="Basir Qwen Backend API",
    description="Visual Question Answering with Qwen2.5-VL-3B",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "VQA Backend is running!",
        "model": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if answer_question is not None else "starting",
        "model_loaded": answer_question is not None,
        "model": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    image: UploadFile = File(...),
):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    if answer_question is None:
        raise HTTPException(status_code=503, detail="The VQA model is not ready.")

    image_bytes = await image.read()
    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            uploaded_image.verify()
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            image_rgb = uploaded_image.convert("RGB")
            image_info = {
                "format": uploaded_image.format,
                "width": uploaded_image.width,
                "height": uploaded_image.height,
            }
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    try:
        answer = answer_question(image_rgb, question)
    except Exception as error:
        print(f"Qwen inference error: {type(error).__name__}: {error}")
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    return {
        "success": True,
        "question": question,
        "image_filename": image.filename,
        "image_info": image_info,
        "answer": answer,
        "model": MODEL_NAME,
        "model_connected": True,
        "device": "cuda",
    }
