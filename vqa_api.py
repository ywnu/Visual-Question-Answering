
from io import BytesIO

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from fastapi.middleware.cors import CORSMiddleware

from models.blip_model import load_blip_model, generate_answer


app = FastAPI(
    title="VQA Backend API",
    description="Visual Question Answering with BLIP",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


@app.on_event("startup")
def load_model():
    global processor, model

    print(f"Loading VQA model on {device}...")

    processor, model = load_blip_model(device)

    print("VQA model loaded successfully!")


@app.get("/")
def home():
    return {
        "message": "VQA Backend is running!",
        "model": "Salesforce/blip-vqa-base",
        "device": device,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
    }


@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    image: UploadFile = File(...),
):
    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

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

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="The VQA model is not ready.",
        )

    try:
        answer = generate_answer(
            processor,
            model,
            image_rgb,
            question,
            device,
        )

    except Exception as error:
        print(f"VQA inference error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer.",
        )

    return {
        "success": True,
        "question": question,
        "image_filename": image.filename,
        "image_info": image_info,
        "answer": answer,
        "model": "Salesforce/blip-vqa-base",
        "model_connected": True,
        "device": device,
    }
