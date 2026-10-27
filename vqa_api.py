
from io import BytesIO

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForQuestionAnswering

app = FastAPI(
    title="VQA Backend API",
    description="Visual Question Answering with BLIP",
    version="2.0.0",
)

MODEL_NAME = "Salesforce/blip-vqa-base"

# Load the model once when the application starts
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


@app.on_event("startup")
def load_model():
    global processor, model

    print(f"Loading VQA model on {device}...")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    print("VQA model loaded successfully!")


@app.get("/")
def home():
    return {
        "message": "VQA Backend is running!",
        "model": MODEL_NAME,
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
        inputs = processor(
            image_rgb,
            question,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
            )

        answer = processor.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

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
        "model": MODEL_NAME,
        "model_connected": True,
        "device": device,
    }
