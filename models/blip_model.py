
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering


MODEL_NAME = "Salesforce/blip-vqa-base"


def load_blip_model(device):
    """Load BLIP processor and model."""

    print(f"Loading BLIP model on {device}...")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    print("BLIP model loaded successfully!")

    return processor, model


def generate_answer(processor, model, image, question, device):
    """Generate an answer from an image and question."""

    inputs = processor(
        image,
        question,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30
        )

    answer = processor.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    return answer
