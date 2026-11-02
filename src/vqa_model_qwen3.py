import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading VQA model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    attn_implementation="eager"
)

model.eval()


def answer_question(image, question):
    """
    Answer questions about images in Arabic or English.

    Args:
        image: Image path or PIL Image.
        question: Question in Arabic or English.

    Returns:
        str: Generated answer.
    """

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("image must be a file path or PIL Image")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
    "You are a visual assistant for blind and visually impaired users. "
    "Answer the question accurately using only what is visible in the image. "
    "Identify the complete object, not just part of its name. "
    "If the question is in Arabic, respond in clear, natural Arabic. "
    "If the question is in English, respond in English. "
    "Keep the answer short and specific. "
    "If you cannot identify the object confidently, say that you are unsure. "
    "Do not guess or invent details.\n"
    f"Question: {question}"
)
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    answer_ids = generated_ids[0][inputs["input_ids"].shape[-1]:]

    answer = processor.decode(
        answer_ids,
        skip_special_tokens=True
    )

    return answer.strip()