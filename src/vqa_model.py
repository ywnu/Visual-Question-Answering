import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# Load processor
print("Loading Qwen2.5-VL processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)


# Load model
print("Loading Qwen2.5-VL model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    attn_implementation="eager"
)

model.eval()


def answer_question(image, question):
    """
    Answer visual questions in Arabic or English.

    Args:
        image: Image file path or PIL Image.
        question: A question in Arabic or English.

    Returns:
        str: Generated answer in the question's language.
    """

    # --------------------------------
    # 1. Validate the question
    # --------------------------------

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    question = question.strip()

    # --------------------------------
    # 2. Load and prepare the image
    # --------------------------------

    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            image = img.convert("RGB")

    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    else:
        raise TypeError("image must be a file path or PIL Image")

    # --------------------------------
    # 3. Detect question language
    # --------------------------------

    is_arabic = any(
        "\u0600" <= char <= "\u06FF"
        for char in question
    )

    if is_arabic:
        language_instruction = (
            "You must answer only in Arabic. "
            "Use clear and natural Arabic."
        )
    else:
        language_instruction = (
            "You must answer only in English."
        )

    # --------------------------------
    # 4. Prepare model messages
    # --------------------------------

    messages = [
        {
            "role": "system",
            "content": (
                "You are a visual question answering assistant "
                "designed for blind and visually impaired users. "

                "Answer questions using only information "
                "that is visible in the provided image. "

                "Identify objects accurately and specifically. "

                "Do not invent objects, colors, text, "
                "or details that are not visible. "

                "If you cannot identify something confidently, "
                "say that you are unsure. "

                "Keep your answers short, clear, "
                "and directly related to the question. "

                + language_instruction
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    # --------------------------------
    # 5. Process inputs
    # --------------------------------

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # --------------------------------
    # 6. Generate answer
    # --------------------------------

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    # --------------------------------
    # 7. Decode generated answer
    # --------------------------------

    answer_ids = generated_ids[0][
        inputs["input_ids"].shape[-1]:
    ]

    answer = processor.decode(
        answer_ids,
        skip_special_tokens=True
    )

    return answer.strip()