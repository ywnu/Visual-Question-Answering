import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Test the Basir VQA model."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the image"
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question in Arabic or English"
    )

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    if not args.question.strip():
        parser.error("Question cannot be empty")

    from src.vqa_model import answer_question

    print("Question:", args.question)

    start = time.perf_counter()

    answer = answer_question(
        args.image,
        args.question
    )

    elapsed = time.perf_counter() - start

    print("Answer:", answer)
    print("Inference Time:", round(elapsed, 2), "seconds")


if __name__ == "__main__":
    main()