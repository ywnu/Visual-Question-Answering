import argparse
import csv
import time
from pathlib import Path

from PIL import Image


def evaluate(input_csv, images_dir, output_csv, use_gpu=False):
    # Read the evaluation dataset
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as file:
        samples = list(csv.DictReader(file))

    if not samples:
        raise ValueError("The evaluation dataset is empty.")

    # Required columns
    required = {"id", "image", "question"}
    missing = required - set(samples[0].keys())

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


    results = []

    for index, sample in enumerate(samples, start=1):
        sample_id = sample["id"]
        image_path = Path(images_dir) / sample["image"]
        question = sample["question"]

        print(f"Processing {index}/{len(samples)}: {sample_id}")

        # Check image availability
        if not image_path.is_file():
            results.append({
                "id": sample_id,
                "image": sample["image"],
                "question": question,
                "qwen_answer": "",
                "inference_seconds": "",
                "status": "image_not_found",
            })
            continue

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            if use_gpu:
                from src.vqa_model_gpu import answer_question
            else:
                from src.vqa_model import answer_question
            start = time.perf_counter()
            answer = answer_question(image, question)
            elapsed = time.perf_counter() - start

            results.append({
                "id": sample_id,
                "image": sample["image"],
                "question": question,
                "qwen_answer": answer,
                "inference_seconds": round(elapsed, 3),
                "status": "success",
            })

        except Exception as error:
            results.append({
                "id": sample_id,
                "image": sample["image"],
                "question": question,
                "qwen_answer": "",
                "inference_seconds": "",
                "status": f"error: {type(error).__name__}: {error}",
            })

    # Export results
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "image",
                "question",
                "qwen_answer",
                "inference_seconds",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation completed. Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Basir Qwen VQA model."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument(
        "--output",
        default="qwen_evaluation_results.csv",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CUDA FP16 module with capped image resolution (requires an NVIDIA GPU).",
    )

    args = parser.parse_args()

    evaluate(args.input, args.images_dir, args.output, use_gpu=args.gpu)