
import csv
import json
import random
import time
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parents[1]
VAL_JSON = PROJECT_DIR.parent / "ML_Project" / "data" / "val.json"
IMAGE_DIR = Path(r"D:\ML_Datasets\val\val")
OUTPUT_CSV = PROJECT_DIR / "outputs" / "blip_evaluation.csv"

API_URL = "http://127.0.0.1:8000/ask"
SAMPLE_SIZE = 20
RANDOM_SEED = 42

def normalize(text):
    return " ".join(str(text).lower().strip().split())

def main():
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    # نختار أسئلة قابلة للإجابة، وصورها موجودة
    eligible = [
        r for r in records
        if r.get("answerable") == 1
        and r.get("question", "").strip()
        and (IMAGE_DIR / r["image"]).is_file()
        and r.get("answers")
    ]

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(eligible, min(SAMPLE_SIZE, len(eligible)))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i, record in enumerate(sample, 1):
        image_path = IMAGE_DIR / record["image"]
        question = record["question"]

        reference_answers = [
            a["answer"]
            for a in record["answers"]
            if isinstance(a, dict) and a.get("answer")
        ]

        print(f"[{i}/{len(sample)}] {record['image']}")

        try:
            start = time.perf_counter()

            with open(image_path, "rb") as image_file:
                response = requests.post(
                    API_URL,
                    data={"question": question},
                    files={"image": (
                        image_path.name,
                        image_file,
                        "image/jpeg"
                    )},
                    timeout=180
                )

            elapsed = round(time.perf_counter() - start, 2)
            response.raise_for_status()
            answer = response.json().get("answer", "").strip()

            # مقياس تطابق نصي أولي، وليس دقة VQA النهائية
            exact_match = int(
                normalize(answer)
                in {normalize(a) for a in reference_answers}
            )
            error = ""

        except (requests.RequestException, ValueError) as exc:
            answer = ""
            exact_match = 0
            elapsed = ""
            error = str(exc)

        results.append({
            "image": record["image"],
            "question": question,
            "reference_answers": " | ".join(reference_answers),
            "blip_answer": answer,
            "exact_match": exact_match,
            "response_seconds": elapsed,
            "error": error
        })

        print(f"  Answer: {answer}")
        print(f"  Exact match: {exact_match}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    successful = [r for r in results if not r["error"]]
    matches = sum(r["exact_match"] for r in successful)

    print("\n--- RESULTS ---")
    print("Requests completed:", len(successful), "/", len(results))
    if successful:
        print("Exact-match rate:", round(
            matches / len(successful) * 100, 2
        ), "%")
    print("Saved:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
