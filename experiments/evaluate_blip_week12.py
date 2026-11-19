#!/usr/bin/env python3
"""
Basir Week 12 — evaluate BLIP on Aseel's fixed 100-item sample.

The script keeps answerable and unanswerable items, sends the exact same
image/question pairs from the shared CSV to FastAPI POST /ask, and records
raw answers, exact-match results, latency, HTTP status, and errors.

Exact Match is a simple normalized string comparison against any reference
answer. It is NOT the official VizWiz VQA metric and does not replace
human semantic review.
"""

import argparse
import csv
import json
import mimetypes
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = (
    PROJECT_DIR / "week12" / "evaluation" / "evaluation_sample_100_checked.csv"
)
DEFAULT_IMAGE_DIR = Path(
    os.environ.get("BASIR_IMAGE_DIR", r"D:\ML_Datasets\val\val")
)
DEFAULT_API_URL = os.environ.get(
    "BASIR_BLIP_API_URL", "http://127.0.0.1:8000/ask"
)
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "blip_week12_100.csv"
DEFAULT_SUMMARY = PROJECT_DIR / "outputs" / "blip_week12_100_summary.json"


REQUIRED_COLUMNS = {"image", "question", "reference_answers", "answerable"}


def normalize_text(value):
    """Match the simple lowercase + whitespace normalization used before."""
    return " ".join(str(value or "").strip().lower().split())


def parse_references(raw_value, row_number):
    try:
        values = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Row {row_number}: reference_answers must be a JSON array."
        ) from exc

    if not isinstance(values, list):
        raise ValueError(
            f"Row {row_number}: reference_answers must decode to a JSON array."
        )

    references = [str(value).strip() for value in values if str(value).strip()]
    if not references:
        raise ValueError(f"Row {row_number}: no reference answers were found.")
    return references


def parse_answerable(raw_value, row_number):
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes"}:
        return 1
    if value in {"0", "false", "no"}:
        return 0
    raise ValueError(
        f"Row {row_number}: answerable must be 0/1 or true/false; got {raw_value!r}."
    )


def load_sample(sample_path, expected_count):
    if not sample_path.is_file():
        raise FileNotFoundError(
            f"Sample CSV not found: {sample_path}\n"
            "The checked sample is on branch feature/data. From the repo root, "
            "run:\n"
            "git checkout feature/data -- week12/evaluation/evaluation_sample_100_checked.csv"
        )

    records = []
    with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                "Sample CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            image_name = (row.get("image") or "").strip()
            question = (row.get("question") or "").strip()
            if not image_name:
                raise ValueError(f"Row {row_number}: image is empty.")
            if not question:
                raise ValueError(f"Row {row_number}: question is empty.")

            records.append(
                {
                    "image": image_name,
                    "question": question,
                    "reference_answers": parse_references(
                        row.get("reference_answers", ""), row_number
                    ),
                    "answerable": parse_answerable(
                        row.get("answerable", ""), row_number
                    ),
                }
            )

    if expected_count and len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} sample rows, but found {len(records)}."
        )

    image_names = [record["image"] for record in records]
    duplicates = sorted(
        {name for name in image_names if image_names.count(name) > 1}
    )
    if duplicates:
        raise ValueError(
            "Duplicate image entries found in the sample: "
            + ", ".join(duplicates[:10])
        )

    return records


def check_images(records, image_dir):
    missing = [
        record["image"]
        for record in records
        if not (image_dir / record["image"]).is_file()
    ]
    return missing


def calculate_group_metrics(rows):
    completed = [row for row in rows if row["status"] == "ok"]
    matches = sum(int(row["exact_match"]) for row in completed)
    latencies = [
        float(row["response_seconds"])
        for row in completed
        if row["response_seconds"] not in ("", None)
    ]
    return {
        "items": len(rows),
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "exact_matches": matches,
        "exact_match_rate_percent": (
            round(matches / len(completed) * 100, 2) if completed else None
        ),
        "mean_response_seconds": (
            round(statistics.mean(latencies), 3) if latencies else None
        ),
    }


def build_summary(results, records, sample_path, image_dir, api_url):
    answerable_rows = [row for row in results if row["answerable"] == 1]
    unanswerable_rows = [row for row in results if row["answerable"] == 0]
    completed = [row for row in results if row["status"] == "ok"]

    return {
        "run_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sample_csv": str(sample_path),
        "image_directory": str(image_dir),
        "api_url": api_url,
        "total_items": len(records),
        "answerable_items": len(answerable_rows),
        "unanswerable_items": len(unanswerable_rows),
        "overall": calculate_group_metrics(results),
        "answerable": calculate_group_metrics(answerable_rows),
        "unanswerable": calculate_group_metrics(unanswerable_rows),
        "status_counts": {
            status: sum(1 for row in results if row["status"] == status)
            for status in sorted({row["status"] for row in results})
        },
        "metric_note": (
            "Exact Match is normalized lowercase/whitespace string matching "
            "against any supplied reference answer. It is not official VizWiz "
            "VQA accuracy and does not measure semantic correctness."
        ),
    }


def run_evaluation(records, sample_path, image_dir, api_url, timeout, output_path, summary_path):
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The requests package is required. Install it with: "
            "python -m pip install requests"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    health_url = api_url.rsplit("/", 1)[0] + "/health"
    try:
        health_response = requests.get(health_url, timeout=min(timeout, 20))
        health_response.raise_for_status()
        health_data = health_response.json()
        if not isinstance(health_data, dict):
            raise ValueError("BLIP API health response must be a JSON object.")
    except requests.RequestException as exc:
        raise RuntimeError(
            f"BLIP API health check failed at {health_url}: {exc}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"BLIP API health endpoint returned invalid JSON: {health_url}"
        ) from exc

    if health_data.get("model_loaded") is False:
        raise RuntimeError(
            "The API is reachable, but model_loaded=false. "
            "Wait until BLIP finishes loading, then rerun."
        )

    output_fields = [
        "image",
        "question",
        "answerable",
        "reference_answers",
        "blip_answer",
        "exact_match",
        "response_seconds",
        "http_status",
        "model",
        "device",
        "status",
        "error",
    ]

    results = []
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()

        for index, record in enumerate(records, start=1):
            image_path = image_dir / record["image"]
            result = {
                "image": record["image"],
                "question": record["question"],
                "answerable": record["answerable"],
                "reference_answers": json.dumps(
                    record["reference_answers"], ensure_ascii=False
                ),
                "blip_answer": "",
                "exact_match": 0,
                "response_seconds": "",
                "http_status": "",
                "model": health_data.get("model", ""),
                "device": health_data.get("device", ""),
                "status": "",
                "error": "",
            }

            print(f"[{index}/{len(records)}] {record['image']}")

            if not image_path.is_file():
                result["status"] = "missing_image"
                result["error"] = f"Image file not found: {image_path}"
            else:
                started = time.perf_counter()
                try:
                    mime_type = (
                        mimetypes.guess_type(image_path.name)[0]
                        or "application/octet-stream"
                    )
                    with image_path.open("rb") as image_file:
                        response = requests.post(
                            api_url,
                            data={"question": record["question"]},
                            files={
                                "image": (
                                    image_path.name,
                                    image_file,
                                    mime_type,
                                )
                            },
                            timeout=timeout,
                        )

                    result["response_seconds"] = round(
                        time.perf_counter() - started, 3
                    )
                    result["http_status"] = response.status_code
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("BLIP API response must be a JSON object.")
                    answer = payload.get("answer", "")
                    if not isinstance(answer, str):
                        answer = str(answer)
                    answer = answer.strip()

                    result["blip_answer"] = answer
                    result["model"] = payload.get("model", result["model"])
                    result["device"] = payload.get("device", result["device"])

                    if not answer:
                        result["status"] = "empty_answer"
                        result["error"] = "API response did not contain an answer."
                    else:
                        normalized_answer = normalize_text(answer)
                        normalized_references = {
                            normalize_text(reference)
                            for reference in record["reference_answers"]
                        }
                        result["exact_match"] = int(
                            normalized_answer in normalized_references
                        )
                        result["status"] = "ok"

                except requests.Timeout as exc:
                    result["response_seconds"] = round(
                        time.perf_counter() - started, 3
                    )
                    result["status"] = "timeout"
                    result["error"] = str(exc)
                except requests.RequestException as exc:
                    result["response_seconds"] = round(
                        time.perf_counter() - started, 3
                    )
                    result["status"] = "request_error"
                    result["error"] = str(exc)
                    if getattr(exc, "response", None) is not None:
                        result["http_status"] = exc.response.status_code
                except (ValueError, TypeError, KeyError) as exc:
                    result["response_seconds"] = round(
                        time.perf_counter() - started, 3
                    )
                    result["status"] = "invalid_response"
                    result["error"] = str(exc)

            writer.writerow(result)
            handle.flush()
            results.append(result)

            print(
                f"  status={result['status']} | "
                f"exact_match={result['exact_match']} | "
                f"seconds={result['response_seconds']}"
            )

    summary = build_summary(results, records, sample_path, image_dir, api_url)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n--- WEEK 12 BLIP RESULTS ---")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"CSV saved: {output_path}")
    print(f"Summary saved: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BLIP on Basir's fixed Week 12 sample."
    )
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the sample and image files without calling the API.",
    )
    args = parser.parse_args()

    try:
        records = load_sample(args.sample_csv, args.expected_count)
        missing = check_images(records, args.image_dir)

        answerable_count = sum(row["answerable"] for row in records)
        unanswerable_count = len(records) - answerable_count

        print(f"Sample rows: {len(records)}")
        print(f"Answerable: {answerable_count}")
        print(f"Unanswerable: {unanswerable_count}")
        print(f"Images found: {len(records) - len(missing)}/{len(records)}")

        if missing:
            print("\nMissing image files (first 20):")
            for image_name in missing[:20]:
                print(f" - {image_name}")

        if args.check_only:
            if missing:
                print("\nPreflight failed: resolve missing image files first.")
                return 2
            print("\nPreflight passed. The sample and all image files are ready.")
            return 0

        if missing:
            print(
                "\nWarning: missing images will be recorded as failed rows; "
                "they will not be silently removed from the sample."
            )

        run_evaluation(
            records=records,
            sample_path=args.sample_csv,
            image_dir=args.image_dir,
            api_url=args.api_url,
            timeout=args.timeout,
            output_path=args.output,
            summary_path=args.summary,
        )
        return 0

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
