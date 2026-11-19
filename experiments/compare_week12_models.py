#!/usr/bin/env python3
"""
Compare Week 12 BLIP and Qwen results on the same fixed 100-item sample.

The shared sample is the source of truth for image, question, answerable label,
and reference answers. Qwen result files may use either response_seconds or
inference_seconds for latency.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from evaluate_blip_week12 import DEFAULT_SAMPLE, load_sample, normalize_text


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BLIP_RESULTS = PROJECT_DIR / "outputs" / "blip_week12_100.csv"
DEFAULT_QWEN_RESULTS = PROJECT_DIR / "outputs" / "qwen_week12_100.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "blip_qwen_week12_comparison.csv"
DEFAULT_SUMMARY = PROJECT_DIR / "outputs" / "blip_qwen_week12_summary.json"


def load_results(path, answer_column, model_name):
    if not path.is_file():
        raise FileNotFoundError(f"{model_name} results file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required = {"image", "question", answer_column, "status"}
        missing = required - columns
        if missing:
            raise ValueError(
                f"{model_name} results are missing columns: "
                + ", ".join(sorted(missing))
            )

        time_column = (
            "response_seconds"
            if "response_seconds" in columns
            else "inference_seconds"
            if "inference_seconds" in columns
            else None
        )

        results = {}
        for row_number, row in enumerate(reader, start=2):
            image = (row.get("image") or "").strip()
            if not image:
                raise ValueError(f"{model_name} row {row_number}: image is empty.")
            if image in results:
                raise ValueError(
                    f"{model_name} results contain duplicate image: {image}"
                )

            results[image] = {
                "question": (row.get("question") or "").strip(),
                "answer": (row.get(answer_column) or "").strip(),
                "status": (row.get("status") or "").strip(),
                "seconds": (row.get(time_column) or "").strip()
                if time_column
                else "",
            }

    return results


def request_succeeded(status):
    return status.strip().lower() in {"ok", "success", "completed"}


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_model(rows, answer_key, status_key, seconds_key, group=None):
    selected = [
        row for row in rows
        if group is None or row["answerable"] == group
    ]
    completed = [
        row for row in selected
        if request_succeeded(row[status_key])
    ]
    usable = [row for row in completed if row[answer_key].strip()]
    matches = sum(int(row[f"{answer_key}_exact_match"]) for row in usable)
    latencies = [
        value for value in (safe_float(row[seconds_key]) for row in completed)
        if value is not None
    ]

    return {
        "items": len(selected),
        "requests_completed": len(completed),
        "request_failures": len(selected) - len(completed),
        "empty_answers": len(completed) - len(usable),
        "exact_matches": matches,
        "exact_match_rate_percent": (
            round(matches / len(usable) * 100, 2) if usable else None
        ),
        "mean_response_seconds": (
            round(statistics.mean(latencies), 3) if latencies else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare BLIP and Qwen Week 12 results."
    )
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--blip-results", type=Path, default=DEFAULT_BLIP_RESULTS)
    parser.add_argument("--qwen-results", type=Path, default=DEFAULT_QWEN_RESULTS)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    try:
        samples = load_sample(args.sample_csv, args.expected_count)
        blip = load_results(args.blip_results, "blip_answer", "BLIP")
        qwen = load_results(args.qwen_results, "qwen_answer", "Qwen")

        sample_images = {item["image"] for item in samples}
        unexpected_blip = sorted(set(blip) - sample_images)
        unexpected_qwen = sorted(set(qwen) - sample_images)
        if unexpected_blip or unexpected_qwen:
            raise ValueError(
                "Result files contain images outside the shared sample. "
                f"BLIP extras: {unexpected_blip[:5]}; "
                f"Qwen extras: {unexpected_qwen[:5]}"
            )

        comparison_rows = []
        for sample in samples:
            image = sample["image"]
            blip_row = blip.get(image)
            qwen_row = qwen.get(image)

            for model_name, model_row in (("BLIP", blip_row), ("Qwen", qwen_row)):
                if model_row is not None and model_row["question"] != sample["question"]:
                    raise ValueError(
                        f"{model_name} question mismatch for {image}. "
                        "Both models must use the exact shared question."
                    )

            references = sample["reference_answers"]
            normalized_references = {normalize_text(ref) for ref in references}

            blip_answer = blip_row["answer"] if blip_row else ""
            qwen_answer = qwen_row["answer"] if qwen_row else ""
            blip_status = blip_row["status"] if blip_row else "missing_result"
            qwen_status = qwen_row["status"] if qwen_row else "missing_result"
            blip_seconds = blip_row["seconds"] if blip_row else ""
            qwen_seconds = qwen_row["seconds"] if qwen_row else ""

            comparison_rows.append(
                {
                    "image": image,
                    "question": sample["question"],
                    "answerable": sample["answerable"],
                    "reference_answers": json.dumps(
                        references, ensure_ascii=False
                    ),
                    "blip_answer": blip_answer,
                    "blip_status": blip_status,
                    "blip_seconds": blip_seconds,
                    "blip_answer_exact_match": int(
                        bool(blip_answer)
                        and normalize_text(blip_answer) in normalized_references
                    ),
                    "qwen_answer": qwen_answer,
                    "qwen_status": qwen_status,
                    "qwen_seconds": qwen_seconds,
                    "qwen_answer_exact_match": int(
                        bool(qwen_answer)
                        and normalize_text(qwen_answer) in normalized_references
                    ),
                }
            )

        output_fields = list(comparison_rows[0].keys())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(comparison_rows)

        summary = {
            "total_items": len(samples),
            "answerable_items": sum(row["answerable"] for row in samples),
            "unanswerable_items": sum(
                1 for row in samples if row["answerable"] == 0
            ),
            "blip": {
                "overall": evaluate_model(
                    comparison_rows,
                    "blip_answer",
                    "blip_status",
                    "blip_seconds",
                ),
                "answerable": evaluate_model(
                    comparison_rows,
                    "blip_answer",
                    "blip_status",
                    "blip_seconds",
                    group=1,
                ),
                "unanswerable": evaluate_model(
                    comparison_rows,
                    "blip_answer",
                    "blip_status",
                    "blip_seconds",
                    group=0,
                ),
            },
            "qwen": {
                "overall": evaluate_model(
                    comparison_rows,
                    "qwen_answer",
                    "qwen_status",
                    "qwen_seconds",
                ),
                "answerable": evaluate_model(
                    comparison_rows,
                    "qwen_answer",
                    "qwen_status",
                    "qwen_seconds",
                    group=1,
                ),
                "unanswerable": evaluate_model(
                    comparison_rows,
                    "qwen_answer",
                    "qwen_status",
                    "qwen_seconds",
                    group=0,
                ),
            },
            "metric_note": (
                "Exact Match uses lowercase + whitespace normalization "
                "against any of the sample's reference answers. It is not "
                "official VizWiz VQA accuracy or semantic accuracy. "
                "Review outputs against the images before choosing a model."
            ),
            "missing_results": {
                "blip": sum(1 for row in comparison_rows if row["blip_status"] == "missing_result"),
                "qwen": sum(1 for row in comparison_rows if row["qwen_status"] == "missing_result"),
            },
        }

        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("Comparison complete.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"CSV saved: {args.output}")
        print(f"Summary saved: {args.summary}")
        return 0

    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
