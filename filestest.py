import json
from pathlib import Path

base = Path(__file__).resolve().parent / "outputs" / "data_preparation" / "samples"
for name in ["train", "val"]:
    data = json.load(open(base / f"{name}_sample.json", encoding="utf-8"))
    rate = sum(d["answerable"] for d in data) / len(data)
    print(name, "| rows:", len(data), "| answerable:", round(rate, 3), "| unique images:", len({d["image"] for d in data}))
    print("   keys:", sorted(data[0].keys()))

rec = json.load(open(base / "train_sample.json", encoding="utf-8"))[0]
print()
print("question:", rec["question"])
print("answers (first 3):", rec["answers"][:3])
print("reference_answers:", rec["reference_answers"])
print("answerable:", rec["answerable"], "| answer_type:", rec["answer_type"])
print("unanswerable_share:", rec["unanswerable_share"], "| top_answer:", rec["top_answer"])