import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

for name in ["train", "val", "test"]:
    with open(BASE_DIR / "data" / f"{name}.json", encoding="utf-8") as f:
        data = json.load(f)
    print(name, "| rows:", len(data), "| keys:", sorted(data[0].keys()))
    if "answerable" in data[0]:
        share = sum(d["answerable"] for d in data) / len(data)
        print("   answerable share:", round(share, 3))