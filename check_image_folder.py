import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "data" / "images"

with open(BASE_DIR / "data" / "train.json", encoding="utf-8") as f:
    train_names = {d["image"] for d in json.load(f)}

files = [p for p in IMG_DIR.iterdir() if p.is_file()]
names = {p.name for p in files}

print("Files in images folder:", len(files))
print("By extension:", dict(Counter(p.suffix.lower() for p in files)))
print("Train images found:", len(train_names & names), "/", len(train_names))

extra = sorted(names - train_names)
print("Files not listed in train.json:", len(extra))
print("Examples:", extra[:10])

prefixes = Counter(
    n.split("_")[1] if n.startswith("VizWiz_") and len(n.split("_")) > 2 else "other"
    for n in extra
)
print("Extra files by name prefix:", dict(prefixes))