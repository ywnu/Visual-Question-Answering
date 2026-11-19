import json
from pathlib import Path

folder = Path(r"C:\Users\KUNYSR\Desktop\vqa_project\outputs\data_preparation\samples")

for filename in ["train_sample.json", "val_sample.json"]:
    with open(folder / filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(filename, ":", len(data), "records")