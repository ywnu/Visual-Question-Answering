import zipfile
from pathlib import Path

# Paste the path you copied between the quotes (keep the r before the first quote)
ZIP_PATH = Path(r"C:\Users\KUNYSR\Desktop\val.zip")

OUT_DIR = Path(__file__).resolve().parent / "data" / "images_val"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    raise SystemExit(f"val.zip not found at: {ZIP_PATH}\nCheck the path and try again.")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
count = 0
with zipfile.ZipFile(ZIP_PATH) as z:
    for member in z.infolist():
        name = Path(member.filename).name
        if member.is_dir() or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with z.open(member) as src, open(OUT_DIR / name, "wb") as dst:
            dst.write(src.read())
        count += 1

print("Extracted images:", count)
print("Saved to:", OUT_DIR)