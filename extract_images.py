import zipfile
from pathlib import Path

ZIP_PATH = Path(r"C:\Users\KUNYSR\Desktop\train.zip")   # change if your zip is somewhere else
IMAGES_DIR = Path(__file__).resolve().parent / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

count = 0
with zipfile.ZipFile(ZIP_PATH) as z:
    for member in z.infolist():
        if member.is_dir():
            continue
        # keep only the file name, so no nested folders are created
        target = IMAGES_DIR / Path(member.filename).name
        with z.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
        count += 1

print("Extracted files:", count)