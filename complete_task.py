import runpy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

BASELINE_FILE = BASE_DIR / "baseline_model.py"     # your original code, unchanged
PREP_EDA_FILE = BASE_DIR / "data_prep_eda.py"      # the data preparation + EDA code


# -----------------------------
# Step 1: run your original code
# -----------------------------
print("=" * 60)
print("STEP 1: Running the original baseline code")
print("=" * 60)
baseline = runpy.run_path(str(BASELINE_FILE))

# Everything defined in the original code is available here,
# for example: baseline["df"], baseline["model"], baseline["y_pred"]
print("\nBaseline finished. Accuracy variables available:",
      [k for k in ("df", "model", "vectorizer", "y_test", "y_pred") if k in baseline])


# -----------------------------
# Step 2: data preparation and EDA
# -----------------------------
print("\n" + "=" * 60)
print("STEP 2: Running data preparation and EDA")
print("=" * 60)
prep = runpy.run_path(str(PREP_EDA_FILE))

print("\nAll done. Check the 'outputs' folder for plots and prepared CSV files.")