# ─── CONFIGURE PATHS ────────────────────────────────────────────
# Kaggle: /kaggle/input/<your-dataset-name>/Data_Set
DATASET_ROOT = "/kaggle/input/datasets/dilniwijesinghe/scotch-bonnet-dataset/Data_Set"
OUTPUT_ROOT  = "/kaggle/working/harvextro_yolov8"

CLASSES = ["green_chili", "red_chili", "yellow_chili"]  # must match _classes.txt order

# create output dirs
for split in ["train", "valid", "test"]:
    for sub in ["images", "labels"]:
        Path(f"{OUTPUT_ROOT}/{split}/{sub}").mkdir(parents=True, exist_ok=True)
