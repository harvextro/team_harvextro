# ─── CONFIGURE PATHS ────────────────────────────────────────────
# Kaggle: /kaggle/input/<your-dataset-name>/Data_Set
DATASET_ROOT = "/kaggle/input/datasets/dilniwijesinghe/scotch-bonnet-dataset/Data_Set"
OUTPUT_ROOT  = "/kaggle/working/harvextro_yolov8"

CLASSES = ["green_chili", "red_chili", "yellow_chili"]  # must match _classes.txt order

# create output dirs
for split in ["train", "valid", "test"]:
    for sub in ["images", "labels"]:
        Path(f"{OUTPUT_ROOT}/{split}/{sub}").mkdir(parents=True, exist_ok=True)

print("✅ Output directories ready")
print(f"   {OUTPUT_ROOT}/")
for s in ["train","valid","test"]:
    print(f"   ├─ {s}/images/  &  {s}/labels/")


# ─── ANNOTATION FORMAT CONVERTER ────────────────────────────────
# Source format (Darknet Keras / Roboflow):
#   <filename> x1,y1,x2,y2,class_id   (absolute pixel coords)
# Target format (YOLOv8):
#   class_id cx cy w h                 (normalised 0-1)

def convert_annotation(txt_path: str, img_path: str, out_path: str) -> int:
    """Convert one annotation file. Returns number of boxes written."""
    try:
        img = Image.open(img_path)
        W, H = img.size
    except Exception:
        return 0

    lines_out = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()          # strips \r\n, \n, spaces
            if not line:
                continue
            parts = line.split()
            # last token is coords; first token may be filename
            coords_str = parts[-1] if len(parts) > 1 else parts[0]
            if coords_str.endswith(".jpg") or coords_str.endswith(".png"):
                continue                 # line is filename only, skip
            try:
                x1, y1, x2, y2, cls = map(int, coords_str.split(","))
            except ValueError as e:
                print(f"  \u26a0\ufe0f  Parse error in {txt_path}: {repr(coords_str)} \u2192 {e}")
                continue
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

     with open(out_path, "w") as f:
        f.write("\n".join(lines_out))
    return len(lines_out)

# ─── COPY + CONVERT ALL SPLITS ──────────────────────────────────
stats = {}
for split in ["train", "valid", "test"]:
    src_dir = Path(DATASET_ROOT) / split
    if not src_dir.exists():
        print(f"\u26a0\ufe0f  {split}/ not found \u2013 skipping")
        continue
    imgs = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    n_imgs, n_boxes = 0, 0
    for img_path in tqdm(imgs, desc=split):
        stem = img_path.stem
        txt_path = src_dir / f"{stem}.txt"
        dst_img  = f"{OUTPUT_ROOT}/{split}/images/{img_path.name}"
        dst_lbl  = f"{OUTPUT_ROOT}/{split}/labels/{stem}.txt"
        shutil.copy(img_path, dst_img)
        if txt_path.exists():
            n_boxes += convert_annotation(str(txt_path), str(img_path), dst_lbl)
        n_imgs += 1
    stats[split] = {"images": n_imgs, "boxes": n_boxes}
    print(f"  {split}: {n_imgs} images, {n_boxes} boxes")

print("\n\u2705 Dataset converted to YOLOv8 format")
print("Stats:", stats)

# ─── WRITE dataset.yaml ─────────────────────────────────────────
yaml_cfg = {
    "path" : OUTPUT_ROOT,
    "train": "train/images",
    "val"  : "valid/images",
    "test" : "test/images",
    "nc"   : len(CLASSES),
    "names": CLASSES
}
yaml_path = f"{OUTPUT_ROOT}/dataset.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(yaml_cfg, f, default_flow_style=False)

print("dataset.yaml contents:")
