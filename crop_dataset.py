import os, random
from pathlib import Path
from PIL import Image

LEFT, TOP, RIGHT, BOTTOM = 0.28, 0.04, 0.72, 0.40
SRC_ROOT = Path("Dataset")
DST_ROOT = Path("Dataset_cropped")
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def crop_chest(img):
    w, h = img.size
    return img.crop((int(w*LEFT), int(h*TOP), int(w*RIGHT), int(h*BOTTOM)))

def process_dir(src, dst):
    total = 0
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir(): continue
        out = dst / cls_dir.name
        out.mkdir(parents=True, exist_ok=True)
        imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXT]
        print(f"  [{cls_dir.name}] {len(imgs)} images")
        for p in sorted(imgs):
            try:
                with Image.open(p).convert("RGB") as img:
                    fmt = p.suffix.lstrip(".").upper()
                    fmt = "JPEG" if fmt in ("JPG","JPEG") else ("PNG" if fmt not in ("PNG","WEBP","BMP") else fmt)
                    crop_chest(img).save(out / p.name, format=fmt)
                total += 1
            except Exception as e:
                print(f"    WARNING {p.name}: {e}")
    return total

print("Cropping Dataset/ ->", DST_ROOT)
n = process_dir(SRC_ROOT, DST_ROOT)
print(f"Done: {n} images written to {DST_ROOT}/")

split = Path("data_split")
if split.exists():
    out_split = Path("data_split_cropped")
    total2 = 0
    for s in ["train","val","test"]:
        total2 += process_dir(split/s, out_split/s)
    print(f"Also wrote {total2} images to data_split_cropped/")
    print("Update your script: DATA_DIR = 'data_split_cropped'")
else:
    print("No data_split/ found - run prepare_dataset() first, then re-run this.")
