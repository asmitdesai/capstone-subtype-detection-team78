# download_benign.py
# Run from D:\Capstone_Team78\
# pip install pandas openpyxl requests Pillow

import pandas as pd
import requests
import time
from pathlib import Path
from PIL import Image

# ── CONFIG ───────────────────────────────────────────────────────────────────
EXCEL_PATH = r"D:\Capstone_Team78\Dataset\Datasets\ibia_metadata.xlsx"
SAVE_ROOT  = Path(r"D:\Capstone_Team78\Dataset\Benign_TIFF_Images")
PNG_ROOT   = Path(r"D:\Capstone_Team78\Dataset\Benign_PNG_Images")
BASE_URL   = "https://ibdc.dbt.gov.in"

# Only these subfolders — MT_Discard and all STAGE2/STAGE3 excluded
BENIGN_SUBFOLDERS = [
    "MT_FND",
    "MT_Follicles",
    "MT_Papillae",
    "ET_FND",
    "ET_Follicles",
    "ET_Pappilae",   # note the typo in their data — double-p
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research/Academic; Team78 PES University capstone)"
}
# ─────────────────────────────────────────────────────────────────────────────

def get_class_label(path_str):
    for folder in BENIGN_SUBFOLDERS:
        if f"/{folder}/" in path_str:
            return folder
    return None

def download_and_convert(dl_url, tiff_path, png_path):
    # Skip if PNG already exists (resume-safe)
    if png_path.exists():
        return "skipped"
    try:
        r = requests.get(dl_url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        tiff_path.parent.mkdir(parents=True, exist_ok=True)
        tiff_path.write_bytes(r.content)
        # Convert to PNG immediately, delete tiff to save space
        img = Image.open(tiff_path).convert("RGB")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(png_path, "PNG")
        tiff_path.unlink()
        return "ok"
    except Exception as e:
        return f"error: {e}"

# ── LOAD & FILTER ─────────────────────────────────────────────────────────────
print("Loading metadata...")
df = pd.read_excel(EXCEL_PATH)
print(f"Total rows in Excel: {len(df)}")

# Filter to STAGE1_BTID only, exclude MT_Discard
stage1_mask  = df["Image File Path"].str.contains("STAGE1_BTID", na=False)
discard_mask = df["Image File Path"].str.contains("MT_Discard", na=False)
benign_df    = df[stage1_mask & ~discard_mask].copy()

print(f"Benign images to download (excl. Discard): {len(benign_df)}")

# Show breakdown
for folder in BENIGN_SUBFOLDERS:
    count = benign_df["Image File Path"].str.contains(f"/{folder}/").sum()
    print(f"  {folder}: {count}")

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
downloaded, skipped, errors = 0, 0, 0
total = len(benign_df)
error_log = []

print(f"\nStarting download to {PNG_ROOT}")
print("(Downloads as TIFF, converts to PNG, deletes TIFF — saves ~40% disk space)\n")

for i, row in enumerate(benign_df.itertuples(index=False), 1):
    rel_path = row._1  # 'Image File Path' column

    dl_url = (
        f"{BASE_URL}/ibia/media/data/IBDCI_100003/IBIA/"
        f"IBIAP_1000000010/HISTOS_1000000014/{rel_path}"
    )

    class_label = get_class_label(rel_path)
    if class_label is None:
        continue

    filename  = Path(rel_path).stem  # e.g. MT_FND_001
    tiff_path = SAVE_ROOT / class_label / (filename + ".tiff")
    png_path  = PNG_ROOT  / class_label / (filename + ".png")

    result = download_and_convert(dl_url, tiff_path, png_path)

    if result == "ok":
        downloaded += 1
    elif result == "skipped":
        skipped += 1
    else:
        errors += 1
        error_log.append(f"{rel_path}: {result}")

    if i % 200 == 0:
        pct = (i / total) * 100
        print(f"  [{pct:.1f}%] {i}/{total} | ✓{downloaded} ↷{skipped} ✗{errors}")

    time.sleep(0.2)  # polite rate limit — ~5 req/sec

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"DONE. Downloaded={downloaded}, Skipped={skipped}, Errors={errors}")
print(f"PNGs saved to: {PNG_ROOT}")

if error_log:
    log_path = Path(r"D:\Capstone_Team78\download_errors.txt")
    log_path.write_text("\n".join(error_log))
    print(f"Error log: {log_path}")

# Final count per class
print("\nFinal PNG counts per class:")
for folder in BENIGN_SUBFOLDERS:
    class_dir = PNG_ROOT / folder
    if class_dir.exists():
        count = len(list(class_dir.glob("*.png")))
        print(f"  {folder}: {count} images")