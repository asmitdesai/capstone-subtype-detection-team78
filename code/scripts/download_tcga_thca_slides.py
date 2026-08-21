"""
Download FFPE diagnostic (DX) whole-slide images for the cvPTC/fvPTC subset
selected in Dataset/tcga_thca_download_subset.csv, via IDC (Imaging Data Commons).

Run (from repo root): python code/scripts/download_tcga_thca_slides.py [csv_path] [--dry-run]
Defaults to the full subset CSV; pass a batch CSV path to run one batch,
e.g. python code/scripts/download_tcga_thca_slides.py Dataset/Datasets/download_batches/batch_01.csv
"""
import sys
import pandas as pd
from idc_index import IDCClient

DEFAULT_SUBSET_CSV = r"D:\Capstone_Team78\Dataset\tcga_thca_download_subset.csv"
DOWNLOAD_ROOT = r"D:\Capstone_Team78\Dataset\TCGA_THCA_Slides"

dry_run = "--dry-run" in sys.argv
csv_args = [a for a in sys.argv[1:] if a != "--dry-run"]
SUBSET_CSV = csv_args[0] if csv_args else DEFAULT_SUBSET_CSV

subset = pd.read_csv(SUBSET_CSV)
c = IDCClient()

for label, group in subset.groupby("label"):
    series_uids = group["SeriesInstanceUID"].tolist()
    dest = fr"{DOWNLOAD_ROOT}\{label}"
    print(f"\n{label}: {len(series_uids)} DX slides -> {dest}")
    c.download_dicom_series(
        seriesInstanceUID=series_uids,
        downloadDir=dest,
        dry_run=dry_run,
        dirTemplate="%PatientID/%SeriesInstanceUID",
        use_s5cmd_sync=True,
    )

print("\nDone." if not dry_run else "\nDry run complete — no files written.")
