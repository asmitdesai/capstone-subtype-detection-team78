import pandas as pd
from sklearn.model_selection import train_test_split

RAW = r"D:\Capstone_Team78\Dataset\RawData"
IMG_LIST = r"D:\Capstone_Team78\Dataset\Datasets\tcga_thca_cvptc_fvptc_case_list.csv"
OUT = r"D:\Capstone_Team78\Dataset\Datasets\paired_multimodal_cases_full.csv"

def read_header_samples(path):
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
    return header[1:]  # drop 'ProbeID'

def patient_barcode(sample_id):
    return "-".join(sample_id.split("-")[:3])

cv_samples = read_header_samples(f"{RAW}\\cvPTC_beta_values_unprocessed.txt")
fv_samples = read_header_samples(f"{RAW}\\fvPTC_beta_values_unprocessed.txt")
print(f"cvPTC raw samples: {len(cv_samples)}")
print(f"fvPTC raw samples: {len(fv_samples)}")

# quick sanity check on suffix format
print("Sample cv[0:3]:", cv_samples[:3])
print("Sample fv[0:3]:", fv_samples[:3])
assert all(s.endswith("_cvPTC") for s in cv_samples), "unexpected cvPTC suffix format"
assert all(s.endswith("_fvPTC") for s in fv_samples), "unexpected fvPTC suffix format"

# reproduce the exact train/test split from ThyroidDataset.ipynb "Final Model" cell
cv_train, cv_test = train_test_split(cv_samples, test_size=0.2, random_state=2569)
fv_train, fv_test = train_test_split(fv_samples, test_size=0.2, random_state=2569)
print(f"\ncvPTC split: {len(cv_train)} train / {len(cv_test)} test")
print(f"fvPTC split: {len(fv_train)} train / {len(fv_test)} test")
print(f"Total train (should be 368): {len(cv_train) + len(fv_train)}")
print(f"Total test  (should be 93):  {len(cv_test) + len(fv_test)}")

# check for duplicate patient barcodes within each cohort (multiple vials per patient)
cv_patients = [patient_barcode(s) for s in cv_samples]
fv_patients = [patient_barcode(s) for s in fv_samples]
print(f"\ncvPTC unique patients: {len(set(cv_patients))} of {len(cv_patients)} samples")
print(f"fvPTC unique patients: {len(set(fv_patients))} of {len(fv_patients)} samples")

# build full methylation cohort table: patient_barcode, meth_label, meth_split
rows = []
for s in cv_samples:
    rows.append({"patient_barcode": patient_barcode(s), "meth_sample_id": s,
                 "meth_label": "cvPTC_classical", "meth_split": "train" if s in cv_train else "test"})
for s in fv_samples:
    rows.append({"patient_barcode": patient_barcode(s), "meth_sample_id": s,
                 "meth_label": "fvPTC_follicular_variant", "meth_split": "train" if s in fv_train else "test"})
meth_df = pd.DataFrame(rows)

dupe_patients = meth_df[meth_df.duplicated("patient_barcode", keep=False)]
if len(dupe_patients):
    print(f"\nWARNING: {dupe_patients['patient_barcode'].nunique()} patient barcodes appear more than once in methylation cohort:")
    print(dupe_patients.sort_values("patient_barcode"))
else:
    print("\nNo duplicate patient barcodes within methylation cohort (1 sample per patient).")

# load image case list
img = pd.read_csv(IMG_LIST)
print(f"\nImage case list: {len(img)} cases, columns: {img.columns.tolist()}")

# 1. full 461 cohort intersect image list
merged = meth_df.merge(img, left_on="patient_barcode", right_on="case_id", how="inner")
print(f"\n=== (1) FULL 461-sample cohort vs INTERSECT case list ===")
print(f"Total paired cases: {len(merged)}")
print(merged.groupby("meth_label").size().rename("count"))
print("\nBy image list label:")
print(merged.groupby("label").size().rename("count"))

# 2. held-out (test) subset intersect image list
heldout_paired = merged[merged["meth_split"] == "test"]
print(f"\n=== (2) HELD-OUT (test-split) methylation samples vs image list ===")
print(f"Total held-out+paired: {len(heldout_paired)}")
print(heldout_paired.groupby("meth_label").size().rename("count"))

# 3. label agreement check
merged["label_match"] = merged["meth_label"] == merged["label"]
mismatches = merged[~merged["label_match"]]
print(f"\n=== (3) Label agreement check ===")
print(f"Mismatches: {len(mismatches)} / {len(merged)}")
if len(mismatches):
    print(mismatches[["patient_barcode", "meth_sample_id", "meth_label", "label", "primary_diagnosis", "morphology"]])

# save
merged_out = merged[["patient_barcode", "meth_sample_id", "meth_label", "meth_split", "label",
                      "primary_diagnosis", "morphology", "label_match"]]
merged_out.to_csv(OUT, index=False)
print(f"\nSaved {len(merged_out)} rows to {OUT}")
