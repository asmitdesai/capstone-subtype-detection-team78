"""
Reproduces the 'Final Model' cell of code/preprocessing/ThyroidDataset.ipynb
(cell 19) standalone, with all 4 label corrections from
docs/label_corrections_log.txt applied, and rebuilds Dataset/Datasets/FinalData.npy.

Corrections are applied AFTER the train/test split (not before), directly to
each patient's recorded meth_split partition (cvPTC_train/cvPTC_test/etc.).
This matches the log's own meth_split bookkeeping exactly and avoids any risk
of a relabel changing which split a patient's row lands in (applying before
the split would re-shuffle that patient's position in its class pool and
isn't guaranteed to preserve the original train/test membership).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import RAW_THYROID_PATH, THYROID_PATH

cvPTC_path_unprocessed = Path(RAW_THYROID_PATH, 'cvPTC_beta_values_unprocessed.txt')
fvPTC_path_unprocessed = Path(RAW_THYROID_PATH, 'fvPTC_beta_values_unprocessed.txt')
norm_path_unprocessed = Path(RAW_THYROID_PATH, 'norm_beta_values_unprocessed.txt')

cvPTC = pd.read_csv(cvPTC_path_unprocessed, sep='\t', index_col=0)
cvPTC.set_index('ProbeID', inplace=True)
cvPTC = cvPTC.T
cvPTC['cancer'] = 1
cvPTC['follicolar'] = 0
cvPTC['type'] = 'classic'

fvPTC = pd.read_csv(fvPTC_path_unprocessed, sep='\t', index_col=0)
fvPTC.set_index('ProbeID', inplace=True)
fvPTC = fvPTC.T
fvPTC['cancer'] = 1
fvPTC['follicolar'] = 1
fvPTC['type'] = 'follicolar'

normal = pd.read_csv(norm_path_unprocessed, sep='\t', index_col=0)
normal.set_index('ProbeID', inplace=True)
normal = normal.T
normal['cancer'] = 0
normal['follicolar'] = 0
normal['type'] = 'normal'

print(f'cvPTC: {len(cvPTC)}, fvPTC: {len(fvPTC)}, normal: {len(normal)}')

cvPTC_train, cvPTC_test = train_test_split(cvPTC, test_size=0.2, random_state=2569)
fvPTC_train, fvPTC_test = train_test_split(fvPTC, test_size=0.2, random_state=2569)
normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=2569)


def _relabel_cv_to_fv(cv_df, fv_df, barcode, split_name):
    if barcode in cv_df.index:
        row = cv_df.loc[[barcode]].copy()
        row['follicolar'] = 1
        row['type'] = 'follicolar'
        cv_df = cv_df.drop(index=barcode)
        fv_df = pd.concat([fv_df, row])
        print(f'Relabeled {barcode} ({split_name}): cvPTC -> fvPTC')
    else:
        print(f'WARNING: {barcode} not found in cvPTC {split_name} index -- relabel not applied')
    return cv_df, fv_df


def _exclude(cv_df, barcode, split_name):
    if barcode in cv_df.index:
        cv_df = cv_df.drop(index=barcode)
        print(f'Excluded {barcode} ({split_name}): genuinely unresolved label disagreement')
    else:
        print(f'WARNING: {barcode} not found in cvPTC {split_name} index -- exclusion not applied')
    return cv_df


# Apply all 4 corrections from docs/label_corrections_log.txt, each to its
# recorded meth_split partition.
cvPTC_train, fvPTC_train = _relabel_cv_to_fv(cvPTC_train, fvPTC_train, 'TCGA-FY-A4B0-01A_cvPTC', 'train')
cvPTC_test, fvPTC_test = _relabel_cv_to_fv(cvPTC_test, fvPTC_test, 'TCGA-BJ-A45K-01A_cvPTC', 'test')
cvPTC_test, fvPTC_test = _relabel_cv_to_fv(cvPTC_test, fvPTC_test, 'TCGA-J8-A3O0-01A_cvPTC', 'test')
cvPTC_test = _exclude(cvPTC_test, 'TCGA-FY-A3R9-01A_cvPTC', 'test')

cancer_train = pd.concat([cvPTC_train, fvPTC_train, normal_train])
cancer_test = pd.concat([cvPTC_test, fvPTC_test, normal_test])
PTC_train = pd.concat([cvPTC_train, fvPTC_train])
PTC_test = pd.concat([cvPTC_test, fvPTC_test])

X_cancer_train, y_cancer_train = shuffle(
    cancer_train.drop(['cancer', 'follicolar', 'type'], axis=1).to_numpy().astype(np.float32),
    cancer_train['cancer'].to_numpy().astype(np.float32), random_state=2569)
X_cancer_test, y_cancer_test = shuffle(
    cancer_test.drop(['cancer', 'follicolar', 'type'], axis=1).to_numpy().astype(np.float32),
    cancer_test['cancer'].to_numpy().astype(np.float32), random_state=2569)

X_subtype_train, y_subtype_train = shuffle(
    PTC_train.drop(['cancer', 'follicolar', 'type'], axis=1).to_numpy().astype(np.float32),
    PTC_train['follicolar'].to_numpy().astype(np.float32), random_state=2569)
X_subtype_test, y_subtype_test = shuffle(
    PTC_test.drop(['cancer', 'follicolar', 'type'], axis=1).to_numpy().astype(np.float32),
    PTC_test['follicolar'].to_numpy().astype(np.float32), random_state=2569)

print(f'X_subtype_train: {X_subtype_train.shape}, cvPTC={int((y_subtype_train==0).sum())}, fvPTC={int((y_subtype_train==1).sum())}')
print(f'X_subtype_test: {X_subtype_test.shape}, cvPTC={int((y_subtype_test==0).sum())}, fvPTC={int((y_subtype_test==1).sum())}')

FinalPath = Path(THYROID_PATH, 'FinalData.npy')
with open(FinalPath, 'wb') as f:
    np.savez(f, X_cancer_train=X_cancer_train, y_cancer_train=y_cancer_train,
              X_cancer_test=X_cancer_test, y_cancer_test=y_cancer_test,
              X_subtype_train=X_subtype_train, y_subtype_train=y_subtype_train,
              X_subtype_test=X_subtype_test, y_subtype_test=y_subtype_test)

print(f'Wrote {FinalPath}')
