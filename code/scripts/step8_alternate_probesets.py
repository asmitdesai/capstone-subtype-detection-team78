"""
Stage 3 of the fine-tuning plan: build train/test splits for the two
alternate probe sets (filtered ~394k probes, differential ~10.9k probes)
that are exact, leakage-free, corrections-applied equivalents of
FinalData.npy's 368-train/92-test split -- so the existing
step2_embedding_diagnostic.py -> step4_cnn_bottleneck_model.py pipeline can
be pointed at them unchanged.

Dataset\\Datasets\\FilteredSubtypeData.npy / DifferentialSubtypeData.npy do
NOT store patient barcodes (only X/y/feature_names), so they cannot be
corrected/split by barcode directly. Rather than trust their row order,
this script rebuilds both probe sets directly from the same raw source files
rebuild_final_data.py uses, keeping the pandas barcode index throughout, so
the split+correction logic (identical to rebuild_final_data.py) is applied
by patient identity, not row position.

Two source quirks handled here:
- The *processed* beta-value files (used for the "filtered" probe set) have
  their sample-barcode columns using '.' instead of '-' (e.g.
  'TCGA.FY.A4B0.01A_cvPTC' vs the unprocessed files' 'TCGA-FY-A4B0-01A_cvPTC'
  -- an R-style column-name sanitization artifact). Normalized back to '-'
  before use so barcodes match the unprocessed files and the correction list.
- Both probe sets are zero-padded/reindexed to the same 485,577-probe master
  column list the frozen CNN backbone expects (matching FinalData.npy's
  width), via the unprocessed 'normal' file's probe columns as the
  reference ordering (same convention as ThyroidDataset.ipynb).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import RAW_THYROID_PATH, THYROID_PATH

RELABEL_TO_FV = {
    'train': ['TCGA-FY-A4B0-01A_cvPTC'],
    'test': ['TCGA-BJ-A45K-01A_cvPTC', 'TCGA-J8-A3O0-01A_cvPTC'],
}
EXCLUDE = {'test': ['TCGA-FY-A3R9-01A_cvPTC']}


def split_and_correct(cvPTC, fvPTC):
    """Identical split+correction logic to rebuild_final_data.py (per-class
    split first, corrections applied to the resulting partitions after),
    factored out so both probe sets use exactly the same procedure. cvPTC/
    fvPTC must already carry a 'follicolar' column (0/1)."""
    cvPTC_train, cvPTC_test = train_test_split(cvPTC, test_size=0.2, random_state=2569)
    fvPTC_train, fvPTC_test = train_test_split(fvPTC, test_size=0.2, random_state=2569)

    def relabel(cv_df, fv_df, barcode, split_name):
        if barcode in cv_df.index:
            row = cv_df.loc[[barcode]].copy()
            row['follicolar'] = 1
            cv_df = cv_df.drop(index=barcode)
            fv_df = pd.concat([fv_df, row])
        else:
            print(f'WARNING: {barcode} not found in cvPTC {split_name} index')
        return cv_df, fv_df

    def exclude(cv_df, barcode, split_name):
        if barcode in cv_df.index:
            cv_df = cv_df.drop(index=barcode)
        else:
            print(f'WARNING: {barcode} not found in cvPTC {split_name} index')
        return cv_df

    for barcode in RELABEL_TO_FV['train']:
        cvPTC_train, fvPTC_train = relabel(cvPTC_train, fvPTC_train, barcode, 'train')
    for barcode in RELABEL_TO_FV['test']:
        cvPTC_test, fvPTC_test = relabel(cvPTC_test, fvPTC_test, barcode, 'test')
    for barcode in EXCLUDE['test']:
        cvPTC_test = exclude(cvPTC_test, barcode, 'test')

    PTC_train = pd.concat([cvPTC_train, fvPTC_train])
    PTC_test = pd.concat([cvPTC_test, fvPTC_test])
    return PTC_train, PTC_test


def build_and_save(name, cvPTC, fvPTC, all_probes, out_path):
    """cvPTC/fvPTC: barcode-indexed beta-value DataFrames (any probe subset,
    'follicolar' column not yet added) for one probe set. all_probes: the
    master 485,577-probe Index to zero-pad to (matches FinalData.npy)."""
    cvPTC = cvPTC.copy()
    fvPTC = fvPTC.copy()
    cvPTC['follicolar'] = 0
    fvPTC['follicolar'] = 1

    PTC_train, PTC_test = split_and_correct(cvPTC, fvPTC)

    def to_xy(df):
        y = df['follicolar'].to_numpy().astype(np.float32)
        X = df.drop(columns=['follicolar'])
        X = X.reindex(columns=all_probes).fillna(0.0).to_numpy().astype(np.float32)
        return X, y

    X_train, y_train = to_xy(PTC_train)
    X_test, y_test = to_xy(PTC_test)
    X_train, y_train = shuffle(X_train, y_train, random_state=2569)
    X_test, y_test = shuffle(X_test, y_test, random_state=2569)

    print(f'{name}: X_subtype_train={X_train.shape} cvPTC={int((y_train==0).sum())} fvPTC={int((y_train==1).sum())}')
    print(f'{name}: X_subtype_test={X_test.shape} cvPTC={int((y_test==0).sum())} fvPTC={int((y_test==1).sum())}')

    with open(out_path, 'wb') as f:
        np.savez(f, X_subtype_train=X_train, y_subtype_train=y_train,
                  X_subtype_test=X_test, y_subtype_test=y_test)
    print(f'Wrote {out_path}')


def load_unprocessed(name):
    path = Path(RAW_THYROID_PATH, f'{name}_beta_values_unprocessed.txt')
    df = pd.read_csv(path, sep='\t', index_col=0)
    df.set_index('ProbeID', inplace=True)
    return df.T


def load_processed(name):
    path = Path(RAW_THYROID_PATH, f'{name}_beta_values_processed.txt')
    df = pd.read_csv(path, sep='\t', index_col=0).T
    df.index = df.index.str.replace('.', '-', regex=False)
    return df


if __name__ == '__main__':
    # Master probe column list (485,577 probes) that the frozen CNN backbone
    # expects -- same reference used to build FinalData.npy.
    normal_unprocessed = load_unprocessed('norm')
    all_probes = normal_unprocessed.columns
    print(f'Master probe list: {len(all_probes)} probes')

    # --- Differential probe set (unprocessed source, restricted to the
    # differentially-methylated CpG list) ---
    diff_cpg_path = Path(RAW_THYROID_PATH, 'dmCpGs_PTC_vs_Norm_logFC1_FDR05.txt')
    diff_cpg_index = list(pd.read_csv(diff_cpg_path, sep='\t', index_col=0).index)
    cvPTC_unproc = load_unprocessed('cvPTC')
    fvPTC_unproc = load_unprocessed('fvPTC')
    print(f'Differential probe list: {len(diff_cpg_index)} probes')
    build_and_save(
        'Differential', cvPTC_unproc[diff_cpg_index], fvPTC_unproc[diff_cpg_index],
        all_probes, Path(THYROID_PATH, 'DifferentialSubtypeFinalData.npy'))

    # --- Filtered probe set (processed source, already QC/probe-filtered) ---
    cvPTC_proc = load_processed('cvPTC')
    fvPTC_proc = load_processed('fvPTC')
    print(f'Filtered probe list: {cvPTC_proc.shape[1]} probes')
    build_and_save(
        'Filtered', cvPTC_proc, fvPTC_proc,
        all_probes, Path(THYROID_PATH, 'FilteredSubtypeFinalData.npy'))

    # Sanity check: rebuilding the raw/unfiltered split this same way should
    # reproduce FinalData.npy's train/test counts exactly (368/286/82,
    # 92/69/23) -- confirms split_and_correct() matches rebuild_final_data.py.
    print('\n=== Sanity check against FinalData.npy (raw/unfiltered probes) ===')
    build_and_save('Sanity-raw', cvPTC_unproc, fvPTC_unproc, all_probes,
                    Path(THYROID_PATH, '_sanity_check_raw.npy'))
