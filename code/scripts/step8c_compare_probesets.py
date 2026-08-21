"""
Stage 3, final comparison: runs the same seeded-harness CV protocol
(step5_seed_stability_harness.seeded_cv, best head config from step4) against
the raw/full 485,577-probe embeddings (baseline) and the two alternate
probe-set embeddings (filtered ~394k, differential ~10.9k), to see whether
either alternate probe set beats the raw baseline established in
Prerequisite A (MCC 0.5039 +/- 0.0864).
"""
import numpy as np
from pathlib import Path

from step5_seed_stability_harness import seeded_cv, summarize, SEEDS
from config import THYROID_PATH

BEST_HEAD_CFG = dict(pca_dim=50, hidden_dim=16, l2=0.01, dropout=0.3, lr=1e-3)

data = np.load(Path(THYROID_PATH, 'FinalData.npy'))
y_raw = data['y_subtype_train'].astype(int)
embeddings_raw = np.load(Path(THYROID_PATH, 'subtype_train_embeddings.npy')).astype(np.float32)

filtered_data = np.load(Path(THYROID_PATH, 'FilteredSubtypeFinalData.npy'))
y_filtered = filtered_data['y_subtype_train'].astype(int)
embeddings_filtered = np.load(Path(THYROID_PATH, 'filtered_subtype_train_embeddings.npy')).astype(np.float32)

diff_data = np.load(Path(THYROID_PATH, 'DifferentialSubtypeFinalData.npy'))
y_diff = diff_data['y_subtype_train'].astype(int)
embeddings_diff = np.load(Path(THYROID_PATH, 'differential_subtype_train_embeddings.npy')).astype(np.float32)

# Sanity: labels should be identical across all three (same 368 patients,
# same corrections applied, only the probe set fed to the CNN differs).
assert np.array_equal(y_raw, y_filtered), 'label mismatch: raw vs filtered'
assert np.array_equal(y_raw, y_diff), 'label mismatch: raw vs differential'
print('Label arrays match across raw/filtered/differential (as expected).')

if __name__ == '__main__':
    results = {}
    for name, emb, y in [('raw (485,577 probes)', embeddings_raw, y_raw),
                          ('filtered (394,337 probes)', embeddings_filtered, y_filtered),
                          ('differential (10,899 probes)', embeddings_diff, y_diff)]:
        print(f'\n=== {name} ===')
        mcc, auc, recall1 = seeded_cv(BEST_HEAD_CFG, seeds=SEEDS, embeddings=emb, y=y)
        results[name] = summarize(name, mcc, auc, recall1)

    print('\n=== Stage 3 summary (best MCC first) ===')
    for name, s in sorted(results.items(), key=lambda kv: -kv[1]['mcc_mean']):
        print(f"{name}: MCC={s['mcc_mean']:.4f}+/-{s['mcc_std']:.4f} "
              f"AUC={s['auc_mean']:.4f}+/-{s['auc_std']:.4f} "
              f"recall={s['recall_mean']:.4f}+/-{s['recall_std']:.4f} "
              f"collapsed={s['n_collapsed']}/{s['n_runs']}")
