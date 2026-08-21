"""
Step 15: PCA component-count sweep under the existing seeded harness.

PCA was locked at 50 components after fixing the original collapse bug
(step2/step3), but 50 itself was never swept as a hyperparameter under the
25-run seeded harness (step5_seed_stability_harness.seeded_cv). This reuses
that exact harness and the exact filtered-probe-set embeddings + Dense(16)
head config that produced the chosen result (MCC 0.5289+/-0.1117, AUC
0.8471+/-0.0434, recall 0.7299+/-0.1198 -- see step8c_compare_probesets.py),
varying only pca_dim.

This is a CV-only sweep -- per the plan, the one-shot held-out test
evaluation is NOT re-run for every component count, only flagged as a
candidate (for the user to approve before burning the one-shot test) if a
component count beats 50's mean MCC by more than roughly one std of the
50-component CV result.
"""
import numpy as np
from pathlib import Path

from step5_seed_stability_harness import seeded_cv, summarize, SEEDS
from config import THYROID_PATH

HEAD_CFG_NO_PCA = dict(hidden_dim=16, l2=0.01, dropout=0.3, lr=1e-3)
PCA_GRID = [20, 35, 50, 75, 100, 150]

filtered_data = np.load(Path(THYROID_PATH, 'FilteredSubtypeFinalData.npy'))
y_filtered = filtered_data['y_subtype_train'].astype(int)
embeddings_filtered = np.load(Path(THYROID_PATH, 'filtered_subtype_train_embeddings.npy')).astype(np.float32)
print(f'Filtered-probe-set embeddings: {embeddings_filtered.shape}, '
      f'cvPTC={int((y_filtered==0).sum())}, fvPTC={int((y_filtered==1).sum())}')

if __name__ == '__main__':
    results = {}
    for pca_dim in PCA_GRID:
        cfg = dict(pca_dim=pca_dim, **HEAD_CFG_NO_PCA)
        print(f'\n=== pca_dim={pca_dim} ===')
        mcc, auc, recall1 = seeded_cv(cfg, seeds=SEEDS, embeddings=embeddings_filtered, y=y_filtered)
        results[pca_dim] = summarize(f'pca_dim={pca_dim}', mcc, auc, recall1)

    print('\n=== Step 2 PCA sweep summary (filtered probe set + Dense(16) head, 25 runs each) ===')
    print(f'{"pca_dim":>8} | {"MCC mean+/-std":>18} | {"AUC mean+/-std":>18} | {"Recall mean+/-std":>18} | collapsed')
    for pca_dim in PCA_GRID:
        s = results[pca_dim]
        print(f'{pca_dim:>8} | {s["mcc_mean"]:.4f}+/-{s["mcc_std"]:.4f}    '
              f'| {s["auc_mean"]:.4f}+/-{s["auc_std"]:.4f}    '
              f'| {s["recall_mean"]:.4f}+/-{s["recall_std"]:.4f}    '
              f'| {s["n_collapsed"]}/{s["n_runs"]}')

    baseline = results[50]
    print(f'\nBaseline (pca_dim=50): MCC {baseline["mcc_mean"]:.4f} +/- {baseline["mcc_std"]:.4f}')
    candidates = []
    for pca_dim in PCA_GRID:
        if pca_dim == 50:
            continue
        s = results[pca_dim]
        delta = s['mcc_mean'] - baseline['mcc_mean']
        if delta > baseline['mcc_std']:
            candidates.append((pca_dim, s, delta))
            print(f'  pca_dim={pca_dim}: MCC {s["mcc_mean"]:.4f} beats baseline by {delta:.4f} '
                  f'(> baseline std {baseline["mcc_std"]:.4f}) -- CANDIDATE, needs approval before test-set eval.')
        else:
            print(f'  pca_dim={pca_dim}: MCC {s["mcc_mean"]:.4f} (delta {delta:+.4f} vs baseline) -- not a clear candidate.')

    if not candidates:
        print('\nNo component count clearly beats the locked-in 50 by more than one baseline std. '
              'No re-run of the one-shot held-out test evaluation is warranted.')
    else:
        print(f'\n{len(candidates)} candidate(s) found -- ask before re-running the one-shot test evaluation with any of these.')

    import json
    out = {str(k): v for k, v in results.items()}
    out_path = Path('code', 'thyroid', 'pca_component_sweep_result.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'\nSaved sweep result to {out_path.resolve()}')
