"""
Prerequisite A of the fine-tuning plan: a seeded, repeated-CV evaluation
harness.

Neither numpy nor TF/Keras random seeds were fixed anywhere in
step4_cnn_bottleneck_model.py (only StratifiedKFold and PCA were seeded) --
only the fold *split* was reproducible, not the Dense head's weight
init/dropout. This is why re-running the same hyperparameter sweep twice
picked two different "winning" configs (both far above baseline, but not a
fair basis for comparing configs against each other).

This script wraps step4's run_cv() in an outer loop over a fixed list of
seeds, keeping StratifiedKFold's split fixed (random_state=42 throughout) so
only training randomness varies across repeats. For a given config it
reports mean +/- std of MCC/AUC/fvPTC-recall over `n_folds x n_seeds` runs.
Every later stage in the fine-tuning plan (Stage 1a/1b/1c, Stage 3, Stage 4)
should be compared under this harness, not a single run, before being
called an improvement.
"""
import numpy as np

from step4_cnn_bottleneck_model import run_cv

SEEDS = [0, 1, 2, 3, 4]


def seeded_cv(config, seeds=SEEDS, **run_cv_kwargs):
    """Runs run_cv(**config) once per seed, returns per-(seed,fold) arrays."""
    all_mcc, all_auc, all_recall1 = [], [], []
    for seed in seeds:
        print(f'  -- seed {seed} --')
        fold_mcc, fold_auc, fold_recall1 = run_cv(
            **config, seed=seed, return_folds=True, **run_cv_kwargs)
        all_mcc.extend(fold_mcc)
        all_auc.extend(fold_auc)
        all_recall1.extend(fold_recall1)
    return np.array(all_mcc), np.array(all_auc), np.array(all_recall1)


def summarize(name, mcc, auc, recall1, n_seeds=len(SEEDS)):
    n_folds = len(mcc) // n_seeds
    n_collapsed = int(np.sum(mcc == 0.0))
    print(f'{name}: n_runs={len(mcc)} (={n_folds} folds x {n_seeds} seeds)')
    print(f'  MCC:    {mcc.mean():.4f} +/- {mcc.std():.4f}  (min={mcc.min():.4f}, max={mcc.max():.4f})')
    print(f'  AUC:    {auc.mean():.4f} +/- {auc.std():.4f}')
    print(f'  Recall: {recall1.mean():.4f} +/- {recall1.std():.4f}')
    print(f'  Runs with exact MCC=0.0 (potential collapse): {n_collapsed}/{len(mcc)}')
    return dict(name=name, mcc_mean=mcc.mean(), mcc_std=mcc.std(),
                auc_mean=auc.mean(), auc_std=auc.std(),
                recall_mean=recall1.mean(), recall_std=recall1.std(),
                n_collapsed=n_collapsed, n_runs=len(mcc))


if __name__ == '__main__':
    # Baseline: the current best config from step4's sweep, evaluated under
    # the seeded harness for a trustworthy reference number.
    baseline_config = dict(pca_dim=50, hidden_dim=16, l2=0.01, dropout=0.3, lr=1e-3)
    print(f'=== Seeded CV baseline: {baseline_config} ===')
    mcc, auc, recall1 = seeded_cv(baseline_config)
    summary = summarize('baseline (pca50/hid16/l2.01/do.3/lr1e-3)', mcc, auc, recall1)

    print('\n=== Summary ===')
    print(summary)
