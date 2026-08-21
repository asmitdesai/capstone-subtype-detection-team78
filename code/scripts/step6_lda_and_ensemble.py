"""
Stage 1a/1b of the fine-tuning plan: LDA as an alternate/hybrid reducer, and
a soft-vote ensemble of the three model families already proven
non-collapsing on PCA(50)-reduced frozen embeddings (Dense NN, SVC, QDA;
see MethylationRemediation.ipynb). Evaluated under the Stage-0 seeded
harness (Dense NN weight init is unseeded otherwise; SVC/QDA are
deterministic given a fixed input, so only the Dense NN component needs
repeat-seeding -- but we still repeat the whole ensemble per seed for a
fair mean+/-std comparison against the other stages).

All reducers (PCA, LDA) are fit per-fold on the TRAINING split only and
only .transform()-ed on the validation split -- no leakage.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
import tf_keras as keras
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score

from step4_cnn_bottleneck_model import build_head
from config import THYROID_PATH

data = np.load(Path(THYROID_PATH, 'FinalData.npy'))
y = data['y_subtype_train'].astype(int)
embeddings = np.load(Path(THYROID_PATH, 'subtype_train_embeddings.npy')).astype(np.float32)
print(f'Embeddings: {embeddings.shape}, cvPTC(0)={int((y==0).sum())}, fvPTC(1)={int((y==1).sum())}')

BEST_HEAD_CFG = dict(pca_dim=50, hidden_dim=16, l2=0.01, dropout=0.3, lr=1e-3)
SEEDS = [0, 1, 2, 3, 4]


def eval_preds(y_val, y_proba, name, fold):
    y_pred = (y_proba >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_proba)
    except ValueError:
        auc = float('nan')
    recall1 = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    print(f'    Fold {fold+1} [{name}]: MCC={mcc:.4f} AUC={auc:.4f} fvPTC_recall={recall1:.4f}')
    return mcc, auc, recall1


# ---------------------------------------------------------------------------
# Stage 1a: LDA as a second reducer (1-dim standalone, and PCA+LDA hybrid)
# ---------------------------------------------------------------------------
def run_lda_variants():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    variants = {'LDA-only (1D) + SVC': [], 'PCA(50)+LDA hybrid (51D) + SVC': [],
                'PCA(50) only (51D->50D baseline) + SVC': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, y)):
        X_train_emb, X_val_emb = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pca = PCA(n_components=50, random_state=42)
        X_train_pca = pca.fit_transform(X_train_emb)
        X_val_pca = pca.transform(X_val_emb)

        lda = LinearDiscriminantAnalysis()
        X_train_lda = lda.fit_transform(X_train_emb, y_train)
        X_val_lda = lda.transform(X_val_emb)

        X_train_hybrid = np.hstack([X_train_pca, X_train_lda])
        X_val_hybrid = np.hstack([X_val_pca, X_val_lda])

        for name, X_train, X_val in [
            ('LDA-only (1D) + SVC', X_train_lda, X_val_lda),
            ('PCA(50)+LDA hybrid (51D) + SVC', X_train_hybrid, X_val_hybrid),
            ('PCA(50) only (51D->50D baseline) + SVC', X_train_pca, X_val_pca),
        ]:
            clf = SVC(C=0.2, class_weight='balanced', probability=True)
            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_val)[:, 1]
            mcc, auc, recall1 = eval_preds(y_val, y_proba, name, fold)
            variants[name].append((mcc, auc, recall1))

    print('\n=== Stage 1a summary: LDA variants (single run, SVC deterministic given fixed input) ===')
    for name, vals in variants.items():
        mccs = np.array([v[0] for v in vals])
        aucs = np.array([v[1] for v in vals])
        recalls = np.array([v[2] for v in vals])
        print(f'{name}: MCC={mccs.mean():.4f} AUC={aucs.mean():.4f} recall={recalls.mean():.4f}')
    return variants


# ---------------------------------------------------------------------------
# Stage 1b: soft-vote ensemble of Dense NN + SVC + QDA on PCA(50) embeddings
# ---------------------------------------------------------------------------
def run_ensemble_once(seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {'dense_nn': [], 'svc': [], 'qda': [],
                     'ensemble_equal': [], 'ensemble_weighted': [], 'ensemble_nn_svc': []}

    # Fixed a priori weights from MethylationRemediation.ipynb's prior CV AUCs
    # (SVC 0.844, Dense NN ~0.83, QDA 0.707) -- NOT tuned on this run's val folds.
    w = {'dense_nn': 0.83, 'svc': 0.844, 'qda': 0.707}
    w_sum = sum(w.values())
    w = {k: v / w_sum for k, v in w.items()}

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, y)):
        X_train_emb, X_val_emb = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pca = PCA(n_components=50, random_state=42)
        X_train = pca.fit_transform(X_train_emb).astype(np.float32)
        X_val = pca.transform(X_val_emb).astype(np.float32)

        from sklearn.utils import class_weight
        cw_vals = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw_dict = dict(enumerate(cw_vals))

        keras.backend.clear_session()
        keras.utils.set_random_seed(seed * 100 + fold)
        nn = build_head(BEST_HEAD_CFG['pca_dim'], BEST_HEAD_CFG['hidden_dim'],
                         BEST_HEAD_CFG['l2'], BEST_HEAD_CFG['dropout'])
        nn.compile(loss='sparse_categorical_crossentropy',
                   optimizer=keras.optimizers.Adam(learning_rate=BEST_HEAD_CFG['lr']),
                   metrics=['accuracy'])
        from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=0)
        nn.fit(X_train, y_train, batch_size=8, epochs=150, validation_data=(X_val, y_val),
               callbacks=[es, rlr], class_weight=cw_dict, verbose=0)
        proba_nn = nn.predict(X_val, verbose=0)[:, 1]

        svc = SVC(C=0.2, class_weight='balanced', probability=True)
        svc.fit(X_train, y_train)
        proba_svc = svc.predict_proba(X_val)[:, 1]

        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        proba_qda = qda.predict_proba(X_val)[:, 1]

        proba_equal = (proba_nn + proba_svc + proba_qda) / 3.0
        proba_weighted = w['dense_nn'] * proba_nn + w['svc'] * proba_svc + w['qda'] * proba_qda
        # Follow-up: QDA (weakest of the 3) dragged the 3-way ensemble below
        # either individual strong model -- try dropping it.
        proba_nn_svc = (proba_nn + proba_svc) / 2.0

        for name, proba in [('dense_nn', proba_nn), ('svc', proba_svc), ('qda', proba_qda),
                             ('ensemble_equal', proba_equal), ('ensemble_weighted', proba_weighted),
                             ('ensemble_nn_svc', proba_nn_svc)]:
            mcc, auc, recall1 = eval_preds(y_val, proba, name, fold)
            fold_results[name].append((mcc, auc, recall1))

    return fold_results


def run_ensemble(seeds=SEEDS):
    all_results = {k: [] for k in ['dense_nn', 'svc', 'qda', 'ensemble_equal', 'ensemble_weighted', 'ensemble_nn_svc']}
    for seed in seeds:
        print(f'  -- seed {seed} --')
        fold_results = run_ensemble_once(seed)
        for k in all_results:
            all_results[k].extend(fold_results[k])

    print('\n=== Stage 1b summary: ensemble vs. individual models (mean +/- std over seeds x folds) ===')
    for name, vals in all_results.items():
        mccs = np.array([v[0] for v in vals])
        aucs = np.array([v[1] for v in vals])
        recalls = np.array([v[2] for v in vals])
        print(f'{name}: MCC={mccs.mean():.4f}+/-{mccs.std():.4f} AUC={aucs.mean():.4f}+/-{aucs.std():.4f} '
              f'recall={recalls.mean():.4f}+/-{recalls.std():.4f}')
    return all_results


if __name__ == '__main__':
    import sys
    if '--skip-lda' not in sys.argv:
        print('=== Stage 1a: LDA variants ===')
        run_lda_variants()

    print('\n=== Stage 1b: ensemble ===')
    run_ensemble()
