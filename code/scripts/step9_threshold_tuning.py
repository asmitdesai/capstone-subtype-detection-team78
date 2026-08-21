"""
Stage 1c of the fine-tuning plan: leakage-free per-fold decision threshold
tuning on top of the winning single-CNN-classifier head (Dense NN, best
config from step4/step5).

Nested split, strictly leakage-free: within each OUTER StratifiedKFold(5,
seed=42) fold, the outer TRAINING indices are further split into an
inner-train / inner-threshold-holdout partition. PCA(50) and the Dense NN
head are both fit on inner-train only. Thresholds are swept on the
inner-holdout predictions (never on the outer val fold) to maximize MCC.
The resulting threshold and the inner-train-fitted model are then applied,
unmodified, to the outer val fold for reporting -- the outer val fold's
labels never influence model fitting OR threshold selection.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
import tf_keras as keras
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score

from step4_cnn_bottleneck_model import build_head
from config import THYROID_PATH

data = np.load(Path(THYROID_PATH, 'FinalData.npy'))
y = data['y_subtype_train'].astype(int)
embeddings = np.load(Path(THYROID_PATH, 'subtype_train_embeddings.npy')).astype(np.float32)

BEST_HEAD_CFG = dict(pca_dim=50, hidden_dim=16, l2=0.01, dropout=0.3, lr=1e-3)
THRESHOLDS = np.arange(0.15, 0.85, 0.025)
SEEDS = [0, 1, 2, 3, 4]


def run_once(seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    default_results, tuned_results = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, y)):
        X_train_emb, X_val_emb = embeddings[train_idx], embeddings[val_idx]
        y_train_outer, y_val = y[train_idx], y[val_idx]

        inner_train_idx, inner_thresh_idx = train_test_split(
            np.arange(len(train_idx)), test_size=0.2, random_state=2569 + seed,
            stratify=y_train_outer)
        X_inner_train_emb = X_train_emb[inner_train_idx]
        X_inner_thresh_emb = X_train_emb[inner_thresh_idx]
        y_inner_train = y_train_outer[inner_train_idx]
        y_inner_thresh = y_train_outer[inner_thresh_idx]

        pca = PCA(n_components=BEST_HEAD_CFG['pca_dim'], random_state=42)
        X_inner_train = pca.fit_transform(X_inner_train_emb).astype(np.float32)
        X_inner_thresh = pca.transform(X_inner_thresh_emb).astype(np.float32)
        X_val = pca.transform(X_val_emb).astype(np.float32)

        cw_vals = class_weight.compute_class_weight('balanced', classes=np.unique(y_inner_train), y=y_inner_train)
        cw_dict = dict(enumerate(cw_vals))

        keras.backend.clear_session()
        keras.utils.set_random_seed(seed * 100 + fold)
        model = build_head(BEST_HEAD_CFG['pca_dim'], BEST_HEAD_CFG['hidden_dim'],
                            BEST_HEAD_CFG['l2'], BEST_HEAD_CFG['dropout'])
        model.compile(loss='sparse_categorical_crossentropy',
                       optimizer=keras.optimizers.Adam(learning_rate=BEST_HEAD_CFG['lr']),
                       metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=0)
        model.fit(X_inner_train, y_inner_train, batch_size=8, epochs=150,
                  validation_data=(X_inner_thresh, y_inner_thresh),
                  callbacks=[es, rlr], class_weight=cw_dict, verbose=0)

        # Sweep thresholds on the inner-holdout only -- outer val never touched here.
        proba_inner_thresh = model.predict(X_inner_thresh, verbose=0)[:, 1]
        best_thresh, best_mcc = 0.5, -1.0
        for t in THRESHOLDS:
            pred = (proba_inner_thresh >= t).astype(int)
            m = matthews_corrcoef(y_inner_thresh, pred)
            if m > best_mcc:
                best_mcc, best_thresh = m, t

        # Apply both default (0.5) and tuned threshold to the outer val fold.
        proba_val = model.predict(X_val, verbose=0)[:, 1]
        for thresh, results_list, label in [(0.5, default_results, 'default'),
                                             (best_thresh, tuned_results, 'tuned')]:
            pred = (proba_val >= thresh).astype(int)
            mcc = matthews_corrcoef(y_val, pred)
            auc = roc_auc_score(y_val, proba_val)
            recall1 = recall_score(y_val, pred, pos_label=1, zero_division=0)
            results_list.append((mcc, auc, recall1))
            print(f'    Fold {fold+1} [{label}, thresh={thresh:.3f}]: MCC={mcc:.4f} AUC={auc:.4f} recall={recall1:.4f}')

    return default_results, tuned_results


def summarize(name, results):
    mccs = np.array([r[0] for r in results])
    aucs = np.array([r[1] for r in results])
    recalls = np.array([r[2] for r in results])
    print(f'{name}: MCC={mccs.mean():.4f}+/-{mccs.std():.4f} AUC={aucs.mean():.4f}+/-{aucs.std():.4f} '
          f'recall={recalls.mean():.4f}+/-{recalls.std():.4f}')


if __name__ == '__main__':
    all_default, all_tuned = [], []
    for seed in SEEDS:
        print(f'-- seed {seed} --')
        default_results, tuned_results = run_once(seed)
        all_default.extend(default_results)
        all_tuned.extend(tuned_results)

    print('\n=== Stage 1c summary: default (0.5) vs. nested-tuned threshold ===')
    summarize('default threshold (0.5)', all_default)
    summarize('nested-tuned threshold', all_tuned)
