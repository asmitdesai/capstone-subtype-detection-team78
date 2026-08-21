"""
Step 4: fix the methylation branch while KEEPING it a 1D CNN + neural head.

Diagnosis recap (step1/step2/step3): the frozen pan-cancer 1D-CNN's conv
features carry a strong, transferable cvPTC/fvPTC signal (proven via SVC on
PCA(50)-reduced embeddings -> CV MCC 0.536). The original CV cell in
FinalModel_modified.ipynb collapsed because its head (Dropout->Dense(64)->
Dropout->Dense(softmax)) was trained directly on the raw 1,942,272-dim
flattened embedding with only ~234 samples/fold.

First attempt at this fix added a *trainable* Dense bottleneck straight from
the 1,942,272-dim embedding down to ~50 units -- that is itself a ~97M
parameter layer trained from near-random init on 234 samples, which
reproduces the exact same failure mode this is supposed to fix (and is far
too slow/memory-heavy on CPU; killed after >100 min with no result).

Corrected approach: apply PCA as a FIXED (per-fold-fit, not trainable, no
leakage) dimensionality reduction on the frozen embeddings -- exactly the
step that made SVC work in step2/step3 -- and only train a small Dense
neural network classifier on top of the resulting 50-dim features. This
keeps the full pipeline a "1D CNN": Conv1D layers extract features (frozen,
pretrained), PCA is a fixed linear projection of those conv features (not a
new trainable layer), and a small trainable dense network classifies. The
learnable part is now sized appropriately for ~234 samples/fold, unlike the
original notebook's head or the failed first attempt at this fix.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
import tf_keras as keras
from tf_keras import regularizers
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score

from config import THYROID_PATH

data = np.load(Path(THYROID_PATH, 'FinalData.npy'))
y = data['y_subtype_train'].astype(int)

emb_path = Path(THYROID_PATH, 'subtype_train_embeddings.npy')
embeddings = np.load(emb_path).astype(np.float32)
print(f'Embeddings: {embeddings.shape}, cvPTC(0)={int((y==0).sum())}, fvPTC(1)={int((y==1).sum())}')


def build_head(input_dim, hidden_dim, l2, dropout):
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(hidden_dim, activation='relu',
                            kernel_regularizer=regularizers.l2(l2))(inp)
    x = keras.layers.Dropout(dropout)(x)
    output = keras.layers.Dense(2, activation='softmax', name='subtype_output')(x)
    return keras.Model(inputs=inp, outputs=output)


def run_cv(pca_dim, hidden_dim, l2, dropout, lr, batch_size=8, epochs=150, patience=15,
           verbose=0, seed=None, embeddings=None, y=None, return_folds=False):
    """Runs the 5-fold CV protocol for one config. If `seed` is given, calls
    keras.utils.set_random_seed(seed) before each fold's model is built, so
    weight init/dropout are reproducible (StratifiedKFold's split itself is
    always fixed at random_state=42 regardless of `seed`, so only the
    model-training randomness varies across seeds, not fold membership).
    Pass `embeddings`/`y` to run against a different feature set (e.g. an
    alternate probe set's embeddings); defaults to this module's globals.
    Returns per-fold MCC/AUC/recall lists if return_folds else their means.
    """
    emb = embeddings if embeddings is not None else globals()['embeddings']
    labels = y if y is not None else globals()['y']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_mcc, fold_auc, fold_recall1 = [], [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(emb, labels)):
        X_train_emb, X_val_emb = emb[train_idx], emb[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        pca = PCA(n_components=pca_dim, random_state=42)
        X_train = pca.fit_transform(X_train_emb).astype(np.float32)
        X_val = pca.transform(X_val_emb).astype(np.float32)

        cw_vals = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw_dict = dict(enumerate(cw_vals))

        keras.backend.clear_session()
        if seed is not None:
            keras.utils.set_random_seed(seed * 100 + fold)
        model = build_head(pca_dim, hidden_dim, l2, dropout)
        model.compile(loss='sparse_categorical_crossentropy',
                       optimizer=keras.optimizers.Adam(learning_rate=lr),
                       metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=0)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(X_val, y_val), callbacks=[es, rlr],
                  class_weight=cw_dict, verbose=verbose)

        y_proba = model.predict(X_val, verbose=0)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        mcc = matthews_corrcoef(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        recall1 = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        fold_mcc.append(mcc)
        fold_auc.append(auc)
        fold_recall1.append(recall1)
        print(f'    Fold {fold+1}: MCC={mcc:.4f} AUC={auc:.4f} fvPTC_recall={recall1:.4f}')

    if return_folds:
        return fold_mcc, fold_auc, fold_recall1
    return np.mean(fold_mcc), np.mean(fold_auc), np.mean(fold_recall1)


configs = [
    dict(pca_dim=50, hidden_dim=16, l2=0.01, dropout=0.3, lr=1e-3),
    dict(pca_dim=50, hidden_dim=32, l2=0.01, dropout=0.4, lr=1e-3),
    dict(pca_dim=30, hidden_dim=16, l2=0.01, dropout=0.3, lr=1e-3),
    dict(pca_dim=50, hidden_dim=16, l2=0.05, dropout=0.5, lr=5e-4),
    dict(pca_dim=20, hidden_dim=8, l2=0.01, dropout=0.3, lr=1e-3),
]

if __name__ == '__main__':
    results = []
    for cfg in configs:
        print(f'\n=== Config: {cfg} ===')
        mcc, auc, recall1 = run_cv(**cfg)
        print(f'  Aggregate: MCC={mcc:.4f} AUC={auc:.4f} fvPTC_recall={recall1:.4f}')
        results.append((cfg, mcc, auc, recall1))

    print('\n=== Sweep summary (best first) ===')
    for cfg, mcc, auc, recall1 in sorted(results, key=lambda r: -r[1]):
        print(f'MCC={mcc:.4f} AUC={auc:.4f} recall={recall1:.4f}  cfg={cfg}')
