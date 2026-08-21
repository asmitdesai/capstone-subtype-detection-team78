"""
Stage 4 (optional/last-resort, only attempted because Stages 1-3 plateaued
around CV MCC 0.50-0.53 with no clear win): partial fine-tuning of the
frozen pan-cancer 1D-CNN backbone's LAST residual block only, not a full
unfreeze -- a full-unfreeze attempt already collapsed in this codebase
(FinalModel_modified.ipynb cell 13's own comment: "causes the model to
collapse to predicting one class for everything").

Architectural note: PCA (used in Stages 1-3) is a fitted, non-differentiable
transform, so it cannot sit inside an end-to-end-trainable Keras graph once
conv layers are unfrozen. Reusing it here would force a return to a huge
*trainable* bottleneck straight from the 1,942,272-dim Flatten output --
exactly the failure mode already fixed once (see step4's docstring: a
~97M-param layer trained on ~234 samples). Instead, this script replaces the
backbone's Flatten with GlobalAveragePooling1D over the last conv block's
output (1,942,272 -> 64 features) -- a parameter-free architectural
reduction, not a fitted one, so it works safely inside a trainable graph and
avoids the parameter explosion entirely.

Unfreezing scope: only the four Conv1D layers in the last residual block
(conv1d_20..conv1d_23, base_model layer indices 38-54, confirmed via
model.summary()). BatchNormalization layers in that block are kept frozen
(both weights and running statistics) -- unfreezing BN running stats with a
small batch size on ~234 training samples is a known source of training
instability, and the plan explicitly calls this out as a precaution.

LR is 1e-5 (two orders below the PCA+head's 1e-3), matching the plan's
guidance for this specific higher-risk stage.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
import numpy as np
from pathlib import Path
import tf_keras as keras
from tf_keras import regularizers
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score

from config import THYROID_PATH, MODEL_PATH

data = np.load(Path(THYROID_PATH, 'FinalData.npy'))
X_raw = np.nan_to_num(data['X_subtype_train']).astype(np.float32)
y = data['y_subtype_train'].astype(int)

LAST_BLOCK_START = 'conv1d_20'
LAST_BLOCK_END = 'add_5'
UNFREEZE_CONV_LAYERS = {'conv1d_20', 'conv1d_21', 'conv1d_22', 'conv1d_23'}


def build_model(hidden_dim=32, l2=0.01, dropout=0.4):
    load_path = os.path.join(MODEL_PATH, 'pan-cancer-leaky-relu')
    base_model = keras.models.load_model(load_path)

    unfreezing = False
    for layer in base_model.layers:
        if layer.name == LAST_BLOCK_START:
            unfreezing = True
        if unfreezing and layer.name in UNFREEZE_CONV_LAYERS:
            layer.trainable = True
        else:
            layer.trainable = False
        if layer.name == LAST_BLOCK_END:
            break
    # Everything after add_5 (average_pooling1d_3, flatten_1, output) is
    # unused -- we branch a new head off add_5 directly below instead.

    block_output = base_model.get_layer(LAST_BLOCK_END).output
    x = keras.layers.GlobalAveragePooling1D()(block_output)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(hidden_dim, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = keras.layers.Dropout(dropout)(x)
    output = keras.layers.Dense(2, activation='softmax', name='subtype_output')(x)
    model = keras.Model(inputs=base_model.input, outputs=output)
    return model


def is_collapsed(y_val, y_pred):
    return len(np.unique(y_pred)) == 1


def run_fold(X_train, y_train, X_val, y_val, seed, hidden_dim=32, l2=0.01, dropout=0.4,
             lr=1e-5, batch_size=8, epochs=25, patience=6, verbose=0):
    cw_vals = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw_vals))

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = build_model(hidden_dim, l2, dropout)
    model.compile(loss='sparse_categorical_crossentropy',
                   optimizer=keras.optimizers.Adam(learning_rate=lr),
                   metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=0)

    t0 = time.time()
    history = model.fit(X_train[..., np.newaxis], y_train, batch_size=batch_size, epochs=epochs,
                         validation_data=(X_val[..., np.newaxis], y_val),
                         callbacks=[es, rlr], class_weight=cw_dict, verbose=verbose)
    elapsed = time.time() - t0
    n_epochs_run = len(history.history['loss'])

    y_proba = model.predict(X_val[..., np.newaxis], verbose=0)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    recall1 = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    collapsed = is_collapsed(y_val, y_pred)

    print(f'    [{elapsed:.1f}s, {n_epochs_run} epochs] MCC={mcc:.4f} AUC={auc:.4f} '
          f'recall={recall1:.4f} collapsed={collapsed}')
    return mcc, auc, recall1, collapsed, elapsed


if __name__ == '__main__':
    smoke_test = '--smoke' in sys.argv

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X_raw, y))

    if smoke_test:
        print('=== SMOKE TEST: 1 fold, 1 seed, capped epochs -- feasibility/timing/collapse check ===')
        train_idx, val_idx = splits[0]
        mcc, auc, recall1, collapsed, elapsed = run_fold(
            X_raw[train_idx], y[train_idx], X_raw[val_idx], y[val_idx],
            seed=0, epochs=25, patience=6, verbose=2)
        print(f'\nSmoke test result: MCC={mcc:.4f} AUC={auc:.4f} recall={recall1:.4f} '
              f'collapsed={collapsed} time={elapsed:.1f}s')
        sys.exit(0)

    SEEDS = [0, 1, 2]  # fewer seeds than Stages 1-3 given the higher per-run cost
    all_results = []
    aborted = False
    for seed in SEEDS:
        print(f'-- seed {seed} --')
        for fold, (train_idx, val_idx) in enumerate(splits):
            mcc, auc, recall1, collapsed, elapsed = run_fold(
                X_raw[train_idx], y[train_idx], X_raw[val_idx], y[val_idx], seed=seed * 100 + fold)
            all_results.append((mcc, auc, recall1, collapsed))
            if collapsed:
                print(f'    !! COLLAPSE detected at seed={seed} fold={fold+1} -- aborting remaining runs')
                aborted = True
                break
        if aborted:
            break

    mccs = np.array([r[0] for r in all_results])
    aucs = np.array([r[1] for r in all_results])
    recalls = np.array([r[2] for r in all_results])
    n_collapsed = sum(r[3] for r in all_results)
    print(f'\n=== Stage 4 summary ({len(all_results)} runs, aborted={aborted}) ===')
    print(f'MCC: {mccs.mean():.4f} +/- {mccs.std():.4f}')
    print(f'AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}')
    print(f'Recall: {recalls.mean():.4f} +/- {recalls.std():.4f}')
    print(f'Collapsed runs: {n_collapsed}/{len(all_results)}')
    print('Compare to Prerequisite-A baseline: MCC=0.5039+/-0.0864, AUC=0.8275+/-0.0483, recall=0.6801+/-0.0993')
