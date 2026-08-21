"""
Step 14: MC Dropout calibration CHECK on the TRUE held-out test set.

The original MC Dropout "confident subset" result (MCC 0.8909 on 26/50
samples) in FinalModel_modified.ipynb was computed on a stratified subset of
the TRAINING data (in-sample) -- explicitly flagged in
docs\HANDOVER_PROJECT_CONTEXT.md SS1/SS4/SS6 as NOT a valid generalization
estimate. This script reruns the exact same MC Dropout technique and
confident/flagged rule, but on the genuine 92-sample held-out test set,
against the actual locked-in production model (step11_final_lockin_and_test_
eval.py's pipeline): frozen pan-cancer-leaky-relu backbone -> filtered probe
set -> PCA(50, fit on full 368-sample training set) -> Dense(16, L2=0.01) ->
Dropout(0.3) -> Dense(2, softmax).

MC Dropout technique (reused verbatim from FinalModel_modified.ipynb, the
only change is WHICH samples/model it's applied to):
  - 50 forward passes with Dropout ACTIVE (training=True)
  - per-sample uncertainty = std across passes of the probability assigned
    to that sample's predicted class
  - uncertainty_threshold = round(mean(sample_uncertainty), 2)
  - confident_mask = sample_uncertainty < uncertainty_threshold

This is a calibration CHECK, not a tuning exercise -- the threshold rule is
reused exactly as documented, not adjusted to improve the split.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
from pathlib import Path
import tf_keras as keras
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score, precision_score, accuracy_score, confusion_matrix

from step4_cnn_bottleneck_model import build_head
from config import THYROID_PATH, MODEL_PATH

LOCKED_CFG = dict(pca_dim=50, hidden_dim=16, l2=0.01, dropout=0.3, lr=1e-3)
N_MC_PASSES = 50

data = np.load(Path(THYROID_PATH, 'FilteredSubtypeFinalData.npy'))
X_train_raw = np.nan_to_num(data['X_subtype_train']).astype(np.float32)
y_train = data['y_subtype_train'].astype(int)
X_test_raw = np.nan_to_num(data['X_subtype_test']).astype(np.float32)
y_test = data['y_subtype_test'].astype(int)
print(f'Train: {X_train_raw.shape}, cvPTC={int((y_train==0).sum())}, fvPTC={int((y_train==1).sum())}')
print(f'Test:  {X_test_raw.shape}, cvPTC={int((y_test==0).sum())}, fvPTC={int((y_test==1).sum())}')
assert X_test_raw.shape[0] == 92, f'expected 92 held-out test samples, got {X_test_raw.shape[0]}'

# Reproduce step11 exactly: embeddings, PCA(50) fit on full 368 train, head
# trained on full train set with a 15% stratified inner val carve-out for
# early stopping only (no hyperparameter tuning here -- config is locked).
train_emb_path = Path(THYROID_PATH, 'filtered_subtype_train_embeddings.npy')
test_emb_path = Path(THYROID_PATH, 'filtered_subtype_test_embeddings.npy')

print('Loading frozen pan-cancer base model...')
load_path = os.path.join(MODEL_PATH, 'pan-cancer-leaky-relu')
base_model = keras.models.load_model(load_path)
embedder = keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

train_embeddings = np.load(train_emb_path).astype(np.float32)
if test_emb_path.exists():
    test_embeddings = np.load(test_emb_path).astype(np.float32)
else:
    print(f'Extracting test embeddings for {X_test_raw.shape[0]} samples...')
    test_embeddings = embedder.predict(X_test_raw[..., np.newaxis], batch_size=4, verbose=1)
    np.save(test_emb_path, test_embeddings)

inner_train_idx, inner_val_idx = train_test_split(
    np.arange(len(y_train)), test_size=0.15, random_state=2569, stratify=y_train)

pca = PCA(n_components=LOCKED_CFG['pca_dim'], random_state=42)
X_fit_pca_train = pca.fit_transform(train_embeddings)
X_inner_train = X_fit_pca_train[inner_train_idx]
X_inner_val = X_fit_pca_train[inner_val_idx]
y_inner_train = y_train[inner_train_idx]
y_inner_val = y_train[inner_val_idx]

cw_vals = class_weight.compute_class_weight('balanced', classes=np.unique(y_inner_train), y=y_inner_train)
cw_dict = dict(enumerate(cw_vals))

keras.backend.clear_session()
keras.utils.set_random_seed(42)
model = build_head(LOCKED_CFG['pca_dim'], LOCKED_CFG['hidden_dim'], LOCKED_CFG['l2'], LOCKED_CFG['dropout'])
model.compile(loss='sparse_categorical_crossentropy',
               optimizer=keras.optimizers.Adam(learning_rate=LOCKED_CFG['lr']),
               metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=0)
model.fit(X_inner_train, y_inner_train, batch_size=8, epochs=150,
          validation_data=(X_inner_val, y_inner_val), callbacks=[es, rlr],
          class_weight=cw_dict, verbose=0)
print('Locked-in model reproduced (same recipe as step11).')

X_test_pca = pca.transform(test_embeddings).astype(np.float32)

# --- Sanity check: standard (non-MC) eval should reproduce 0.5831/0.8097/0.8478 ---
y_proba_std = model.predict(X_test_pca, verbose=0)[:, 1]
y_pred_std = (y_proba_std >= 0.5).astype(int)
mcc_std = matthews_corrcoef(y_test, y_pred_std)
auc_std = roc_auc_score(y_test, y_proba_std)
acc_std = accuracy_score(y_test, y_pred_std)
print(f'\n=== Sanity check (standard inference, dropout OFF) ===')
print(f'MCC={mcc_std:.4f} AUC={auc_std:.4f} Acc={acc_std:.4f}  (expect ~0.5831/0.8097/0.8478)')

# --- MC Dropout: 50 forward passes, dropout active, on the 92-sample test set ---
def mc_dropout_predict(model, X, n_iterations=N_MC_PASSES):
    preds = [model(X, training=True).numpy() for _ in range(n_iterations)]
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)

print(f'\nRunning MC Dropout ({N_MC_PASSES} passes) on the 92-sample TRUE held-out test set...')
mc_mean_probs, mc_uncertainty = mc_dropout_predict(model, X_test_pca)

mc_y_pred = np.argmax(mc_mean_probs, axis=1)
mc_y_proba = mc_mean_probs[:, 1]
sample_uncertainty = mc_uncertainty[np.arange(len(mc_y_pred)), mc_y_pred]

mcc_mc = matthews_corrcoef(y_test, mc_y_pred)
auc_mc = roc_auc_score(y_test, mc_y_proba)
acc_mc = accuracy_score(y_test, mc_y_pred)
recall_mc = recall_score(y_test, mc_y_pred, pos_label=1, zero_division=0)
prec_mc = precision_score(y_test, mc_y_pred, pos_label=1, zero_division=0)
cm_mc = confusion_matrix(y_test, mc_y_pred)

print(f'\n=== MC Dropout on FULL 92-sample held-out test set ===')
print(f'MCC={mcc_mc:.4f} AUC={auc_mc:.4f} Acc={acc_mc:.4f} fvPTC_recall={recall_mc:.4f} fvPTC_precision={prec_mc:.4f}')
print(f'Confusion matrix:\n{cm_mc}')
print(f'Mean uncertainty: {sample_uncertainty.mean():.4f}, Min: {sample_uncertainty.min():.4f}, Max: {sample_uncertainty.max():.4f}')

# Same rule as FinalModel_modified.ipynb: threshold = round(mean, 2), confident = below
uncertainty_threshold = round(float(sample_uncertainty.mean()), 2)
confident_mask = sample_uncertainty < uncertainty_threshold
flagged_mask = ~confident_mask

print(f'\nConfidence threshold = {uncertainty_threshold} (mean uncertainty, same rule as original)')
print(f'Confident: {confident_mask.sum()}/{len(mc_y_pred)}  Flagged for review: {flagged_mask.sum()}/{len(mc_y_pred)}')


def metrics_on_subset(mask, label):
    n = int(mask.sum())
    if n < 2 or len(np.unique(y_test[mask])) < 2:
        print(f'\n{label}: n={n} -- too small or single-class, MCC/AUC undefined')
        return dict(n=n, mcc=None, auc=None, acc=None, recall=None, precision=None, confusion_matrix=None)
    yt = y_test[mask]
    yp = mc_y_pred[mask]
    ypr = mc_y_proba[mask]
    mcc = matthews_corrcoef(yt, yp)
    try:
        auc = roc_auc_score(yt, ypr)
    except ValueError:
        auc = None
    acc = accuracy_score(yt, yp)
    recall = recall_score(yt, yp, pos_label=1, zero_division=0)
    prec = precision_score(yt, yp, pos_label=1, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    print(f'\n{label}: n={n}')
    print(f'  MCC={mcc:.4f} AUC={auc if auc is None else round(auc,4)} Acc={acc:.4f} '
          f'fvPTC_recall={recall:.4f} fvPTC_precision={prec:.4f}')
    print(f'  Confusion matrix:\n{cm}')
    return dict(n=n, mcc=mcc, auc=auc, acc=acc, recall=recall, precision=prec,
                confusion_matrix=cm.tolist())


conf_metrics = metrics_on_subset(confident_mask, 'CONFIDENT subset')
flag_metrics = metrics_on_subset(flagged_mask, 'FLAGGED (for review) subset')

print('\n=== Calibration verdict ===')
if conf_metrics['mcc'] is not None:
    delta = conf_metrics['mcc'] - mcc_mc
    if delta > 0.02:
        verdict = f'Confident subset MCC ({conf_metrics["mcc"]:.4f}) is meaningfully higher than full-set MCC ({mcc_mc:.4f}); confidence signal appears to transfer to held-out data.'
    elif delta < -0.02:
        verdict = f'Confident subset MCC ({conf_metrics["mcc"]:.4f}) is LOWER than full-set MCC ({mcc_mc:.4f}); confidence signal does NOT transfer to held-out data as the in-sample number suggested.'
    else:
        verdict = f'Confident subset MCC ({conf_metrics["mcc"]:.4f}) is roughly tied with full-set MCC ({mcc_mc:.4f}) -- no clear evidence the confidence signal is informative on held-out data.'
else:
    verdict = 'Confident subset too small/single-class to compute MCC -- cannot assess whether the confidence signal transfers.'
print(verdict)

# --- Save full result ---
out = dict(
    description='MC Dropout calibration check on the TRUE 92-sample held-out test set (not in-sample).',
    n_mc_passes=N_MC_PASSES,
    locked_config=LOCKED_CFG,
    sanity_check_standard_inference=dict(mcc=float(mcc_std), auc=float(auc_std), acc=float(acc_std)),
    full_set_mc_dropout=dict(mcc=float(mcc_mc), auc=float(auc_mc), acc=float(acc_mc),
                              recall=float(recall_mc), precision=float(prec_mc),
                              confusion_matrix=cm_mc.tolist()),
    uncertainty_threshold=uncertainty_threshold,
    n_confident=int(confident_mask.sum()),
    n_flagged=int(flagged_mask.sum()),
    frac_flagged=float(flagged_mask.mean()),
    confident_subset_metrics=conf_metrics,
    flagged_subset_metrics=flag_metrics,
    verdict=verdict,
    per_sample=[
        dict(idx=int(i), true_label=int(y_test[i]), pred_label=int(mc_y_pred[i]),
             pred_proba_fvptc=float(mc_y_proba[i]), uncertainty=float(sample_uncertainty[i]),
             confident=bool(confident_mask[i]))
        for i in range(len(y_test))
    ],
)

out_path = Path('code', 'thyroid', 'mc_dropout_true_holdout_result.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved full result to {out_path.resolve()}')
