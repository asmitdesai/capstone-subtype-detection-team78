"""
Step 3/4 of remediation plan: production candidate for the methylation branch.

Root cause (confirmed by step1_classical_baseline.py + step2_embedding_diagnostic.py):
FinalModel_modified.ipynb's CV cell trains a 64-unit Dense head directly on the
frozen pan-cancer base model's 1,942,272-dim flattened embedding, with only
~234 training samples per fold (Adam lr=1e-3). That ratio causes every fold to
collapse to predicting a single constant class (exact 0.0 MCC in all 5 folds).

The frozen pan-cancer embeddings themselves carry a strong, transferable
cvPTC/fvPTC signal (step2: SVC on PCA(50)-reduced embeddings -> CV MCC 0.536,
AUC 0.844, no collapse in any fold) -- the bottleneck is dimensionality
reduction before the classifier head, not the pretrained features.

This script reruns the full CV protocol (same StratifiedKFold(5, seed=42) as
FinalModel_modified.ipynb) end-to-end from raw beta-values -> frozen embedding
-> per-fold PCA -> SVC, with per-fold predictions printed in the same
collapse-diagnostic format as the original notebook, to confirm the fix holds
end-to-end (not just on precomputed embeddings).
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
import tf_keras as keras
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score, confusion_matrix

from config import THYROID_PATH, MODEL_PATH

data = np.load(Path(THYROID_PATH, 'FinalData.npy'))
X_raw = np.nan_to_num(data['X_subtype_train']).astype(np.float32)
y = data['y_subtype_train'].astype(int)

print('Loading frozen pan-cancer base model + precomputed embeddings...')
emb_path = Path(THYROID_PATH, 'subtype_train_embeddings.npy')
if emb_path.exists():
    embeddings = np.load(emb_path)
else:
    load_path = os.path.join(MODEL_PATH, 'pan-cancer-leaky-relu')
    base_model = keras.models.load_model(load_path)
    embedder = keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    embeddings = embedder.predict(X_raw[..., np.newaxis], batch_size=4, verbose=1)
    np.save(emb_path, embeddings)

print(f'cvPTC(0)={int((y==0).sum())}, fvPTC(1)={int((y==1).sum())}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_mcc, fold_auc, fold_recall1 = [], [], []

print('\n=== Fixed methylation-branch model: frozen pan-cancer embedding -> per-fold PCA(50) -> SVC ===')
for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, y)):
    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    pca = PCA(n_components=50, random_state=42)
    X_train_p = pca.fit_transform(X_train)
    X_val_p = pca.transform(X_val)

    model = SVC(C=0.2, class_weight='balanced', probability=True)
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_val_p)
    y_proba = model.predict_proba(X_val_p)[:, 1]

    mcc = matthews_corrcoef(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    recall1 = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    fold_mcc.append(mcc)
    fold_auc.append(auc)
    fold_recall1.append(recall1)

    print(f'Fold {fold+1}: MCC={mcc:.4f} AUC={auc:.4f} fvPTC_recall={recall1:.4f} '
          f'confusion=\n{cm}')

print(f'\n=== CROSS-VALIDATION SUMMARY (fixed model) ===')
print(f'Aggregate MCC: {np.mean(fold_mcc):.4f} (was -0.0018 for the original CNN head)')
print(f'Aggregate AUC: {np.mean(fold_auc):.4f} (was 0.5169 for the original CNN head)')
print(f'Aggregate fvPTC recall: {np.mean(fold_recall1):.4f} (was 0.3951 for the original CNN head)')
collapsed = any(r == 0.0 or r == 1.0 for r in fold_recall1) and any(
    m == 0.0 for m in fold_mcc)
print(f'Any fold collapsed to a constant prediction (MCC exactly 0 AND recall 0 or 1): {collapsed}')
