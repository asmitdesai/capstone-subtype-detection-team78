"""
Stage 5/6: lock in the winning config from the fine-tuning plan and run the
ONE final held-out test evaluation.

Winner (Stage 5 decision, documented in the fine-tuning plan/handover):
filtered probe set (394,337 real probes, zero-padded to 485,577) fed through
the frozen pan-cancer 1D-CNN backbone, PCA(50) (fit on the full 368-sample
training set this time, not per-CV-fold), small Dense head
(Dense(16, L2=0.01) -> Dropout(0.3) -> Dense(2, softmax), Adam lr=1e-3,
class_weight='balanced') -- CV MCC 0.5289+/-0.1117, AUC 0.8471+/-0.0434,
fvPTC recall 0.7299+/-0.1198, statistically tied with the best MCC found
(SVC, 0.5300) but with substantially better recall and, unlike SVC, a fully
neural classifier head on top of the CNN backbone.

This script:
1. Extracts frozen-backbone embeddings for the filtered probe set's TEST
   split (only the train split was extracted before, for CV).
2. Fits PCA(50) on the FULL 368-sample filtered training embeddings (no CV
   split now -- CV's job was config selection, already done).
3. Trains the locked head on all 368 training samples, with a small internal
   validation carve-out used ONLY for early-stopping (not for any further
   hyperparameter tuning -- those are already fixed).
4. Evaluates EXACTLY ONCE on the corrected 92-sample held-out test set.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

data = np.load(Path(THYROID_PATH, 'FilteredSubtypeFinalData.npy'))
X_train_raw = np.nan_to_num(data['X_subtype_train']).astype(np.float32)
y_train = data['y_subtype_train'].astype(int)
X_test_raw = np.nan_to_num(data['X_subtype_test']).astype(np.float32)
y_test = data['y_subtype_test'].astype(int)
print(f'Train: {X_train_raw.shape}, cvPTC={int((y_train==0).sum())}, fvPTC={int((y_train==1).sum())}')
print(f'Test:  {X_test_raw.shape}, cvPTC={int((y_test==0).sum())}, fvPTC={int((y_test==1).sum())}')

# Step 1: train embeddings already exist; extract test embeddings now.
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
    print(f'Saved {test_emb_path}')

# Step 2/3: fit PCA + train head on the FULL 368-sample training set, with a
# small internal validation carve-out (stratified) used only for early
# stopping -- hyperparameters are already locked, not re-tuned here.
inner_train_idx, inner_val_idx = train_test_split(
    np.arange(len(y_train)), test_size=0.15, random_state=2569, stratify=y_train)

pca = PCA(n_components=LOCKED_CFG['pca_dim'], random_state=42)
X_fit_pca_train = pca.fit_transform(train_embeddings)  # fit PCA on ALL 368 (this is the final production fit)
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
print('Final model trained on full 368-sample training set (locked config).')

# Step 4: the ONE final held-out evaluation.
X_test_pca = pca.transform(test_embeddings)
y_proba_test = model.predict(X_test_pca, verbose=0)[:, 1]
y_pred_test = (y_proba_test >= 0.5).astype(int)

mcc = matthews_corrcoef(y_test, y_pred_test)
auc = roc_auc_score(y_test, y_proba_test)
acc = accuracy_score(y_test, y_pred_test)
recall1 = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
precision1 = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
cm = confusion_matrix(y_test, y_pred_test)

print('\n=== FINAL HELD-OUT TEST EVALUATION (92 samples, evaluated once) ===')
print(f'MCC:            {mcc:.4f}')
print(f'AUC:             {auc:.4f}')
print(f'Accuracy:        {acc:.4f}')
print(f'fvPTC Recall:    {recall1:.4f}')
print(f'fvPTC Precision: {precision1:.4f}')
print(f'Confusion matrix (rows=true, cols=pred, [cvPTC, fvPTC]):\n{cm}')
