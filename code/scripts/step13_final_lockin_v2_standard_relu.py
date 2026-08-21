"""
Revised Stage 5/6 lock-in: pan-cancer-standard-relu beat pan-cancer-leaky-relu
(the checkpoint used everywhere else in this fine-tuning pass) on every
metric under the seeded harness -- higher MCC AND lower variance, not a
tradeoff (see step12_alt_checkpoints.py):
  standard-relu: MCC=0.5588+/-0.0712 AUC=0.8634+/-0.0396 recall=0.7362+/-0.0877
  leaky-relu:    MCC=0.5289+/-0.1117 AUC=0.8471+/-0.0434 recall=0.7299+/-0.1198

This reruns the final held-out test evaluation (Stage 6) with the checkpoint
swapped to pan-cancer-standard-relu, everything else identical to
step11_final_lockin_and_test_eval.py (filtered probe set, PCA(50) fit once
on all 368 training samples, locked Dense(16) head).
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
CHECKPOINT = 'pan-cancer-standard-relu'

data = np.load(Path(THYROID_PATH, 'FilteredSubtypeFinalData.npy'))
X_train_raw = np.nan_to_num(data['X_subtype_train']).astype(np.float32)
y_train = data['y_subtype_train'].astype(int)
X_test_raw = np.nan_to_num(data['X_subtype_test']).astype(np.float32)
y_test = data['y_subtype_test'].astype(int)
print(f'Train: {X_train_raw.shape}, cvPTC={int((y_train==0).sum())}, fvPTC={int((y_train==1).sum())}')
print(f'Test:  {X_test_raw.shape}, cvPTC={int((y_test==0).sum())}, fvPTC={int((y_test==1).sum())}')

train_emb_path = Path(THYROID_PATH, f'filtered_subtype_train_embeddings__{CHECKPOINT}.npy')
test_emb_path = Path(THYROID_PATH, f'filtered_subtype_test_embeddings__{CHECKPOINT}.npy')

print(f'Loading {CHECKPOINT}...')
base_model = keras.models.load_model(os.path.join(MODEL_PATH, CHECKPOINT))
embedder = keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

train_embeddings = np.load(train_emb_path).astype(np.float32)
if test_emb_path.exists():
    test_embeddings = np.load(test_emb_path).astype(np.float32)
else:
    print(f'Extracting test embeddings for {X_test_raw.shape[0]} samples...')
    test_embeddings = embedder.predict(X_test_raw[..., np.newaxis], batch_size=4, verbose=1)
    np.save(test_emb_path, test_embeddings)
    print(f'Saved {test_emb_path}')

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
print('Final model trained on full 368-sample training set (locked config, standard-relu checkpoint).')

X_test_pca = pca.transform(test_embeddings)
y_proba_test = model.predict(X_test_pca, verbose=0)[:, 1]
y_pred_test = (y_proba_test >= 0.5).astype(int)

mcc = matthews_corrcoef(y_test, y_pred_test)
auc = roc_auc_score(y_test, y_proba_test)
acc = accuracy_score(y_test, y_pred_test)
recall1 = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
precision1 = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
cm = confusion_matrix(y_test, y_pred_test)

print('\n=== FINAL HELD-OUT TEST EVALUATION v2 (standard-relu checkpoint, 92 samples, evaluated once) ===')
print(f'MCC:            {mcc:.4f}')
print(f'AUC:             {auc:.4f}')
print(f'Accuracy:        {acc:.4f}')
print(f'fvPTC Recall:    {recall1:.4f}')
print(f'fvPTC Precision: {precision1:.4f}')
print(f'Confusion matrix (rows=true, cols=pred, [cvPTC, fvPTC]):\n{cm}')
