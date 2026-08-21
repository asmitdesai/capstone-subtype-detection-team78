"""
Step 2 of remediation plan: extract frozen pan-cancer base-model embeddings
(base_model.layers[-2].output, the penultimate/pre-Dense layer used as the
head's input in FinalModel_modified.ipynb's get_optimized_subtype_model) for
all 368 training samples, then run the same classical models from Step 1 on
those embeddings instead of raw beta-values, same CV scheme, to check whether
the frozen pan-cancer features carry a learnable cvPTC/fvPTC signal.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(os.path.abspath('.'))
import numpy as np
from pathlib import Path
import tf_keras as keras
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score

from config import THYROID_PATH, MODEL_PATH

print('Loading pan-cancer base model...')
load_path = os.path.join(MODEL_PATH, 'pan-cancer-leaky-relu')
base_model = keras.models.load_model(load_path)
embedder = keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
print(f'Embedding dim: {embedder.output_shape}')

data = np.load(Path(THYROID_PATH, 'FinalData.npy'))
X_raw = np.nan_to_num(data['X_subtype_train']).astype(np.float32)
y = data['y_subtype_train'].astype(int)

print(f'Extracting embeddings for {X_raw.shape[0]} samples (this may take a while on CPU)...')
X_input = X_raw[..., np.newaxis]  # (N, 485577, 1) as expected by Conv1D input
embeddings = embedder.predict(X_input, batch_size=4, verbose=1)
print(f'Embeddings shape: {embeddings.shape}')

np.save(Path(THYROID_PATH, 'subtype_train_embeddings.npy'), embeddings)
print('Saved embeddings to subtype_train_embeddings.npy')

# Embedding dim (1,942,272) is itself too high-dim for QDA/SVC at n=294/fold —
# apply PCA (fit per-fold, not on full data, to avoid leakage) to a modest
# number of components so the comparison to Step 1 raw-beta-value classical
# baselines is fair (there, features were already lower-dimensional than the
# embedding here).
N_COMPONENTS = 50

models = {
    'QDA': lambda: QuadraticDiscriminantAnalysis(),
    'SVC': lambda: SVC(C=0.2, class_weight='balanced', probability=True),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f'\n=== Classical models on frozen pan-cancer embeddings (PCA -> {N_COMPONENTS} comps, fit per fold) ===')
for name, make_model in models.items():
    print(f'\n--- {name} ---')
    fold_mcc, fold_auc, fold_recall1 = [], [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, y)):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        X_train_p = pca.fit_transform(X_train)
        X_val_p = pca.transform(X_val)

        model = make_model()
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_val_p)
        y_proba = model.predict_proba(X_val_p)[:, 1]

        mcc = matthews_corrcoef(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_proba)
        except ValueError:
            auc = float('nan')
        recall1 = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

        fold_mcc.append(mcc)
        fold_auc.append(auc)
        fold_recall1.append(recall1)
        print(f'  Fold {fold+1}: MCC={mcc:.4f} AUC={auc:.4f} fvPTC_recall={recall1:.4f}')

    print(f'  Aggregate: MCC={np.mean(fold_mcc):.4f} AUC={np.nanmean(fold_auc):.4f} '
          f'fvPTC_recall={np.mean(fold_recall1):.4f}')
