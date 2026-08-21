"""
Step 1 of remediation plan (docs/HANDOVER_PROJECT_CONTEXT.md investigation):
run QDA / SVC directly on FinalData.npy's full 485,577-probe X_subtype_train
(the exact input the CNN sees), using the same StratifiedKFold(n_splits=5) CV
scheme as the CNN's CV cell in FinalModel_modified.ipynb, to check whether a
learnable cvPTC/fvPTC signal exists in the full unfiltered probe set.
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score

from config import THYROID_PATH

data = np.load(Path(THYROID_PATH, 'FinalData.npy'))
X = np.nan_to_num(data['X_subtype_train']).astype(np.float32)
y = data['y_subtype_train'].astype(int)

print(f'X: {X.shape}, cvPTC(0)={int((y==0).sum())}, fvPTC(1)={int((y==1).sum())}')

models = {
    'QDA': lambda: QuadraticDiscriminantAnalysis(),
    'SVC': lambda: SVC(C=0.2, class_weight='balanced', probability=True),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, make_model in models.items():
    print(f'\n=== {name} ===')
    fold_mcc, fold_auc, fold_recall1 = [], [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        mcc = matthews_corrcoef(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_proba)
        except ValueError:
            auc = float('nan')
        recall1 = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

        fold_mcc.append(mcc)
        fold_auc.append(auc)
        fold_recall1.append(recall1)
        print(f'  Fold {fold+1}: MCC={mcc:.4f} AUC={auc:.4f} fvPTC_recall={recall1:.4f} '
              f'(train n={len(train_idx)}, val n={len(val_idx)})')

    print(f'  Aggregate: MCC={np.mean(fold_mcc):.4f} AUC={np.nanmean(fold_auc):.4f} '
          f'fvPTC_recall={np.mean(fold_recall1):.4f}')
