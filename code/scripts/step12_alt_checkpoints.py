"""
Follow-up to the fine-tuning plan: test the two other pan-cancer pretrained
checkpoints on disk (pan-cancer-standard-relu, pan-cancer-solid-only) --
everything so far only ever used pan-cancer-leaky-relu. Architecturally
identical (58 layers, same input shape, same penultimate embedding shape),
so this is a drop-in swap: extract frozen embeddings from each checkpoint
for the FILTERED probe set's 368-sample training data (the winning probe
set from Stage 3), then run the same locked PCA(50)+Dense(16) head under
the seeded 25-run harness, exactly comparable to the current best result.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
import tf_keras as keras

from step5_seed_stability_harness import seeded_cv, summarize, SEEDS
from config import THYROID_PATH, MODEL_PATH

LOCKED_CFG = dict(pca_dim=50, hidden_dim=16, l2=0.01, dropout=0.3, lr=1e-3)

data = np.load(Path(THYROID_PATH, 'FilteredSubtypeFinalData.npy'))
X_raw = np.nan_to_num(data['X_subtype_train']).astype(np.float32)
y = data['y_subtype_train'].astype(int)
X_input = X_raw[..., np.newaxis]

CHECKPOINTS = ['pan-cancer-leaky-relu', 'pan-cancer-standard-relu', 'pan-cancer-solid-only']

if __name__ == '__main__':
    results = {}
    for ckpt in CHECKPOINTS:
        if ckpt == 'pan-cancer-leaky-relu':
            # Already extracted in step8b_extract_alt_probeset_embeddings.py (this is
            # the checkpoint every other experiment in this fine-tuning pass used).
            emb_path = Path(THYROID_PATH, 'filtered_subtype_train_embeddings.npy')
        else:
            emb_path = Path(THYROID_PATH, f'filtered_subtype_train_embeddings__{ckpt}.npy')
        if emb_path.exists():
            embeddings = np.load(emb_path).astype(np.float32)
            print(f'{ckpt}: loaded cached embeddings {embeddings.shape}')
        else:
            print(f'{ckpt}: loading checkpoint and extracting embeddings...')
            base_model = keras.models.load_model(os.path.join(MODEL_PATH, ckpt))
            embedder = keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
            embeddings = embedder.predict(X_input, batch_size=4, verbose=1)
            np.save(emb_path, embeddings)
            print(f'{ckpt}: saved embeddings {embeddings.shape}')
            keras.backend.clear_session()

        print(f'\n=== CV: {ckpt} (filtered probes, locked PCA(50)+Dense(16) head) ===')
        mcc, auc, recall1 = seeded_cv(LOCKED_CFG, seeds=SEEDS, embeddings=embeddings, y=y)
        results[ckpt] = summarize(ckpt, mcc, auc, recall1)

    print('\n=== Checkpoint comparison summary (best MCC first) ===')
    for name, s in sorted(results.items(), key=lambda kv: -kv[1]['mcc_mean']):
        print(f"{name}: MCC={s['mcc_mean']:.4f}+/-{s['mcc_std']:.4f} "
              f"AUC={s['auc_mean']:.4f}+/-{s['auc_std']:.4f} "
              f"recall={s['recall_mean']:.4f}+/-{s['recall_std']:.4f} "
              f"collapsed={s['n_collapsed']}/{s['n_runs']}")
