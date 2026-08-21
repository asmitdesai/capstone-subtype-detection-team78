"""
Stage 3 continued: extract frozen pan-cancer backbone embeddings for the two
alternate probe sets built by step8_alternate_probesets.py, mirroring
step2_embedding_diagnostic.py's extraction exactly (same frozen base model,
same penultimate-layer embedder), so step4_cnn_bottleneck_model.py's
run_cv() can be pointed at them with no code changes.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
import tf_keras as keras

from config import THYROID_PATH, MODEL_PATH

print('Loading frozen pan-cancer base model...')
load_path = os.path.join(MODEL_PATH, 'pan-cancer-leaky-relu')
base_model = keras.models.load_model(load_path)
embedder = keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

for name, filename in [('Filtered', 'FilteredSubtypeFinalData.npy'),
                        ('Differential', 'DifferentialSubtypeFinalData.npy')]:
    data = np.load(Path(THYROID_PATH, filename))
    X_raw = np.nan_to_num(data['X_subtype_train']).astype(np.float32)
    print(f'{name}: extracting embeddings for {X_raw.shape[0]} samples...')
    X_input = X_raw[..., np.newaxis]
    embeddings = embedder.predict(X_input, batch_size=4, verbose=1)
    out_path = Path(THYROID_PATH, f'{name.lower()}_subtype_train_embeddings.npy')
    np.save(out_path, embeddings)
    print(f'{name}: saved embeddings {embeddings.shape} to {out_path}')
