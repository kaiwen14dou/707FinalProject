"""
End-to-end pipeline:
1) Load raw MIMIC ECG memmap + metadata
2) Convert to (N, 4096, 12) CODE-format tensors (in memory)
3) Extract 6 arrhythmia labels from ICD-10
4) Load official CODE model (automatic-ecg-diagnosis)
5) Predict probabilities on ALL ECGs
6) Sweep thresholds (0.01–0.99) to maximize F1 per class

No intermediate files (X_all.npy, labels_all.csv) are saved.
Only final threshold/F1 results are saved.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import tensorflow as tf

# ------------------------------------------------------------
# 0. PATHS YOU MUST UPDATE
# ------------------------------------------------------------

MIMIC_ROOT = "/hpc/group/honglab/kkgroup/mimic_preproc_out"

DF_ECG_PKL   = f"{MIMIC_ROOT}/memmap/df_memmap.pkl"
MM_NPY       = f"{MIMIC_ROOT}/memmap/memmap.npy"
MM_META      = f"{MIMIC_ROOT}/memmap/memmap_meta.npz"
DF_DIAG_PKL  = f"{MIMIC_ROOT}/records_w_diag_icd10_folds.pkl"

# path to the cloned repo automatic-ecg-diagnosis
CODE_REPO = "/hpc/group/honglab/kkgroup/yuejun_workspace/automatic-ecg-diagnosis"

# path inside that repo to pretrained weights
MODEL_WEIGHTS = os.path.join(CODE_REPO, "model.hdf5")

# append repo to Python path
sys.path.append(CODE_REPO)
from model import get_model   # official architecture

# ------------------------------------------------------------
# 1. LOAD RAW TABLES
# ------------------------------------------------------------

print("Loading df_ecg and df_diag...")
df_ecg = pd.read_pickle(DF_ECG_PKL)
df_diag = pd.read_pickle(DF_DIAG_PKL)

keys = ['study_id','subject_id','ecg_time']
df = df_ecg.merge(
    df_diag[keys + ['gender','all_diag_all','fold']],
    on=keys,
    how='left'
)
print("Merged shape:", df.shape)

# ------------------------------------------------------------
# 2. ICD-10 → Six CODE arrhythmia labels
# ------------------------------------------------------------

MAP = {
    '1dAVb': ['I440'],
    'RBBB' : ['I451'],
    'LBBB' : ['I447'],
    'SB'   : ['R001'],
    'ST'   : ['R000'],
    'AF'   : ['I48'],
}

def codes_to_labels(codes):
    if not isinstance(codes, (list, tuple)):
        return {k:0 for k in MAP}
    codes = [str(c).replace('.','').upper() for c in codes]
    out = {}
    for lab, pref_list in MAP.items():
        out[lab] = int(
            any(any(c.startswith(p) for c in codes) for p in pref_list)
        )
    return out

print("Extracting arrhythmia labels...")
label_df = pd.DataFrame(list(df['all_diag_all'].apply(codes_to_labels)))
df = pd.concat([df, label_df], axis=1)

LABEL_COLS = list(MAP.keys())
y_all = df[LABEL_COLS].values.astype("float32")   # (N,6)
print("Positive rates:\n", df[LABEL_COLS].mean())

# ------------------------------------------------------------
# 3. LOAD RAW MEMMAP AND CONVERT TO (4096 × 12)
# ------------------------------------------------------------

print("Loading memmap meta...")
meta = np.load(MM_META, allow_pickle=True)

starts = np.asarray(meta['start']).astype(np.int64)
lens   = np.asarray(meta['length']).astype(np.int64)

dtype = np.dtype(meta.get('dtype', np.float32))

Xflat = np.memmap(MM_NPY, mode="r", dtype=dtype).reshape(-1, 12)
print("Flat memmap:", Xflat.shape)

assert len(df) == len(starts) == len(lens)

TARGET_T = 4096
CH = 12

def pad_or_crop(x):
    t = x.shape[0]
    if t >= TARGET_T:
        return x[:TARGET_T]
    out = np.zeros((TARGET_T, CH), dtype=np.float32)
    out[:t] = x
    return out

N = len(df)
print(f"Preparing in-memory ECG tensor for {N} records...")

X_all = np.zeros((N, TARGET_T, CH), dtype=np.float32)

for i in range(N):
    st = int(starts[i])
    ln = int(lens[i])
    raw = np.asarray(Xflat[st:st+ln])
    X_all[i] = pad_or_crop(raw)
    if i % 5000 == 0:
        print(f"Processed {i}/{N}")

print("Final X_all shape:", X_all.shape)

# ------------------------------------------------------------
# 4. BUILD OFFICIAL CODE MODEL + LOAD WEIGHTS
# ------------------------------------------------------------

n_classes = len(LABEL_COLS)

print("Building official CODE-CNN model...")
model = get_model(n_classes=n_classes, last_layer="sigmoid")

print("Loading pretrained weights:", MODEL_WEIGHTS)
model.load_weights(MODEL_WEIGHTS)

model.summary()

# ------------------------------------------------------------
# 5. PREDICT PROBABILITIES ON ALL ECGs
# ------------------------------------------------------------

class AllECGSequence(tf.keras.utils.Sequence):
    def __init__(self, X, batch_size=256):
        self.X = X
        self.indices = np.arange(len(X), dtype=np.int64)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        # CODE model does NOT require per-sample standardization
        return np.asarray(self.X[batch_idx], dtype=np.float32)

all_seq = AllECGSequence(X_all, batch_size=256)

print("Predicting probabilities on ALL ECGs...")
y_prob = model.predict(all_seq, verbose=1)
print("y_prob shape:", y_prob.shape)

# ------------------------------------------------------------
# 6. PER-CLASS THRESHOLD SWEEP (F1 MAXIMIZATION)
# ------------------------------------------------------------

thresholds = np.linspace(0.01, 0.99, 99)

best_t = np.zeros(n_classes)
best_f1 = np.zeros(n_classes)

for c in range(n_classes):
    probs = y_prob[:, c]
    true  = y_all[:, c]

    f1_list = []
    for t in thresholds:
        pred = (probs > t).astype(int)
        f1 = f1_score(true, pred, average="binary", zero_division=0)
        f1_list.append(f1)

    f1_arr = np.array(f1_list)
    best_idx = np.argmax(f1_arr)
    best_t[c] = thresholds[best_idx]
    best_f1[c] = f1_arr[best_idx]

    print(f"{LABEL_COLS[c]}: best threshold = {best_t[c]:.3f}, F1 = {best_f1[c]:.4f}")

# Save summary
summary = pd.DataFrame({
    "class": LABEL_COLS,
    "best_threshold": best_t,
    "best_F1": best_f1,
})
summary.to_csv("per_class_thresholds.csv", index=False)
print("Saved per_class_thresholds.csv")

plt.figure(figsize=(7,4))
plt.bar(LABEL_COLS, best_f1)
plt.title("Best F1 per class")
plt.tight_layout()
plt.savefig("per_class_best_f1.png", dpi=200)
plt.close()
print("Saved per_class_best_f1.png")

# ------------------------------------------------------------
# 7. FINAL BINARY PREDICTIONS
# ------------------------------------------------------------

y_pred = (y_prob > best_t.reshape(1,-1)).astype(int)
np.save("y_pred_final.npy", y_pred)
print("Saved final predictions to y_pred_final.npy")

print("\n=== All Done ===")
