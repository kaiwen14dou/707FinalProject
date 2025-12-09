"""
====================================================================
ECG Fairness Project
Data Preparation Script (No-Fold Version)
Author: Yuejun Xu
Description:
    This script prepares ECG sequences and diagnostic labels
    for fairness evaluation under male / female / combined groups.
====================================================================
"""

import os
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


# ====================================================================
# 1. PATH CONFIGURATION
# ====================================================================
ROOT = "/hpc/group/honglab/kkgroup/mimic_preproc_out"

# Non-fold versions of the MIMIC-ECG processed tables
DF_ECG_PKL   = f"{ROOT}/memmap/df_memmap.pkl"
DF_DIAG_PKL  = f"{ROOT}/records_w_diag_icd10.pkl"   # Use non-fold diagnostic table

# Memmap waveform storage
MM_NPY  = f"{ROOT}/memmap/memmap.npy"
MM_META = f"{ROOT}/memmap/memmap_meta.npz"

# Output directory
OUT_DIR = "/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/ECG_prepared_no_fold"
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed parameters
TARGET_T = 4096
CH = 12
N_MAX = None
CHUNK = 2000


# ====================================================================
# 2. LOAD & MERGE TABLES
# ====================================================================
print("Loading ECG metadata...")
df_ecg  = pd.read_pickle(DF_ECG_PKL)
df_diag = pd.read_pickle(DF_DIAG_PKL)

keys = ['study_id', 'subject_id', 'ecg_time']

# Merge diagnostic codes + gender (no-fold version)
df = df_ecg.merge(df_diag[keys + ['gender', 'all_diag_all']], 
                  on=keys, how='left')

print("Merged shape:", df.shape)


# ====================================================================
# 3. MAP ICD10 CODES → SIX TARGET DIAGNOSTIC LABELS
# ====================================================================
MAP = {
    '1dAVb': ['I440'],   # First-degree AV block
    'RBBB' : ['I451'],   # Right bundle branch block
    'LBBB' : ['I447'],   # Left bundle branch block
    'SB'   : ['R001'],   # Sinus bradycardia
    'ST'   : ['R000'],   # Sinus tachycardia
    'AF'   : ['I48'],    # Atrial fibrillation
}

def codes_to_labels(codes):
    """Convert ICD-10 diagnostic code list to six binary categories."""
    if not isinstance(codes, (list, tuple)):
        return {k: 0 for k in MAP}
    codes = [str(c).replace('.', '').upper() for c in codes]
    out = {}
    for lab, prefix_list in MAP.items():
        out[lab] = int(
            any(any(c.startswith(p) for p in prefix_list) for c in codes)
        )
    return out

label_df = pd.DataFrame(list(df["all_diag_all"].apply(codes_to_labels)))
df = pd.concat([df, label_df], axis=1)

print("Label positive rates:")
print(df[['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']].mean())


# ====================================================================
# 4. NORMALIZE GENDER → {0: FEMALE, 1: MALE}
# ====================================================================
sex_raw = df["gender"]

if sex_raw.dtype == object:
    sex = sex_raw.astype(str).str.upper().map({
        "M": 1, "MALE": 1,
        "F": 0, "FEMALE": 0
    })
else:
    v = sex_raw.fillna(sex_raw.median())
    sex = (v > v.median()).astype(int)

df["_sex_bin"] = sex.fillna(0).astype(int)


# ====================================================================
# 5. LOAD MEMMAP WAVEFORM INDEXES
# ====================================================================
print("Loading memmap waveform index...")
meta = np.load(MM_META, allow_pickle=True)

starts = np.asarray(meta['start']).astype(np.int64).ravel()
lens   = np.asarray(meta['length']).astype(np.int64).ravel()

dt = meta.get("dtype", np.float32)
if isinstance(dt, np.ndarray):
    dt = dt.item()
dtype = np.dtype(dt)

Xflat = np.memmap(MM_NPY, mode="r", dtype=dtype).reshape(-1, CH)

print("Flat memmap:", Xflat.shape, Xflat.dtype)

assert len(df) == len(starts) == len(lens), \
    "Mismatch between metadata and dataframe sample count."


# ====================================================================
# 6. UTILITY FUNCTIONS
# ====================================================================
def pad_or_crop(x):
    """Pad or crop each ECG sequence to TARGET_T × CH."""
    t = x.shape[0]
    if t >= TARGET_T:
        return x[:TARGET_T]
    out = np.zeros((TARGET_T, CH), dtype=np.float32)
    out[:t] = x
    return out


def write_group(name, idx):
    """
    Write ECG waveform arrays and diagnostic labels for a given group.
    Groups: male / female / all
    """
    if N_MAX is not None:
        idx = idx[:N_MAX]

    n = len(idx)
    if n == 0:
        print(f"[{name}] empty, skip")
        return

    x_path = os.path.join(OUT_DIR, f"X_{name}.npy")
    y_path = os.path.join(OUT_DIR, f"labels_{name}.csv")

    Xout = open_memmap(
        x_path, mode="w+", dtype=np.float32,
        shape=(n, TARGET_T, CH)
    )

    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        block_idx = idx[s:e]

        buf = np.empty((e - s, TARGET_T, CH), dtype=np.float32)
        for k, j in enumerate(block_idx):
            st, ln = int(starts[j]), int(lens[j])
            buf[k] = pad_or_crop(np.asarray(Xflat[st:st + ln]))

        Xout[s:e] = buf
        print(f"[{name}] wrote {e}/{n}")

    label_cols = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
    df.iloc[idx][label_cols].astype(int).to_csv(y_path, index=False)

    del Xout
    print(f"[{name}] done → {x_path} , {y_path}")


# ====================================================================
# 7. EXPORT MALE / FEMALE / ALL GROUPS
# ====================================================================
groups = {
    "male":   np.where(df["_sex_bin"].values == 1)[0],
    "female": np.where(df["_sex_bin"].values == 0)[0],
    "all":    np.arange(len(df)),
}

print("Writing output groups...")
for gname, gidx in groups.items():
    write_group(gname, gidx)

print("All groups exported successfully.")
