"""
====================================================================
ECG Fairness Project
Metadata Construction Script (Aligned With Preprocessing Pipeline)
File: build_meta_table.py  (suggested name)

Description:
    This script:
      1) Loads the preprocessed ECG and diagnosis tables
      2) Merges them using the same keys as the main preprocessing pipeline
      3) Converts ICD-10 diagnostic codes into six binary label columns
      4) Normalizes gender into a binary variable
      5) Saves a meta_all.csv file that aligns exactly with X_all.npy rows

This file is required for downstream merging, fairness analysis,
and consistent handling of sample metadata.
====================================================================
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================
# Must match the preprocessing script directories
ROOT_PREPROC = "/hpc/group/honglab/kkgroup/mimic_preproc_out"
DF_ECG_PKL   = f"{ROOT_PREPROC}/memmap/df_memmap.pkl"
DF_DIAG_PKL  = f"{ROOT_PREPROC}/records_w_diag_icd10.pkl"  # non-fold version

# Must match the directory where X_all.npy was created
OUT_DIR = "/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/ECG_prepared_no_fold"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2. LOAD ECG + DIAGNOSIS TABLES
# ============================================================
print("Loading ECG and diagnosis tables...")
df_ecg  = pd.read_pickle(DF_ECG_PKL)
df_diag = pd.read_pickle(DF_DIAG_PKL)

keys = ['study_id', 'subject_id', 'ecg_time']

# Merge exactly as done in the preprocessing pipeline
df = df_ecg.merge(
    df_diag[keys + ['gender', 'all_diag_all']],
    on=keys,
    how='left'
)

print("Merged shape:", df.shape)

# ============================================================
# 3. MAP ICD-10 CODES → SIX BINARY LABELS (MUST MATCH PREPROCESSING)
# ============================================================
MAP = {
    '1dAVb': ['I440'],
    'RBBB' : ['I451'],
    'LBBB' : ['I447'],
    'SB'   : ['R001'],
    'ST'   : ['R000'],
    'AF'   : ['I48'],
}

def codes_to_labels(codes):
    """Convert ICD-10 code list to six diagnostic binary labels."""
    if not isinstance(codes, (list, tuple)):  # handles NaN or missing
        return {k: 0 for k in MAP}
    codes = [str(c).replace('.', '').upper() for c in codes]
    out = {}
    for lab, pref_list in MAP.items():
        out[lab] = int(any(any(c.startswith(p) for p in pref_list) for c in codes))
    return out

label_df = pd.DataFrame(list(df['all_diag_all'].apply(codes_to_labels)))
df = pd.concat([df, label_df], axis=1)

label_cols = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
print("Label positives (overall mean):")
print(df[label_cols].mean())

# ============================================================
# 4. NORMALIZE GENDER INTO BINARY VARIABLE (MATCH PREPROCESSING)
# ============================================================
sex_raw = df['gender']
if sex_raw.dtype == object:
    sex = sex_raw.astype(str).str.upper().map({
        'M': 1, 'MALE': 1,
        'F': 0, 'FEMALE': 0
    })
else:
    v = sex_raw.fillna(sex_raw.median())
    sex = (v > v.median()).astype(int)

df['_sex_bin'] = sex.fillna(0).astype(int)

# ============================================================
# 5. SAVE META TABLE (ALIGNED WITH X_all.npy ROW ORDER)
# ============================================================
meta_cols = ['subject_id', 'study_id', 'ecg_time', 'gender', '_sex_bin'] + label_cols
meta_path = os.path.join(OUT_DIR, "meta_all.csv")

# Use row_idx as index → corresponds exactly to X_all.npy row number
df[meta_cols].to_csv(meta_path, index_label="row_idx")

print(f"Saved meta table to: {meta_path}")
