"""
====================================================================
ECG Fairness Project
Subject-Level Train/Val/Test Split Script
File: make_subject_level_splits.py  (suggested name)

Description:
    This script:
      1) Loads the meta_all.csv table with one row per ECG
      2) Performs a subject-level split into train / validation / test
         (no subject appears in more than one split)
      3) Preserves label prevalence summaries for each split
      4) Saves ECG-level index arrays as:
            - train_indices.npy
            - val_indices.npy
            - test_indices.npy
====================================================================
"""

import os
import numpy as np
import pandas as pd

ROOT_FAIR = "/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/ECG_prepared_no_fold"
META_PATH = os.path.join(ROOT_FAIR, "meta_all.csv")

df_meta = pd.read_csv(META_PATH)
print("meta shape:", df_meta.shape)

label_cols = ['1dAVb','RBBB','LBBB','SB','ST','AF']

# 1. Get all unique patients and shuffle
unique_subj = df_meta['subject_id'].unique()
rng = np.random.RandomState(42)
rng.shuffle(unique_subj)

n_pat = len(unique_subj)
n_train = int(0.7 * n_pat)
n_val   = int(0.1 * n_pat)
n_test  = n_pat - n_train - n_val

train_subj = set(unique_subj[:n_train])
val_subj   = set(unique_subj[n_train:n_train+n_val])
test_subj  = set(unique_subj[n_train+n_val:])

print(f"#patients: total={n_pat}, train={len(train_subj)}, val={len(val_subj)}, test={len(test_subj)}")

# 2. Expand to ECG-level indices
train_idx = df_meta.index[df_meta['subject_id'].isin(train_subj)].to_numpy()
val_idx   = df_meta.index[df_meta['subject_id'].isin(val_subj)].to_numpy()
test_idx  = df_meta.index[df_meta['subject_id'].isin(test_subj)].to_numpy()

print(f"#ECGs: total={len(df_meta)}, "
      f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# 3. Sanity checks
assert train_subj.isdisjoint(val_subj)
assert train_subj.isdisjoint(test_subj)
assert val_subj.isdisjoint(test_subj)
assert len(train_idx) + len(val_idx) + len(test_idx) == len(df_meta)

print("OK: subject_id sets are disjoint, and sample counts sum to total.")

print("\nLabel prevalence (overall):")
print(df_meta[label_cols].mean())

print("\nLabel prevalence (train):")
print(df_meta.loc[train_idx, label_cols].mean())

print("\nLabel prevalence (val):")
print(df_meta.loc[val_idx, label_cols].mean())

print("\nLabel prevalence (test):")
print(df_meta.loc[test_idx, label_cols].mean())

# 4. Save indices
np.save(os.path.join(ROOT_FAIR, "train_indices.npy"), train_idx)
np.save(os.path.join(ROOT_FAIR, "val_indices.npy"),   val_idx)
np.save(os.path.join(ROOT_FAIR, "test_indices.npy"),  test_idx)

print("\nSaved indices to train_indices.npy / val_indices.npy / test_indices.npy")
