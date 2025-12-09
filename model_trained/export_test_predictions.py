"""
====================================================================
ECG Fairness Project
Test-Time Prediction & Export Script
File: export_test_predictions.py

Description:
    This script:
      1) Loads the best-trained ECG fairness model and test set indices
      2) Predicts per-class probabilities on the test set
      3) Applies per-class thresholds to obtain binary predictions
      4) Saves results in multiple formats (.npy, .npz, .csv)
      5) Exports test row metadata for downstream analysis
      6) Splits predictions by sex (male / female) and saves CSVs
====================================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# ============================================================
# 0. PATH CONFIG
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT = "/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/ECG_prepared_no_fold"

X_PATH    = os.path.join(ROOT, "X_all.npy")
Y_PATH    = os.path.join(ROOT, "labels_all.csv")
TEST_I    = os.path.join(ROOT, "test_indices.npy")

MODEL_PATH       = os.path.join(SCRIPT_DIR, "model_fairness_weighted_best.h5")
THRESHOLD_CSV    = os.path.join(SCRIPT_DIR, "per_class_thresholds.csv")

# All exported files will be saved under SCRIPT_DIR
OUT_NPZ_PATH     = os.path.join(SCRIPT_DIR, "test_predictions_package.npz")
OUT_Y_PROB_PATH  = os.path.join(SCRIPT_DIR, "test_y_prob.npy")
OUT_Y_PRED_PATH  = os.path.join(SCRIPT_DIR, "test_y_pred_binary.npy")
OUT_Y_TRUE_PATH  = os.path.join(SCRIPT_DIR, "test_y_true.npy")
OUT_META_CSV     = os.path.join(SCRIPT_DIR, "test_row_metadata.csv")

# ============================================================
# 1. Load labels, indices, thresholds
# ============================================================

print("Loading test indices and labels...")
test_idx = np.load(TEST_I)            # (N_test,)
labels_df = pd.read_csv(Y_PATH)       # keep column names
class_names = list(labels_df.columns)
y_all = labels_df.values.astype("float32")   # (N_all, C)
y_test = y_all[test_idx]                      # (N_test, C)

n_classes = y_test.shape[1]
print(f"Test size = {len(test_idx)}, n_classes = {n_classes}")
print("Classes:", class_names)

# Load per-class thresholds
print("Loading per-class thresholds from:", THRESHOLD_CSV)
thr_df = pd.read_csv(THRESHOLD_CSV)

# Sanity check: ensure class order matches
if list(thr_df["class"]) != class_names:
    print("[WARNING] class order in per_class_thresholds.csv "
          "does not match labels_all.csv – please double-check!")
best_thresholds = thr_df["best_threshold"].values.astype("float32")   # shape (C,)

print("Per-class thresholds:", best_thresholds)

# ============================================================
# 2. Load X_all as memmap + build TestSequence
# ============================================================

print("Opening X_all as memmap...")
X_all = np.load(X_PATH, mmap_mode="r")   # (N_all, 4096, 12)
print("X_all shape:", X_all.shape)

class TestSequence(tf.keras.utils.Sequence):
    """No shuffle; uses the same normalization as training."""
    def __init__(self, X_mem, indices, batch_size=256):
        self.X_mem   = X_mem
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = np.asarray(self.X_mem[batch_idx], dtype=np.float32)

        # Per-sample normalization (same as during training)
        mean = X.mean(axis=(1, 2), keepdims=True)
        std  = X.std(axis=(1, 2), keepdims=True) + 1e-6
        X = (X - mean) / std
        X = np.clip(X, -5.0, 5.0)
        return X

test_seq = TestSequence(X_all, test_idx, batch_size=256)

# ============================================================
# 3. Load model & predict probabilities
# ============================================================

print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Predicting probabilities on test set...")
y_prob = model.predict(test_seq, verbose=1)   # (N_test, C)
print("y_prob shape:", y_prob.shape)

# ============================================================
# 4. Convert to binary predictions using per-class thresholds
# ============================================================

# reshape (1, C) for broadcasting
thr_row = best_thresholds.reshape(1, -1)
y_pred_binary = (y_prob > thr_row).astype("int8")   # (N_test, C)

print("y_pred_binary shape:", y_pred_binary.shape)
print("y_true shape:", y_test.shape)

# Sanity check: ensure no NaNs
print("Any NaN in y_prob? ", np.isnan(y_prob).any())
print("Any NaN in y_pred_binary? ", np.isnan(y_pred_binary).any())
print("Any NaN in y_test? ", np.isnan(y_test).any())

# ============================================================
# 5. Save results for collaborators
# ============================================================

# 1) Save individual .npy files
np.save(OUT_Y_PROB_PATH,  y_prob)
np.save(OUT_Y_PRED_PATH,  y_pred_binary)
np.save(OUT_Y_TRUE_PATH,  y_test)

print("Saved prob to   :", OUT_Y_PROB_PATH)
print("Saved binary to :", OUT_Y_PRED_PATH)
print("Saved labels to :", OUT_Y_TRUE_PATH)

# 2) Pack into a single .npz file for convenience
np.savez_compressed(
    OUT_NPZ_PATH,
    y_prob=y_prob,
    y_pred_binary=y_pred_binary,
    y_true=y_test,
    test_indices=test_idx,
    class_names=np.array(class_names),
    per_class_thresholds=best_thresholds
)
print("Saved packed file to:", OUT_NPZ_PATH)

# 3) Export metadata CSV (for merging downstream)
#    Currently includes only rows for test_idx; can be extended (e.g., sex/subject_id)
meta = pd.read_csv(os.path.join(ROOT, "meta_all.csv"))
meta_test = meta.iloc[test_idx].copy()
meta_test.reset_index(drop=True, inplace=True)
meta_test.to_csv(OUT_META_CSV, index=False)
print("Saved test_row_metadata.csv to:", OUT_META_CSV)

print("\nDone. You can share:")
print("  - test_predictions_package.npz")
print("  - test_y_prob.npy / test_y_pred_binary.npy / test_y_true.npy")
print("  - test_row_metadata.csv")

# ============================================================
# 6. Also export CSV versions for collaborators
# ============================================================

import pandas as pd

# y_prob → CSV
df_prob = pd.DataFrame(y_prob, columns=class_names)
df_prob.to_csv(os.path.join(SCRIPT_DIR, "test_y_prob.csv"), index=False)
print("Saved CSV: test_y_prob.csv")

# y_pred_binary → CSV
df_pred = pd.DataFrame(y_pred_binary, columns=class_names)
df_pred.to_csv(os.path.join(SCRIPT_DIR, "test_y_pred_binary.csv"), index=False)
print("Saved CSV: test_y_pred_binary.csv")

# y_true → CSV
df_true = pd.DataFrame(y_test, columns=class_names)
df_true.to_csv(os.path.join(SCRIPT_DIR, "test_y_true.csv"), index=False)
print("Saved CSV: test_y_true.csv")

# -----------------------------------------------------
# Split indices by sex
# -----------------------------------------------------
female_mask = (sex_test == "F")
male_mask   = (sex_test == "M")

# female subgroup
y_prob_F = y_prob[female_mask]
y_pred_F = y_pred_final[female_mask]
y_true_F = y_test[female_mask]

# male subgroup
y_prob_M = y_prob[male_mask]
y_pred_M = y_pred_final[male_mask]
y_true_M = y_test[male_mask]

# -----------------------------------------------------
# Convert to DataFrames with proper column names
# -----------------------------------------------------
df_prob_F = pd.DataFrame(y_prob_F, columns=[f"{c}_prob" for c in class_names])
df_pred_F = pd.DataFrame(y_pred_F, columns=[f"{c}_pred" for c in class_names])
df_true_F = pd.DataFrame(y_true_F, columns=[f"{c}_true" for c in class_names])

df_prob_M = pd.DataFrame(y_prob_M, columns=[f"{c}_prob" for c in class_names])
df_pred_M = pd.DataFrame(y_pred_M, columns=[f"{c}_pred" for c in class_names])
df_true_M = pd.DataFrame(y_true_M, columns=[f"{c}_true" for c in class_names])

# -----------------------------------------------------
# Save CSVs
# -----------------------------------------------------
df_prob_F.to_csv("test_y_prob_F.csv", index=False)
df_pred_F.to_csv("test_y_pred_binary_F.csv", index=False)
df_true_F.to_csv("test_y_true_F.csv", index=False)

df_prob_M.to_csv("test_y_prob_M.csv", index=False)
df_pred_M.to_csv("test_y_pred_binary_M.csv", index=False)
df_true_M.to_csv("test_y_true_M.csv", index=False)

print("Saved 6 CSV files:")
print(" - test_y_prob_F.csv")
print(" - test_y_pred_binary_F.csv")
print(" - test_y_true_F.csv")
print(" - test_y_prob_M.csv")
print(" - test_y_pred_binary_M.csv")
print(" - test_y_true_M.csv")
