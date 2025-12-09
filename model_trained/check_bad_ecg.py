"""
====================================================================
ECG Fairness Project
ECG Data Quality Scan Script
File: scan_bad_ecg_samples.py  (suggested name)

Description:
    This script:
      1) Loads the full ECG memmap array (X_all.npy)
      2) Scans each ECG sample for NaN or infinite values
      3) Records indices of corrupted samples
      4) Saves bad sample indices for later exclusion in training/testing
====================================================================
"""

import os
import numpy as np

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================
ROOT = "/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/ECG_prepared_no_fold"
X_PATH = os.path.join(ROOT, "X_all.npy")

# ============================================================
# 2. LOAD MEMMAP ECG ARRAY
# ============================================================
print("Loading X_all...")
X_all = np.load(X_PATH, mmap_mode="r")  # (N, 4096, 12)
print("X_all shape:", X_all.shape)

# ============================================================
# 3. SCAN FOR NaN / INF VALUES
# ============================================================
print("Scanning for NaN/inf...")

# Boolean mask marking ECG samples containing NaN or inf
bad_mask = np.isnan(X_all).any(axis=(1, 2)) | np.isinf(X_all).any(axis=(1, 2))
bad_idx = np.where(bad_mask)[0]

print("Number of bad samples:", len(bad_idx))
if len(bad_idx) > 0:
    print("First few bad indices:", bad_idx[:20])

# ============================================================
# 4. SAVE BAD SAMPLE INDICES
# ============================================================
OUT_BAD = os.path.join(ROOT, "bad_indices.npy")
np.save(OUT_BAD, bad_idx)

print("Saved bad_indices to:", OUT_BAD)

