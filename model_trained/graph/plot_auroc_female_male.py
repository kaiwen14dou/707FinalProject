"""
====================================================================
ECG Fairness Project
Sex-Specific AUROC Plotting Script
File: plot_auroc_female_vs_male.py  (suggested name)

Description:
    This script:
      1) Loads sex-specific true labels and predicted probabilities
         (female and male cohorts)
      2) Computes ROC curves and AUROC for each ECG diagnosis class
         separately for females and males
      3) Plots side-by-side ROC curves (female vs. male) in a 2×3 grid
      4) Saves the resulting figure as a PNG file
====================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ============================================================
# 1. PATHS & CLASS NAMES
# ============================================================
BASE_DIR = "/hpc/group/honglab/kkgroup/model 1_trained_results"  # TODO: update to your own directory if needed

# True labels & probabilities (Female / Male)
TRUE_F_PATH = os.path.join(BASE_DIR, "test_y_true_F.csv")
PROB_F_PATH = os.path.join(BASE_DIR, "test_y_prob_F.csv")

TRUE_M_PATH = os.path.join(BASE_DIR, "test_y_true_M.csv")
PROB_M_PATH = os.path.join(BASE_DIR, "test_y_prob_M.csv")

# ECG diagnosis classes
CLASS_NAMES = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]

# ============================================================
# 2. LOAD DATA
# ============================================================
print("Loading CSVs...")
y_true_F = pd.read_csv(TRUE_F_PATH)
y_prob_F = pd.read_csv(PROB_F_PATH)

y_true_M = pd.read_csv(TRUE_M_PATH)
y_prob_M = pd.read_csv(PROB_M_PATH)

print("Female shape (true, prob):", y_true_F.shape, y_prob_F.shape)
print("Male   shape (true, prob):", y_true_M.shape, y_prob_M.shape)

# Simple check: ensure that all class names are present as columns
for cls in CLASS_NAMES:
    assert cls in y_true_F.columns, f"{cls} not found in female true CSV"
    assert cls in y_prob_F.columns, f"{cls} not found in female prob CSV"
    assert cls in y_true_M.columns, f"{cls} not found in male true CSV"
    assert cls in y_prob_M.columns, f"{cls} not found in male prob CSV"

# ============================================================
# 3. PLOT ROC CURVES (2×3 SUBPLOTS)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

fig.suptitle("AUROC Curves: Female vs Male", fontsize=16)

for idx, cls in enumerate(CLASS_NAMES):
    ax = axes[idx]

    # ------- Female -------
    y_t_F = y_true_F[cls].values.astype(int)
    y_s_F = y_prob_F[cls].values.astype(float)

    # Drop NaN probabilities
    mask_F = ~np.isnan(y_s_F)
    y_t_F = y_t_F[mask_F]
    y_s_F = y_s_F[mask_F]

    if len(np.unique(y_t_F)) >= 2:
        fpr_F, tpr_F, _ = roc_curve(y_t_F, y_s_F)
        auc_F = auc(fpr_F, tpr_F)
    else:
        # Degenerate case: labels are all 0 or all 1
        fpr_F, tpr_F, auc_F = [0, 1], [0, 1], np.nan

    # ------- Male -------
    y_t_M = y_true_M[cls].values.astype(int)
    y_s_M = y_prob_M[cls].values.astype(float)

    mask_M = ~np.isnan(y_s_M)
    y_t_M = y_t_M[mask_M]
    y_s_M = y_s_M[mask_M]

    if len(np.unique(y_t_M)) >= 2:
        fpr_M, tpr_M, _ = roc_curve(y_t_M, y_s_M)
        auc_M = auc(fpr_M, tpr_M)
    else:
        fpr_M, tpr_M, auc_M = [0, 1], [0, 1], np.nan

    # ------- Plot -------
    ax.plot(fpr_F, tpr_F, label=f"Female (AUC = {auc_F:.3f})")
    ax.plot(fpr_M, tpr_M, label=f"Male (AUC = {auc_M:.3f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC = 0.500)")

    ax.set_title(cls)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend(loc="lower right", fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = os.path.join(BASE_DIR, "auroc_female_vs_male_model1.png")
plt.savefig(out_path, dpi=300)
print("Saved figure to:", out_path)
plt.close()
