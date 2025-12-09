"""
====================================================================
ECG Fairness Project
Per-Class Metrics & Fairness Summary Script
File: compute_per_class_metrics.py  (suggested name)

Description:
    This script:
      1) Loads overall and sex-specific prediction CSVs
      2) Computes per-class confusion matrix statistics (TP/FP/TN/FN)
      3) Computes AUROC, accuracy, sensitivity, specificity per class
      4) Handles NaN / infinite probabilities safely for AUROC
      5) Saves separate per-class metrics tables for:
            - All patients
            - Female patients only
            - Male patients only
====================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ============================================================
# PATH CONFIGURATION: change BASE_DIR to where your CSVs are stored
# ============================================================
BASE_DIR = "/hpc/group/honglab/kkgroup/model 1_trained_results"

# ------------------------------------------------------------
# Utility: given y_true / y_pred / y_prob DataFrames, compute
# a per-class metrics table
# ------------------------------------------------------------
def compute_per_class_table(y_true_df, y_pred_df, y_prob_df, has_suffix):
    """
    y_true_df, y_pred_df, y_prob_df: pandas DataFrame
    has_suffix:
        True  -> column names like 1dAVb_true / 1dAVb_pred / 1dAVb_prob
        False -> column names are just 1dAVb / RBBB / ...
    """
    if has_suffix:
        # Strip *_true suffix to recover base class names
        class_names = [c[:-5] if c.endswith("_true") else c for c in y_true_df.columns]
    else:
        class_names = list(y_true_df.columns)

    rows = []

    for name in class_names:
        if has_suffix:
            c_true = f"{name}_true"
            c_pred = f"{name}_pred"
            c_prob = f"{name}_prob"
        else:
            c_true = name
            c_pred = name
            c_prob = name

        y_t = y_true_df[c_true].values.astype(int)
        y_p = y_pred_df[c_pred].values.astype(int)
        y_s = y_prob_df[c_prob].values.astype(float)

        # ================== TP / FP / TN / FN (ignore NaN in prob) ==================
        tp = int(((y_t == 1) & (y_p == 1)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        tn = int(((y_t == 0) & (y_p == 0)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())

        total = len(y_t)
        positive = int((y_t == 1).sum())

        # Basic metrics
        acc  = (tp + tn) / (total + 1e-8)
        sens = tp / (tp + fn + 1e-8)      # recall / sensitivity
        spec = tn / (tn + fp + 1e-8)

        # ================== AUROC: drop NaN / inf first ==================
        # Some probabilities may be NaN; roc_auc_score does not accept NaN,
        # so we only keep finite-valued samples
        mask_valid = np.isfinite(y_s)
        y_t_auc = y_t[mask_valid]
        y_s_auc = y_s[mask_valid]

        # Track how many valid samples were used (optional)
        used_n  = len(y_t_auc)
        num_nan = total - used_n

        if used_n == 0 or len(np.unique(y_t_auc)) < 2:
            # No valid samples, or labels are all 0 or all 1 → AUROC undefined
            auc = np.nan
        else:
            try:
                auc = roc_auc_score(y_t_auc, y_s_auc)
            except Exception:
                auc = np.nan

        rows.append({
            "Label":       name,
            "TP":          tp,
            "FP":          fp,
            "TN":          tn,
            "FN":          fn,
            "AUROC":       auc,
            "Accuracy":    acc,
            "Sensitivity": sens,
            "Specificity": spec,
            "Positive":    positive,
            "Total":       total,
            "Used_N":      used_n,   # number of samples used for AUROC
            "Num_NaN":     num_nan   # number of samples with NaN/inf prob
        })

    df_out = pd.DataFrame(rows)
    return df_out


# ============================================================
# 1) Overall: All patients
#    Uses: test_y_true.csv / test_y_pred_binary.csv / test_y_prob.csv
# ============================================================
print("=== Computing metrics for ALL patients ===")
true_all = pd.read_csv("/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/test_y_true.csv")
pred_all = pd.read_csv("/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/test_y_pred_binary.csv")
prob_all = pd.read_csv("/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/test_y_prob.csv")

df_all = compute_per_class_table(true_all, pred_all, prob_all, has_suffix=False)
out_all = "/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/per_class_metrics_all.csv"
df_all.to_csv(out_all, index=False)
print("Saved:", out_all)

# ============================================================
# 2) Female only
#    Uses: test_y_true_F.csv / test_y_pred_binary_F.csv / test_y_prob_F.csv
# ============================================================
print("=== Computing metrics for FEMALE patients ===")
true_F = pd.read_csv(os.path.join(BASE_DIR, "test_y_true_F.csv"))
pred_F = pd.read_csv(os.path.join(BASE_DIR, "test_y_pred_binary_F.csv"))
prob_F = pd.read_csv(os.path.join(BASE_DIR, "test_y_prob_F.csv"))

# Automatically detect whether columns use *_true suffix
has_suffix_F = any(col.endswith("_true") for col in true_F.columns)
print("Female CSV has _true suffix? ->", has_suffix_F)

df_F = compute_per_class_table(true_F, pred_F, prob_F, has_suffix=has_suffix_F)
out_F = os.path.join(BASE_DIR, "per_class_metrics_female.csv")
df_F.to_csv(out_F, index=False)
print("Saved:", out_F)

# ============================================================
# 3) Male only
#    Uses: test_y_true_M.csv / test_y_pred_binary_M.csv / test_y_prob_M.csv
# ============================================================
print("=== Computing metrics for MALE patients ===")
true_M = pd.read_csv(os.path.join(BASE_DIR, "test_y_true_M.csv"))
pred_M = pd.read_csv(os.path.join(BASE_DIR, "test_y_pred_binary_M.csv"))
prob_M = pd.read_csv(os.path.join(BASE_DIR, "test_y_prob_M.csv"))

has_suffix_M = any(col.endswith("_true") for col in true_M.columns)
print("Male CSV has _true suffix? ->", has_suffix_M)

df_M = compute_per_class_table(true_M, pred_M, prob_M, has_suffix=has_suffix_M)
out_M = os.path.join(BASE_DIR, "per_class_metrics_male.csv")
df_M.to_csv(out_M, index=False)
print("Saved:", out_M)
