#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate gender-based performance and fairness metrics for Model 1
Generates three CSV files:
1. model1_test_evaluation_results_female.csv - Female group performance metrics
2. model1_test_evaluation_results_male.csv - Male group performance metrics
3. model1_test_fairness_metrics.csv - Fairness metrics
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from scipy.spatial.distance import jensenshannon

# Data paths
DATA_DIR = '/hpc/group/honglab/kkgroup/model 1_trained_results'
OUTPUT_DIR = '/hpc/group/honglab/kkgroup/sicheng_workplace'

print("="*80)
print("Calculating gender-based performance and fairness metrics for Model 1 test set")
print("="*80)

# ===== 1) Load data =====
print("\n1. Loading data...")

# Female group
preds_female_binary = pd.read_csv(os.path.join(DATA_DIR, 'test_y_pred_binary_F.csv'))
preds_female_scores = pd.read_csv(os.path.join(DATA_DIR, 'test_y_prob_F.csv'))
labels_female = pd.read_csv(os.path.join(DATA_DIR, 'test_y_true_F.csv'))

# Male group
preds_male_binary = pd.read_csv(os.path.join(DATA_DIR, 'test_y_pred_binary_M.csv'))
preds_male_scores = pd.read_csv(os.path.join(DATA_DIR, 'test_y_prob_M.csv'))
labels_male = pd.read_csv(os.path.join(DATA_DIR, 'test_y_true_M.csv'))

print(f"   Female samples: {len(preds_female_binary)}")
print(f"   Male samples: {len(preds_male_binary)}")

# Get class names (use label column order as standard)
label_names = labels_female.columns.tolist()
print(f"   Number of classes: {len(label_names)}")
print(f"   Classes: {label_names}")

# Ensure prediction score columns match label order
preds_female_scores = preds_female_scores[label_names]
preds_male_scores = preds_male_scores[label_names]
print(f"   Prediction score columns reordered to match label order")

# ===== 2) Calculate performance metrics for each group =====
print("\n2. Calculating performance metrics...")

def calculate_metrics(y_true_df, y_pred_df, y_pred_scores_df, group_name):
    """Calculate all performance metrics for a group"""
    results = []

    n_total = len(y_true_df)

    for label_name in label_names:
        y_t = y_true_df[label_name].values
        y_p = y_pred_df[label_name].values
        y_p_score = y_pred_scores_df[label_name].values

        # Filter NaN values
        valid_mask = ~np.isnan(y_p_score)
        y_t_clean = y_t[valid_mask]
        y_p_score_clean = y_p_score[valid_mask]

        # Number of positive samples
        n_pos = y_t.sum()

        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        except:
            tn, fp, fn, tp = 0, 0, 0, 0

        # AUROC
        try:
            if y_t_clean.sum() > 0 and y_t_clean.sum() < len(y_t_clean):
                auroc = roc_auc_score(y_t_clean, y_p_score_clean)
            else:
                auroc = np.nan
        except:
            auroc = np.nan

        # Accuracy
        accuracy = accuracy_score(y_t, y_p)

        # Sensitivity (TPR)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Specificity (TNR)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results.append({
            'Label': label_name,
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn),
            'AUROC': auroc,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Positive': int(n_pos),
            'Total': n_total
        })

        print(f"   {group_name} - {label_name}: AUROC={auroc:.4f}, Acc={accuracy:.4f}, Sens={sensitivity:.4f}, Spec={specificity:.4f}, Pos={n_pos}/{n_total}")

    return results

# Calculate female group metrics
print("\nCalculating female group metrics...")
female_results = calculate_metrics(labels_female, preds_female_binary, preds_female_scores, "Female")

# Calculate male group metrics
print("\nCalculating male group metrics...")
male_results = calculate_metrics(labels_male, preds_male_binary, preds_male_scores, "Male")

# ===== 3) Save performance metrics tables =====
print("\n3. Saving performance metrics tables...")

df_female = pd.DataFrame(female_results)
df_male = pd.DataFrame(male_results)

female_file = os.path.join(OUTPUT_DIR, 'model1_test_evaluation_results_female.csv')
male_file = os.path.join(OUTPUT_DIR, 'model1_test_evaluation_results_male.csv')

df_female.to_csv(female_file, index=False)
df_male.to_csv(male_file, index=False)

print(f"   Female group results saved to: {female_file}")
print(f"   Male group results saved to: {male_file}")

# ===== 4) Calculate fairness metrics =====
print("\n4. Calculating fairness metrics...")

def calculate_ece(y_true, y_pred_score, n_bins=10):
    """Calculate Expected Calibration Error"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_score, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred_score[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece

fairness_results = []

for label_name in label_names:
    print(f"   Processing {label_name}...")

    # Female group data
    y_t_f = labels_female[label_name].values
    y_p_binary_f = preds_female_binary[label_name].values
    y_p_score_f = preds_female_scores[label_name].values

    # Filter NaN values in female group
    valid_mask_f = ~np.isnan(y_p_score_f)
    y_t_f_clean = y_t_f[valid_mask_f]
    y_p_score_f_clean = y_p_score_f[valid_mask_f]

    # Male group data
    y_t_m = labels_male[label_name].values
    y_p_binary_m = preds_male_binary[label_name].values
    y_p_score_m = preds_male_scores[label_name].values

    # Filter NaN values in male group
    valid_mask_m = ~np.isnan(y_p_score_m)
    y_t_m_clean = y_t_m[valid_mask_m]
    y_p_score_m_clean = y_p_score_m[valid_mask_m]

    # === 1) Equalized Odds Gap ===
    try:
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_t_f, y_p_binary_f, labels=[0, 1]).ravel()
        tpr_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        fpr_f = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0
    except:
        tpr_f, fpr_f = 0, 0

    try:
        tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_t_m, y_p_binary_m, labels=[0, 1]).ravel()
        tpr_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
        fpr_m = fp_m / (fp_m + tn_m) if (fp_m + tn_m) > 0 else 0
    except:
        tpr_m, fpr_m = 0, 0

    equalized_odds_gap = abs(tpr_f - tpr_m) + abs(fpr_f - fpr_m)

    # === 2) ECE Gap ===
    ece_f = calculate_ece(y_t_f_clean, y_p_score_f_clean)
    ece_m = calculate_ece(y_t_m_clean, y_p_score_m_clean)
    ece_gap = abs(ece_f - ece_m)

    # === 3) JS Divergence ===
    # Calculate JS divergence using histograms (using actual data range)
    # Merge both groups' data to determine range
    all_scores = np.concatenate([y_p_score_f_clean, y_p_score_m_clean])
    score_min = all_scores.min()
    score_max = all_scores.max()

    # Create histograms using actual data range
    hist_f, _ = np.histogram(y_p_score_f_clean, bins=50, range=(score_min, score_max))
    hist_m, _ = np.histogram(y_p_score_m_clean, bins=50, range=(score_min, score_max))

    # Normalize to probability distributions (add smoothing to avoid zero probabilities)
    p_f = hist_f + 1e-10
    p_f = p_f / p_f.sum()

    p_m = hist_m + 1e-10
    p_m = p_m / p_m.sum()

    js_div = jensenshannon(p_f, p_m)

    # === 4) Worst AUC ===
    try:
        if y_t_f_clean.sum() > 0 and y_t_f_clean.sum() < len(y_t_f_clean):
            auroc_f = roc_auc_score(y_t_f_clean, y_p_score_f_clean)
        else:
            auroc_f = np.nan
    except:
        auroc_f = np.nan

    try:
        if y_t_m_clean.sum() > 0 and y_t_m_clean.sum() < len(y_t_m_clean):
            auroc_m = roc_auc_score(y_t_m_clean, y_p_score_m_clean)
        else:
            auroc_m = np.nan
    except:
        auroc_m = np.nan

    # Worst AUC = minimum AUC between two groups
    if not (np.isnan(auroc_f) or np.isnan(auroc_m)):
        worst_auc = min(auroc_f, auroc_m)
    else:
        worst_auc = np.nan

    fairness_results.append({
        'Label': label_name,
        'Equalized_Odds_Gap': equalized_odds_gap,
        'ECE_Gap': ece_gap,
        'JS_Divergence': js_div,
        'Worst_AUC': worst_auc,
    })

# ===== 5) Save fairness metrics table =====
print("\n5. Saving fairness metrics table...")

df_fairness = pd.DataFrame(fairness_results)
fairness_file = os.path.join(OUTPUT_DIR, 'model1_test_fairness_metrics.csv')
df_fairness.to_csv(fairness_file, index=False)

print(f"   Fairness metrics saved to: {fairness_file}")

# ===== 6) Display summary =====
print("\n6. Fairness metrics summary:")
print("="*80)
print(f"Average Equalized Odds Gap: {df_fairness['Equalized_Odds_Gap'].mean():.4f}")
print(f"Average ECE Gap: {df_fairness['ECE_Gap'].mean():.4f}")
print(f"Average JS Divergence: {df_fairness['JS_Divergence'].mean():.4f}")
print(f"Average Worst AUC: {df_fairness['Worst_AUC'].mean():.4f}")

print(f"\nWorst classes (by Equalized Odds Gap):")
print(df_fairness.nlargest(3, 'Equalized_Odds_Gap')[['Label', 'Equalized_Odds_Gap']])

print("\n" + "="*80)
print("Calculation completed!")
print("="*80)

