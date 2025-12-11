#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate gender-based performance and fairness metrics for Model 2
Generates three CSV files:
1. model2_evaluation_results_female.csv - Female group performance metrics
2. model2_evaluation_results_male.csv - Male group performance metrics
3. model2_fairness_metrics.csv - Fairness metrics
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from scipy.spatial.distance import jensenshannon

# Label names
LABEL_NAMES = ['IAVB', 'AF', 'AFL', 'Brady', 'RBBB', 'IRBBB', 'LAnFB', 'LAD',
               'LBBB', 'LQRSV', 'NSIVCB', 'PR', 'PAC', 'PVC', 'LPR', 'LQT',
               'QAb', 'RAD', 'SA', 'SB', 'SNR', 'STach', 'TAb', 'TInv']

print("="*80)
print("Calculating gender-based performance and fairness metrics for Model 2")
print("="*80)

# ===== 1) Load true labels and gender information =====
print("\n1. Loading true labels and gender information...")
test_csv = '/hpc/group/honglab/kkgroup/sicheng_workplace/test_data/test_data.csv'
df_true = pd.read_csv(test_csv)

print(f"   Test samples: {len(df_true)}")
print(f"   Gender distribution:")
print(f"     Female: {(df_true['gender'] == 'F').sum()}")
print(f"     Male: {(df_true['gender'] == 'M').sum()}")
print(f"     Unknown: {(df_true['gender'] == 'Unknown').sum()}")

# ===== 2) Load prediction results =====
print("\n2. Loading prediction results...")
output_dir = '/hpc/group/honglab/kkgroup/sicheng_workplace/test_outputs'

y_pred_labels = []
y_pred_scores = []
valid_indices = []

for i in range(len(df_true)):
    if (i + 1) % 10000 == 0:
        print(f"   Progress: {i+1}/{len(df_true)}")
    
    pred_file = os.path.join(output_dir, f'test_sample_{i:06d}.csv')
    
    try:
        with open(pred_file, 'r') as f:
            lines = f.readlines()
        
        labels = [int(x) for x in lines[2].strip().split(',')]
        scores = [float(x) for x in lines[3].strip().split(',')]

        # Skip samples with NaN
        if any(np.isnan(scores)):
            continue

        y_pred_labels.append(labels)
        y_pred_scores.append(scores)
        valid_indices.append(i)
    except Exception as e:
        continue

y_pred_labels = np.array(y_pred_labels)
y_pred_scores = np.array(y_pred_scores)

# Keep only valid samples
df_valid = df_true.iloc[valid_indices].reset_index(drop=True)
y_true = df_valid[LABEL_NAMES].values

print(f"   Valid samples: {len(valid_indices)}")

# ===== 3) Group by gender =====
print("\n3. Grouping by gender...")
female_mask = df_valid['gender'] == 'F'
male_mask = df_valid['gender'] == 'M'

n_female = female_mask.sum()
n_male = male_mask.sum()

print(f"   Female samples: {n_female}")
print(f"   Male samples: {n_male}")

# ===== 4) Calculate performance metrics for each group =====
print("\n4. Calculating performance metrics...")

def calculate_metrics(y_true_group, y_pred_labels_group, y_pred_scores_group, group_name, n_total):
    """Calculate all performance metrics for a group"""
    results = []

    for i, label_name in enumerate(LABEL_NAMES):
        y_t = y_true_group[:, i]
        y_p_label = y_pred_labels_group[:, i]
        y_p_score = y_pred_scores_group[:, i]

        # Number of positive samples
        n_pos = y_t.sum()

        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y_t, y_p_label, labels=[0, 1]).ravel()
        except:
            tn, fp, fn, tp = 0, 0, 0, 0

        # AUROC
        try:
            if n_pos > 0 and n_pos < len(y_t):
                auroc = roc_auc_score(y_t, y_p_score)
            else:
                auroc = np.nan
        except:
            auroc = np.nan

        # Accuracy
        accuracy = accuracy_score(y_t, y_p_label)

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

        print(f"   {group_name} - {label_name}: AUROC={auroc:.4f}, Sens={sensitivity:.4f}, Spec={specificity:.4f}, Pos={n_pos}/{n_total}")

    return results

# Calculate female group metrics
print("\nCalculating female group metrics...")
female_results = calculate_metrics(
    y_true[female_mask],
    y_pred_labels[female_mask],
    y_pred_scores[female_mask],
    "Female",
    n_female
)

# Calculate male group metrics
print("\nCalculating male group metrics...")
male_results = calculate_metrics(
    y_true[male_mask],
    y_pred_labels[male_mask],
    y_pred_scores[male_mask],
    "Male",
    n_male
)

# ===== 5) Save performance metrics tables =====
print("\n5. Saving performance metrics tables...")

df_female = pd.DataFrame(female_results)
df_male = pd.DataFrame(male_results)

female_file = '/hpc/group/honglab/kkgroup/sicheng_workplace/model2_evaluation_results_female.csv'
male_file = '/hpc/group/honglab/kkgroup/sicheng_workplace/model2_evaluation_results_male.csv'

df_female.to_csv(female_file, index=False)
df_male.to_csv(male_file, index=False)

print(f"   Female group results saved to: {female_file}")
print(f"   Male group results saved to: {male_file}")

# ===== 6) Calculate fairness metrics =====
print("\n6. Calculating fairness metrics...")

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

for i, label_name in enumerate(LABEL_NAMES):
    print(f"   Processing {label_name}...")

    # Female group data
    y_t_f = y_true[female_mask, i]
    y_p_label_f = y_pred_labels[female_mask, i]
    y_p_score_f = y_pred_scores[female_mask, i]

    # Male group data
    y_t_m = y_true[male_mask, i]
    y_p_label_m = y_pred_labels[male_mask, i]
    y_p_score_m = y_pred_scores[male_mask, i]

    # Check if there are enough samples
    if y_t_f.sum() == 0 or y_t_m.sum() == 0:
        print(f"     Skipped (no positive samples in one group)")
        continue

    # === 1) Equalized Odds Gap ===
    try:
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_t_f, y_p_label_f, labels=[0, 1]).ravel()
        tpr_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        fpr_f = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0
    except:
        tpr_f, fpr_f = 0, 0

    try:
        tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_t_m, y_p_label_m, labels=[0, 1]).ravel()
        tpr_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
        fpr_m = fp_m / (fp_m + tn_m) if (fp_m + tn_m) > 0 else 0
    except:
        tpr_m, fpr_m = 0, 0

    equalized_odds_gap = abs(tpr_f - tpr_m) + abs(fpr_f - fpr_m)

    # === 2) ECE Gap ===
    ece_f = calculate_ece(y_t_f, y_p_score_f)
    ece_m = calculate_ece(y_t_m, y_p_score_m)
    ece_gap = abs(ece_f - ece_m)

    # === 3) JS Divergence ===
    # Calculate JS divergence using histograms
    hist_f, _ = np.histogram(y_p_score_f, bins=20, range=(0, 1))
    hist_m, _ = np.histogram(y_p_score_m, bins=20, range=(0, 1))

    # Normalize to probability distributions (add smoothing to avoid zero probabilities)
    p_f = hist_f + 1e-10
    p_f = p_f / p_f.sum()

    p_m = hist_m + 1e-10
    p_m = p_m / p_m.sum()

    js_div = jensenshannon(p_f, p_m)

    # === 4) Worst AUC ===
    try:
        if y_t_f.sum() > 0 and y_t_f.sum() < len(y_t_f):
            auroc_f = roc_auc_score(y_t_f, y_p_score_f)
        else:
            auroc_f = np.nan
    except:
        auroc_f = np.nan

    try:
        if y_t_m.sum() > 0 and y_t_m.sum() < len(y_t_m):
            auroc_m = roc_auc_score(y_t_m, y_p_score_m)
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
        'Worst_AUC': worst_auc
    })

# ===== 7) Save fairness metrics table =====
print("\n7. Saving fairness metrics table...")

df_fairness = pd.DataFrame(fairness_results)
fairness_file = '/hpc/group/honglab/kkgroup/sicheng_workplace/model2_fairness_metrics.csv'
df_fairness.to_csv(fairness_file, index=False)

print(f"   Fairness metrics saved to: {fairness_file}")

# ===== 8) Display summary =====
print("\n8. Fairness metrics summary:")
print("="*80)
print(f"Average Equalized Odds Gap: {df_fairness['Equalized_Odds_Gap'].mean():.4f}")
print(f"Average ECE Gap: {df_fairness['ECE_Gap'].mean():.4f}")
print(f"Average JS Divergence: {df_fairness['JS_Divergence'].mean():.4f}")
print(f"Average Worst AUC: {df_fairness['Worst_AUC'].mean():.4f}")

print(f"\nMost unfair classes (by Equalized Odds Gap):")
print(df_fairness.nlargest(5, 'Equalized_Odds_Gap')[['Label', 'Equalized_Odds_Gap']])

print("\n" + "="*80)
print("Calculation completed!")
print("="*80)

