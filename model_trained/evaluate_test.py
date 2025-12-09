"""
====================================================================
ECG Fairness Project
Threshold Selection & Fairness Evaluation Script
File: tune_thresholds_and_fairness.py  (suggested name)

Description:
    This script:
      1) Loads the trained ECG model and test set
      2) Predicts per-class probabilities on the test set
      3) Performs a threshold sweep (0.01–0.99) per class to find
         the best F1 score for each label
      4) Saves per-class optimal thresholds and F1 scores
      5) Plots a bar chart of best F1 per class
      6) Applies the per-class thresholds to get final binary predictions
      7) Computes simple fairness metrics (sensitivity, specificity, F1)
         for male vs. female groups, aggregated over all classes
====================================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# ============================================================
# PATH CONFIG
# ============================================================
ROOT = "/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/ECG_prepared_no_fold"
MODEL_PATH = "/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/model_fairness_weighted_best.h5"

X_PATH    = os.path.join(ROOT, "X_all.npy")
Y_PATH    = os.path.join(ROOT, "labels_all.csv")
TEST_I    = os.path.join(ROOT, "test_indices.npy")
META_PATH = os.path.join(ROOT, "meta_all.csv")

# ============================================================
# 1. Load data
# ============================================================
print("Loading test indices...")
test_idx = np.load(TEST_I)

# Keep column names as class_names
labels_df = pd.read_csv(Y_PATH)
class_names = list(labels_df.columns)
y_all = labels_df.values.astype("float32")   # (N, C)

X_all = np.load(X_PATH, mmap_mode="r")

y_test = y_all[test_idx]     # (N_test, C)

print("Loading metadata...")
meta = pd.read_csv(META_PATH)
sex_test = meta.iloc[test_idx]["gender"].values   # "M"/"F"

n_classes = y_test.shape[1]
print("Test size =", len(test_idx))
print("Classes:", class_names)

# ============================================================
# 2. Build test Sequence (same as training)
# ============================================================
class TestSequence(tf.keras.utils.Sequence):
    def __init__(self, X_mem, indices, batch_size=128):
        self.X_mem = X_mem
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        X = np.asarray(self.X_mem[batch_idx], dtype=np.float32)

        # Standardization (same as during training)
        mean = X.mean(axis=(1, 2), keepdims=True)
        std  = X.std(axis=(1, 2), keepdims=True) + 1e-6
        X = (X - mean) / std
        X = np.clip(X, -5.0, 5.0)
        return X

test_seq = TestSequence(X_all, test_idx, batch_size=256)

# ============================================================
# 3. Load model
# ============================================================
print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# ============================================================
# 4. Predict probabilities on Test Set
# ============================================================
print("Predicting probabilities...")
y_prob = model.predict(test_seq, verbose=1)    # (N_test, C)

# ============================================================
# 5. Per-class threshold sweep (0.01 - 0.99) → best F1 per class
# ============================================================
thresholds = np.linspace(0.01, 0.99, 99)

best_t_per_class = np.zeros(n_classes, dtype=np.float32)
best_f1_per_class = np.zeros(n_classes, dtype=np.float32)

# Optionally also store the F1 curve for each class
all_f1_curves = []

for c in range(n_classes):
    y_true_c = y_test[:, c]
    y_prob_c = y_prob[:, c]

    f1_list = []
    for t in thresholds:
        y_pred_c = (y_prob_c > t).astype(int)
        # Binary F1 for this class; avoid zero-division warnings
        f1 = f1_score(y_true_c, y_pred_c, average="binary", zero_division=0)
        f1_list.append(f1)

    f1_list = np.array(f1_list)
    all_f1_curves.append(f1_list)

    best_idx = np.argmax(f1_list)
    best_t = float(thresholds[best_idx])
    best_f1 = float(f1_list[best_idx])

    best_t_per_class[c] = best_t
    best_f1_per_class[c] = best_f1

    print(f"[{class_names[c]}] best_thresh = {best_t:.3f}, best_F1 = {best_f1:.4f}")

all_f1_curves = np.stack(all_f1_curves, axis=0)   # (C, n_thresholds)

# Save per-class threshold results
df_per_class = pd.DataFrame({
    "class": class_names,
    "best_threshold": best_t_per_class,
    "best_F1": best_f1_per_class
})
df_per_class.to_csv("per_class_thresholds.csv", index=False)
print("Saved: per_class_thresholds.csv")

# Compute a simple summary: mean of per-class best F1
macro_best_f1_mean = best_f1_per_class.mean()
print("\n==============================")
print(" Mean of per-class best F1 =", macro_best_f1_mean)
print("==============================\n")

# Plot a simple bar chart: best F1 for each class
plt.figure(figsize=(8, 4))
plt.bar(range(n_classes), best_f1_per_class)
plt.xticks(range(n_classes), class_names, rotation=45, ha="right")
plt.ylabel("Best F1 per class")
plt.tight_layout()
plt.savefig("per_class_best_f1.png", dpi=200)
print("Saved: per_class_best_f1.png")

# ============================================================
# 6. Compute final binary predictions using per-class thresholds
# ============================================================
# y_pred_final[i, c] = 1  if y_prob[i, c] > best_t_per_class[c]
y_pred_final = (y_prob > best_t_per_class.reshape(1, -1)).astype(int)

# ============================================================
# 7. Fairness metrics (by sex, aggregated over classes)
# ============================================================
def sensitivity(y_true, y_pred):
    # Aggregate over all classes
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn + 1e-6)

def specificity(y_true, y_pred):
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tn / (tn + fp + 1e-6)

rows = []
for group_name, group_value in [("Male", "M"), ("Female", "F")]:
    mask = (sex_test == group_value)

    y_true_g = y_test[mask]
    y_pred_g = y_pred_final[mask]

    sens = sensitivity(y_true_g, y_pred_g)
    spec = specificity(y_true_g, y_pred_g)
    f1g  = f1_score(y_true_g.ravel(), y_pred_g.ravel(),
                    average="binary", zero_division=0)

    rows.append([group_name, sens, spec, f1g])

df_group = pd.DataFrame(rows,
                        columns=["Group", "Sensitivity", "Specificity", "MacroF1_overall"])
df_group.to_csv("test_group_metrics.csv", index=False)
print("Saved: test_group_metrics.csv")

print("\n=== Fairness Results (aggregated over all classes) ===")
print(df_group)
