"""
====================================================================
ECG Fairness Project
Training Script (TensorFlow, Weighted BCE)
File: train_fairness_tf.py

Description:
    This script trains a multi-label ECG diagnosis model using
    TensorFlow/Keras on preprocessed memmap ECG data.

    Key features:
    - Uses memmap-based loader to avoid loading all ECGs into RAM
    - Per-class positive weighting to address label imbalance
    - Train/validation split based on precomputed index files
    - Optional filtering of known bad samples (bad_indices.npy)
    - Standard Keras callbacks: LR scheduling, early stopping,
      TensorBoard logging, CSV logging, and model checkpointing
====================================================================
"""

# train_fairness_tf.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard,
    ReduceLROnPlateau, CSVLogger, EarlyStopping
)

from model import get_model

# =============== Global Settings ===============
# Fix random seeds (optional)
SEED = 2025
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============== Path Configuration ===============
ROOT = "/hpc/group/honglab/kkgroup/yuejun_workspace/ECG_fairness/ECG_prepared_no_fold"

X_PATH   = os.path.join(ROOT, "X_all.npy")
Y_PATH   = os.path.join(ROOT, "labels_all.csv")
TRAIN_I  = os.path.join(ROOT, "train_indices.npy")
VAL_I    = os.path.join(ROOT, "val_indices.npy")
TEST_I   = os.path.join(ROOT, "test_indices.npy")   # Used later for test/fairness evaluation

# ===== Debug switch: set to True only when debugging with small subset =====
DEBUG_SMALL        = False
MAX_TRAIN_SAMPLES  = 5000
MAX_VAL_SAMPLES    = 1000

BATCH_SIZE = 64
EPOCHS     = 20          # Training epochs (can be modified)
LR         = 1e-4        # Learning rate

# =============== 1. Load labels + indices ===============
print("Loading labels and indices...")
y_all = pd.read_csv(Y_PATH).values.astype("float32")   # (N, C)

train_idx = np.load(TRAIN_I)
val_idx   = np.load(VAL_I)
test_idx  = np.load(TEST_I)

# =============== 2. Filter bad samples (bad_indices) ===============
BAD_I_PATH = os.path.join(ROOT, "bad_indices.npy")
if os.path.exists(BAD_I_PATH):
    bad_idx = np.load(BAD_I_PATH)
    bad_set = set(bad_idx.tolist())
    print(f"Found {len(bad_set)} bad samples, filtering them out of indices...")

    def filter_indices(idx, bad_set):
        idx = np.asarray(idx, dtype=np.int64)
        mask = np.array([i not in bad_set for i in idx], dtype=bool)
        return idx[mask]

    train_idx = filter_indices(train_idx, bad_set)
    val_idx   = filter_indices(val_idx, bad_set)
    test_idx  = filter_indices(test_idx, bad_set)

    print(f"After filtering, Train/Val/Test sizes: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")
else:
    print("No bad_indices.npy found, skipping filtering.")

# =============== 3. Compute class weights (positive reweighting) on train set only ===============
y_train = y_all[train_idx]   # (N_train, n_classes)
pos_counts = y_train.sum(axis=0)
neg_counts = len(y_train) - pos_counts
eps = 1e-6
pos_weight_raw = (neg_counts + eps) / (pos_counts + eps)

print("pos_counts:", pos_counts)
print("neg_counts:", neg_counts)
print("raw pos_weight:", pos_weight_raw)

# Clip very large weights to avoid extreme imbalance
MAX_W = 50.0
pos_weight_np = np.minimum(pos_weight_raw, MAX_W)
print("clipped pos_weight:", pos_weight_np)

pos_weight = tf.constant(pos_weight_np.astype("float32"))

# =============== 4. Debug small-sample trimming (done after computing class weights) ===============
if DEBUG_SMALL:
    train_idx = train_idx[:MAX_TRAIN_SAMPLES]
    val_idx   = val_idx[:MAX_VAL_SAMPLES]
    print(f"[DEBUG] Using only {len(train_idx)} train and {len(val_idx)} val samples")

print(f"Total samples: {len(y_all)}")
print(f"Train/Val/Test index sizes: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")

# =============== 5. Open X_all using memmap ===============
print("Opening X_all as memmap...")
X_all = np.load(X_PATH, mmap_mode="r")  # (N, 4096, 12)
print("X_all shape:", X_all.shape)

n_classes = y_all.shape[1]

# Basic sanity check
print("Checking NaN/inf in first 100 samples of X_all:")
print("  any NaN:", np.isnan(X_all[:100]).any(),
      " any inf:", np.isinf(X_all[:100]).any())
print("Check unique labels:", np.unique(y_all))

# =============== 6. Define Sequence (Normalization + NaN checks) ===============
class MemmapECGSequence(tf.keras.utils.Sequence):
    def __init__(self, X_mem, y_all, indices, batch_size=64, shuffle=True):
        self.X_mem      = X_mem
        self.y_all      = y_all
        self.indices    = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end   = min((idx + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]

        X = np.asarray(self.X_mem[batch_indices], dtype=np.float32)   # (B, 4096, 12)
        y = self.y_all[batch_indices].astype("float32")               # (B, C)

        # ----- Per-sample normalization -----
        mean = X.mean(axis=(1, 2), keepdims=True)
        std  = X.std(axis=(1, 2), keepdims=True) + 1e-6
        X = (X - mean) / std
        X = np.clip(X, -5.0, 5.0)
        # ------------------------------------

        # ====== NaN / inf checks ======
        if np.isnan(X).any() or np.isinf(X).any():
            print(">>> Found NaN/inf in X batch. batch_indices =", batch_indices[:10], "...")
            raise ValueError("NaN/inf in X")
        if np.isnan(y).any() or np.isinf(y).any():
            print(">>> Found NaN/inf in y batch. batch_indices =", batch_indices[:10], "...")
            raise ValueError("NaN/inf in y")
        # ==============================

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


train_seq = MemmapECGSequence(
    X_all, y_all, train_idx,
    batch_size=BATCH_SIZE, shuffle=True
)
val_seq   = MemmapECGSequence(
    X_all, y_all, val_idx,
    batch_size=BATCH_SIZE, shuffle=False
)

# =============== 7. Loss: BCE with per-class positive weights ===============
def bce_with_pos_weight(y_true, y_pred):
    """
    BCE with per-class pos_weight:
    loss = - [ pos_weight * y * log(p) + (1-y) * log(1-p) ]
    """
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    # Broadcasting: pos_weight shape = (C,) → automatically expands to (B, C)
    loss_pos = pos_weight * y_true * tf.math.log(y_pred)
    loss_neg = (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    loss = -(loss_pos + loss_neg)
    return tf.reduce_mean(loss)

# =============== 8. Build and compile model ===============
model = get_model(n_classes=n_classes, last_layer='sigmoid')
model.summary()

opt = Adam(learning_rate=LR, clipnorm=1.0)   # Gradient clipping to avoid explosion

auc = tf.keras.metrics.AUC(
    multi_label=True,
    num_labels=n_classes,
    name="auc"
)

model.compile(
    optimizer=opt,
    loss=bce_with_pos_weight,
    metrics=[auc]
)

# =============== 9. Callbacks ===============
callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        min_lr=LR / 100
    ),
    EarlyStopping(
        patience=5,
        min_delta=1e-5,
        restore_best_weights=True
    ),
    TensorBoard(
        log_dir='./logs_fairness_weighted',
        write_graph=False
    ),
    CSVLogger('training_fairness_weighted.log', append=False),
    ModelCheckpoint('./model_fairness_weighted_last.h5'),
    ModelCheckpoint('./model_fairness_weighted_best.h5', save_best_only=True),
]

# =============== 10. Training ===============
history = model.fit(
    train_seq,
    epochs=EPOCHS,
    validation_data=val_seq,
    callbacks=callbacks,
    verbose=1
)

model.save("./model_fairness_weighted_final.h5")
print("Training finished, model saved to model_fairness_weighted_final.h5")
