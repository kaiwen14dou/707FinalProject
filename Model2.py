print(data.shape)
print(data.columns)
print(data.head())

import os, sys, numpy as np, pandas as pd, pickle

# --- patch for old NumPy pickles ---
import numpy
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.numeric'] = numpy.core.numeric

# ===== Paths =====
ROOT = "/hpc/group/honglab/kkgroup/mimic_preproc_out"
DF_ECG_PKL  = f"{ROOT}/memmap/df_memmap.pkl"
DF_DIAG_PKL = f"{ROOT}/records_w_diag_icd10_folds.pkl"
MM_NPY      = f"{ROOT}/memmap/memmap.npy"
MM_META     = f"{ROOT}/memmap/memmap_meta.npz"

# ===== Load tables =====
df_ecg  = pd.read_pickle(DF_ECG_PKL)
df_diag = pd.read_pickle(DF_DIAG_PKL)

keys = ['study_id','subject_id','ecg_time']
assert all(k in df_ecg.columns  for k in keys)
assert all(k in df_diag.columns for k in keys)

df = df_ecg.merge(df_diag[keys + ['gender','all_diag_all','fold']], on=keys, how='left')

# ===== ICD10 → 6 labels =====
# ICD-10 prefix map (no dots, just use prefix matching)
MAP = {
    'SNR' : ['I958', 'R942'],         # Normal sinus rhythm (no arrhythmia; depends on your dataset — can be used as "others"/control)
    'AF'  : ['I48'],                  # Atrial fibrillation / flutter
    'IAVB': ['I440', 'I441', 'I442'], # First/Second-degree AV block
    'LBBB': ['I447'],                 # Left bundle branch block
    'RBBB': ['I451'],                 # Right bundle branch block
    'PAC' : ['I491'],                 # Premature atrial contraction
    'PVC' : ['I493'],                 # Premature ventricular contraction
    'STD' : ['I214', 'I219'],         # ST-segment depression / NSTEMI
    'STE' : ['I213'],                 # ST-segment elevation / STEMI
}

def codes_to_labels(codes):
    if not isinstance(codes, (list, tuple)):
        return {k:0 for k in MAP}
    codes = [str(c).replace('.', '').upper() for c in codes]
    out = {}
    for lab, prefixes in MAP.items():
        out[lab] = int(any(c.startswith(p) for p in prefixes for c in codes))
    return out

labels_df = pd.DataFrame(list(df['all_diag_all'].apply(codes_to_labels)))
df = pd.concat([df, labels_df], axis=1)
# Remove duplicated columns (keep the first occurrence)
df = df.loc[:, ~df.columns.duplicated()]

# Check again
print("Positive rates:")
print(df[list(MAP.keys())].mean().round(4))

arrhythmia_cols = ['SNR','AF','IAVB','LBBB','RBBB','PAC','PVC','STD','STE']

# co-occurrence counts
co_occurrence = df[arrhythmia_cols].T.dot(df[arrhythmia_cols])
print(co_occurrence)

df['n_labels'] = df[arrhythmia_cols].sum(axis=1)

multi_label_df = df[df['n_labels'] > 1]
print("Number of multi-label ECGs:", len(multi_label_df))
multi_label_df[['subject_id','study_id','ecg_time','n_labels'] + arrhythmia_cols].head(10)

df['n_labels'] = df[arrhythmia_cols].sum(axis=1)
summary = df['n_labels'].value_counts().sort_index()
print("ECG-level summary:")
print(summary)

# collapse to one row per patient (if any ECG has that arrhythmia)
patient_labels = df.groupby('subject_id')[arrhythmia_cols].max()

# count number of arrhythmia types per patient
patient_labels['n_labels'] = patient_labels.sum(axis=1)

# summarize
patient_summary = patient_labels['n_labels'].value_counts().sort_index()
print("Patient-level summary:")
print(patient_summary)


arrhythmia_cols = ['SNR','AF','IAVB','LBBB','RBBB','PAC','PVC','STD','STE']

# Keep only ECGs with exactly one positive label
df_single_ecg = df[df[arrhythmia_cols].sum(axis=1) == 1].copy()

# Identify which label it is
df_single_ecg['primary_label'] = df_single_ecg[arrhythmia_cols].idxmax(axis=1)


# Collapse by subject_id to see how many arrhythmia types per patient
patient_labels = df_single_ecg.groupby('subject_id')[arrhythmia_cols].max()
patient_labels['n_labels'] = patient_labels.sum(axis=1)

# Keep patients with exactly one arrhythmia type
single_label_patients = patient_labels[patient_labels['n_labels'] == 1].index
df_single_patient = df_single_ecg[df_single_ecg['subject_id'].isin(single_label_patients)].copy()

print("Final dataset:", df_single_patient.shape)
print("Unique patients:", df_single_patient['subject_id'].nunique())
print("Labels:")
print(df_single_patient['primary_label'].value_counts())

df_single_patient['ecg_time'] = pd.to_datetime(df_single_patient['ecg_time'], errors='coerce')

df_final = (
    df_single_patient
    .sort_values(['subject_id', 'ecg_time'])
    .groupby('subject_id', as_index=False)
    .tail(1)
)

print("Final dataset:", df_final.shape)
print("Unique patients:", df_final['subject_id'].nunique())
print("Labels:")
print(df_final['primary_label'].value_counts())


co_occurrence = df_final[arrhythmia_cols].T.dot(df_final[arrhythmia_cols])
print(co_occurrence)

X_mem = X_mem.reshape(800035, 1000, 12)

# ✅ Verify
print("After reshape:", X_mem.shape)
print("One ECG shape:", X_mem[0].shape)

# Split folds (if you already have a fold column)
train_folds = [1,2,3,4,5,6,7,8]
val_folds   = [9]
test_folds  = [10]

df_train = df_final[df_final['fold'].isin(train_folds)]
df_val   = df_final[df_final['fold'].isin(val_folds)]
df_test  = df_final[df_final['fold'].isin(test_folds)]

train_set = MIMIC_ECG_Dataset('train', df_train, X_mem)
val_set   = MIMIC_ECG_Dataset('val', df_val, X_mem)
test_set  = MIMIC_ECG_Dataset('test', df_test, X_mem)

# Check a sample
x, y = train_set[0]
print("Input shape:", x.shape)
print("Mean:", x.mean().item())
print("Std:", x.std().item())

from torch.utils.data import DataLoader

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
x, y = next(iter(train_loader))
print("Batch ECG shape:", x.shape)   # (64, 12, 4096)
print("Batch labels shape:", y.shape)  # (64, 9)

