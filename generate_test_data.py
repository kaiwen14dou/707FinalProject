#!/usr/bin/env python3
"""
Generate test data: second ECG per patient
Uses the same 24-class mapping as training data
"""

import os, numpy as np, pandas as pd
from scipy.io import savemat

# ===== Path configuration =====
ROOT = "/hpc/group/honglab/kkgroup/mimic_preproc_out"
DF_ECG_PKL   = f"{ROOT}/memmap/df_memmap.pkl"
MM_NPY       = f"{ROOT}/memmap/memmap.npy"
MM_META      = f"{ROOT}/memmap/memmap_meta.npz"
DF_DIAG_PKL  = f"{ROOT}/records_w_diag_icd10_folds.pkl"

OUT_DIR = "/hpc/group/honglab/kkgroup/sicheng_workplace/test_data"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_T = 4096
CH = 12

# ===== PhysioNet 2020 24-class mapping (identical to training data) =====
LABEL_MAP = {
    'IAVB':   ['I440'],
    'AF':     ['I48'],
    'AFL':    ['I481'],
    'Brady':  ['R001'],
    'RBBB':   ['I4510'],
    'IRBBB':  ['I4510'],
    'LAnFB':  ['I442'],
    'LAD':    ['I444', 'R9431'],
    'LBBB':   ['I447'],
    'LQRSV':  ['R9431'],
    'NSIVCB': ['I4560'],
    'PR':     ['Z950'],
    'PAC':    ['I491'],
    'PVC':    ['I493'],
    'LPR':    ['I440', 'R9431'],
    'LQT':    ['I454', 'R9431'],
    'QAb':    ['R9431'],
    'RAD':    ['R9431'],
    'SA':     ['R000'],
    'SB':     ['R001'],
    'SNR':    [],
    'STach':  ['R000'],
    'TAb':    ['R9431'],
    'TInv':   ['R9431'],
}
LABEL_COLS = list(LABEL_MAP.keys())

# ===== Utility functions =====
def codes_to_labels(codes):
    """Convert ICD10 codes to 24-class labels (identical logic to training data)"""
    if not isinstance(codes, (list, tuple)):
        codes = []
    codes = [str(c).replace('.', '').upper() for c in codes]
    out = {}

    # Process all classes except SNR first
    for lab, pref in LABEL_MAP.items():
        if lab == 'SNR':
            continue
        out[lab] = int(any(any(c.startswith(p) for p in pref) for c in codes))

    # SNR: set to 1 when no other abnormality (mutual exclusion logic)
    has_any_abnormality = any(out.values())
    out['SNR'] = int(not has_any_abnormality)

    return out

def pad_or_crop(x):
    """Pad or crop to target length and transpose to (CH, TARGET_T)"""
    t = x.shape[0]
    if t >= TARGET_T:
        cropped = x[:TARGET_T]
    else:
        out = np.zeros((TARGET_T, CH), dtype=np.float32)
        out[:t] = x
        cropped = out
    return cropped.T  # (12, 4096)

print("="*80)
print("Generating test data (second ECG per patient)")
print("="*80)

# ===== 1) Load and merge tables =====
print("\n1. Loading data...")
df_ecg  = pd.read_pickle(DF_ECG_PKL)
df_diag = pd.read_pickle(DF_DIAG_PKL)
keys = ['study_id', 'subject_id', 'ecg_time']
df = df_ecg.merge(df_diag[keys + ['gender', 'all_diag_all']], on=keys, how='left')
print(f"   After merge: {df.shape}")

# Filter second ECG per patient
df = df.sort_values(['subject_id', 'ecg_time'])
df['_original_idx'] = df.index
# Group by patient, take second ECG (if exists)
df_second = df.groupby('subject_id').nth(1).reset_index()
print(f"   After filtering second ECG: {len(df_second)} samples")

# ===== 2) ICD10 -> 24-class labels =====
print("\n2. Extracting 24-class labels...")
label_df = pd.DataFrame(list(df_second['all_diag_all'].apply(codes_to_labels)))
df_second = pd.concat([df_second, label_df], axis=1)
print("   Label positive rates:")
print(df_second[LABEL_COLS].mean())

# ===== 3) Gender standardization =====
print("\n3. Processing gender...")
sex_raw = df_second['gender']
if sex_raw.dtype == object:
    sex = sex_raw.astype(str).str.upper().map({'M': 1, 'MALE': 1, 'F': 0, 'FEMALE': 0})
else:
    v = sex_raw.fillna(sex_raw.median())
    sex = (v > v.median()).astype(int)
df_second['_sex_bin'] = sex.fillna(0).astype(int)

# ===== 4) Load memmap + indices =====
print("\n4. Loading memmap...")
meta = np.load(MM_META, allow_pickle=True)
starts_all = np.asarray(meta['start']).astype(np.int64).ravel()
lens_all   = np.asarray(meta['length']).astype(np.int64).ravel()
dt = meta.get('dtype', np.float32)
if isinstance(dt, np.ndarray): dt = dt.item()
dtype = np.dtype(dt)
Xflat = np.memmap(MM_NPY, mode='r', dtype=dtype).reshape(-1, CH)
print(f"   Memmap: {Xflat.shape}, dtype={Xflat.dtype}")

# Use original indices to get corresponding start and length
original_indices = df_second['_original_idx'].values
starts = starts_all[original_indices]
lens = lens_all[original_indices]
print(f"   Number of filtered indices: {len(starts)}")


# ===== 5) Generate .mat files and CSV =====
print("\n5. Generating .mat files and CSV...")
csv_data = []

for i in range(len(df_second)):
    if (i + 1) % 5000 == 0:
        print(f"   Progress: {i+1}/{len(df_second)}")

    # Load ECG data
    st, ln = int(starts[i]), int(lens[i])
    ecg = np.asarray(Xflat[st:st+ln])  # (Ti, 12)
    ecg = pad_or_crop(ecg)  # (12, 4096)

    # Save .mat file
    filename = f'test_sample_{i:06d}.mat'
    filepath = os.path.join(OUT_DIR, filename)
    savemat(filepath, {'val': ecg})  # (12, 4096)

    # Get gender
    gender_raw = df_second.iloc[i].get('gender', 'Unknown')
    if pd.isna(gender_raw) or gender_raw == 'missing_gender':
        gender = 'Unknown'
    else:
        gender = str(gender_raw).upper()
        if gender in ['M', 'MALE']:
            gender = 'M'
        elif gender in ['F', 'FEMALE']:
            gender = 'F'
        else:
            gender = 'Unknown'

    # Build CSV row (excluding subject_id and fold)
    csv_row = {
        'filename': filepath,
        'fs': 500,
        'age': -1,
        'gender': gender,
    }

    # Add 24-class labels
    for lab in LABEL_COLS:
        csv_row[lab] = int(df_second.iloc[i][lab])

    csv_data.append(csv_row)

print(f"   Generated {len(df_second)} .mat files")

# ===== 6) Save CSV file =====
print("\n6. Saving CSV file...")
csv_df = pd.DataFrame(csv_data)

# Save test CSV
test_csv_path = os.path.join(OUT_DIR, 'test_data.csv')
csv_df.to_csv(test_csv_path, index=False)
print(f"   Test CSV: {test_csv_path}")

print("\n" + "="*80)
print("Test data generation completed!")
print(f"Total samples: {len(df_second)}")
print(f"Number of classes: 24")
print(f"Output directory: {OUT_DIR}")
print("="*80)

