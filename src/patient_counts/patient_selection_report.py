#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from mimic_ecg_preprocessing import prepare_mimic_ecg


def ensure_records(zip_file_path: Path, target_path: Path) -> pd.DataFrame:
    from extract_headers import extract_and_open_files_in_zip

    records_path = target_path / "records.pkl"
    if records_path.exists():
        return pd.read_pickle(records_path)
    target_path.mkdir(parents=True, exist_ok=True)
    df_records = extract_and_open_files_in_zip(zip_file_path, ".hea")
    df_records.to_pickle(records_path)
    return df_records


def _load_hosp_metadata(mimic_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_hosp_icd_description = pd.read_csv(mimic_path / "hosp/d_icd_diagnoses.csv.gz")
    df_hosp_icd_diagnoses = pd.read_csv(mimic_path / "hosp/diagnoses_icd.csv.gz")
    df_hosp_admissions = pd.read_csv(mimic_path / "hosp/admissions.csv.gz")
    df_hosp_admissions["admittime"] = pd.to_datetime(df_hosp_admissions["admittime"])
    df_hosp_admissions["dischtime"] = pd.to_datetime(df_hosp_admissions["dischtime"])
    df_hosp_admissions["deathtime"] = pd.to_datetime(df_hosp_admissions["deathtime"])
    return df_hosp_icd_description, df_hosp_icd_diagnoses, df_hosp_admissions


def ensure_records_with_diag(
    mimic_path: Path,
    target_path: Path,
    df_records: pd.DataFrame,
) -> pd.DataFrame:
    diag_path = target_path / "records_w_diag.pkl"
    if diag_path.exists():
        return pd.read_pickle(diag_path)

    from icdmappings import Mapper

    mapper = Mapper()
    (
        df_hosp_icd_description,
        df_hosp_icd_diagnoses,
        df_hosp_admissions,
    ) = _load_hosp_metadata(mimic_path)

    df_hosp_icd_description["icd10_code"] = df_hosp_icd_description.apply(
        lambda row: row["icd_code"]
        if row["icd_version"] == 10
        else mapper.map(row["icd_code"], source="icd9", target="icd10"),
        axis=1,
    )
    df_ed_stays = pd.read_csv(mimic_path / "ed/edstays.csv.gz")
    df_ed_stays["intime"] = pd.to_datetime(df_ed_stays["intime"])
    df_ed_stays["outtime"] = pd.to_datetime(df_ed_stays["outtime"])
    df_ed_diagnosis = pd.read_csv(mimic_path / "ed/diagnosis.csv.gz")

    def get_diagnosis_hosp(subject_id, ecg_time):
        mask = (
            (df_hosp_admissions.subject_id == subject_id)
            & (df_hosp_admissions.admittime < ecg_time)
            & (
                (df_hosp_admissions.dischtime > ecg_time)
                | (df_hosp_admissions.deathtime > ecg_time)
            )
        )
        df_ecg_during_hosp = df_hosp_admissions[mask]
        if len(df_ecg_during_hosp) == 0:
            return [], np.nan
        if len(df_ecg_during_hosp) > 1:
            print(
                "Warning: multiple hospital admissions matched",
                subject_id,
                ecg_time,
                "- using the first match.",
            )
        hadm_id = df_ecg_during_hosp.hadm_id.iloc[0]
        hosp_codes = df_hosp_icd_diagnoses[
            (df_hosp_icd_diagnoses.subject_id == subject_id)
            & (df_hosp_icd_diagnoses.hadm_id == hadm_id)
        ].sort_values(by=["seq_num"]).icd_code.tolist()
        return hosp_codes, hadm_id

    def get_diagnosis_ed(subject_id, ecg_time, include_hadm=True):
        mask = (
            (df_ed_stays.subject_id == subject_id)
            & (df_ed_stays.intime < ecg_time)
            & (df_ed_stays.outtime > ecg_time)
        )
        df_ecg_during_ed = df_ed_stays[mask]
        if len(df_ecg_during_ed) == 0:
            if include_hadm:
                return [], [], np.nan, np.nan
            return [], np.nan
        if len(df_ecg_during_ed) > 1:
            print(
                "Warning: multiple ED stays matched",
                subject_id,
                ecg_time,
                "- using the first match.",
            )
        stay_id = df_ecg_during_ed.stay_id.iloc[0]
        hadm_id = df_ecg_during_ed.hadm_id.iloc[0]
        ed_codes = df_ed_diagnosis[
            (df_ed_diagnosis.subject_id == subject_id)
            & (df_ed_diagnosis.stay_id == stay_id)
        ].sort_values(by=["seq_num"]).icd_code.tolist()
        if not include_hadm:
            return ed_codes, stay_id
        hosp_codes = df_hosp_icd_diagnoses[
            (df_hosp_icd_diagnoses.subject_id == subject_id)
            & (df_hosp_icd_diagnoses.hadm_id == hadm_id)
        ].sort_values(by=["seq_num"]).icd_code.tolist()
        return ed_codes, hosp_codes, stay_id, (np.nan if hadm_id is None else hadm_id)

    rows = []
    for _, row in tqdm(df_records.iterrows(), total=len(df_records)):
        hosp_codes, hosp_hadm_id = get_diagnosis_hosp(row["subject_id"], row["ecg_time"])
        ed_codes, ed_hosp_codes, ed_stay_id, ed_hadm_id = get_diagnosis_ed(
            row["subject_id"], row["ecg_time"]
        )
        rows.append(
            {
                "file_name": row["file_name"],
                "study_id": row["study_id"],
                "subject_id": row["subject_id"],
                "ecg_time": row["ecg_time"],
                "hosp_diag_hosp": hosp_codes if hosp_codes else [],
                "hosp_hadm_id": hosp_hadm_id,
                "ed_diag_ed": ed_codes if ed_codes else [],
                "ed_diag_hosp": ed_hosp_codes if ed_hosp_codes else [],
                "ed_stay_id": ed_stay_id,
                "ed_hadm_id": ed_hadm_id,
            }
        )

    df_diag = pd.DataFrame(rows)
    df_diag.to_pickle(diag_path)
    return df_diag


def ensure_records_with_icd10(
    mimic_path: Path,
    target_path: Path,
    df_diag: pd.DataFrame,
) -> pd.DataFrame:
    icd10_path = target_path / "records_w_diag_icd10.pkl"
    if icd10_path.exists():
        return pd.read_pickle(icd10_path)

    from icdmappings import Mapper

    mapper = Mapper()
    df_hosp_icd_description = pd.read_csv(mimic_path / "hosp/d_icd_diagnoses.csv.gz")
    df_hosp_icd_description["icd10_code"] = df_hosp_icd_description.apply(
        lambda row: row["icd_code"]
        if row["icd_version"] == 10
        else mapper.map(row["icd_code"], source="icd9", target="icd10"),
        axis=1,
    )
    icd_mapping = dict(
        zip(
            df_hosp_icd_description["icd_code"], df_hosp_icd_description["icd10_code"]
        )
    )

    def map_codes(codes):
        if not isinstance(codes, (list, tuple)):
            return []
        mapped = {icd_mapping.get(code) for code in codes if code and code != "NoDx"}
        return [code for code in mapped if code]

    df_full = df_diag.copy()
    df_full["hosp_diag_hosp"] = df_full["hosp_diag_hosp"].apply(map_codes)
    df_full["ed_diag_hosp"] = df_full["ed_diag_hosp"].apply(map_codes)
    df_full["ed_diag_ed"] = df_full["ed_diag_ed"].apply(map_codes)
    df_full["all_diag_hosp"] = df_full.apply(
        lambda row: list(set(row["hosp_diag_hosp"] + row["ed_diag_hosp"])), axis=1
    )
    df_full["all_diag_all"] = df_full.apply(
        lambda row: row["all_diag_hosp"] if row["all_diag_hosp"] else row["ed_diag_ed"],
        axis=1,
    )

    df_hosp_patients = pd.read_csv(mimic_path / "hosp/patients.csv.gz")
    df_full = df_full.join(df_hosp_patients.set_index("subject_id"), on="subject_id")
    df_full["ecg_time"] = pd.to_datetime(df_full["ecg_time"])
    df_full["dod"] = pd.to_datetime(df_full["dod"])
    df_full["age"] = (
        df_full["ecg_time"].dt.year - df_full["anchor_year"] + df_full["anchor_age"]
    )

    df_full = df_full.sort_values(["subject_id", "ecg_time"], ascending=True)
    df_full["ecg_no_within_stay"] = -1
    df_full.loc[~df_full.ed_stay_id.isna(), "ecg_no_within_stay"] = (
        df_full[~df_full.ed_stay_id.isna()]
        .groupby("ed_stay_id", as_index=False)
        .cumcount()
    )
    df_full.loc[~df_full.hosp_hadm_id.isna(), "ecg_no_within_stay"] = (
        df_full[~df_full.hosp_hadm_id.isna()]
        .groupby("hosp_hadm_id", as_index=False)
        .cumcount()
    )

    df_full["ecg_taken_in_ed"] = df_full["ed_stay_id"].notnull()
    df_full["ecg_taken_in_hosp"] = df_full["hosp_hadm_id"].notnull()
    df_full["ecg_taken_in_ed_or_hosp"] = (
        df_full["ecg_taken_in_ed"] | df_full["ecg_taken_in_hosp"]
    )
    df_full["gender"] = df_full["gender"].fillna("missing_gender")

    base_columns = [
        "file_name",
        "study_id",
        "subject_id",
        "ecg_time",
        "ed_stay_id",
        "ed_hadm_id",
        "hosp_hadm_id",
        "ed_diag_ed",
        "ed_diag_hosp",
        "hosp_diag_hosp",
        "all_diag_hosp",
        "all_diag_all",
        "gender",
        "age",
        "anchor_year",
        "anchor_age",
        "dod",
        "ecg_no_within_stay",
        "ecg_taken_in_ed",
        "ecg_taken_in_hosp",
        "ecg_taken_in_ed_or_hosp",
    ]

    df_to_save = df_full[base_columns].copy()
    df_to_save.to_pickle(icd10_path)
    df_to_save.to_csv(target_path / "records_w_diag_icd10.csv", index=False)
    return df_to_save


def count_unique(df: pd.DataFrame) -> Tuple[int, int]:
    return len(df), df["subject_id"].nunique()


def subset_location_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    if subset.startswith("all"):
        return df["ecg_taken_in_ed_or_hosp"]
    if subset.startswith("ed"):
        return df["ecg_taken_in_ed"]
    if subset.startswith("hosp"):
        return df["ecg_taken_in_hosp"]
    raise ValueError(f"Unsupported subset option: {subset}")


def summarize_subset(
    df: pd.DataFrame, subset: str, label_column: str, has_statements_column: str
) -> Tuple[List[Dict[str, int]], pd.DataFrame]:
    summaries: List[Dict[str, int]] = []
    current = df.copy()

    def add_stage(label: str, data: pd.DataFrame) -> pd.DataFrame:
        records, patients = count_unique(data)
        summaries.append(
            {
                "stage": label,
                "records": records,
                "patients": patients,
            }
        )
        return data

    current = add_stage("Start (post-ICD10 mapping)", current)

    scene_mask = subset_location_mask(current, subset)
    current = add_stage(
        f"After subset '{subset}' location filter", current[scene_mask].copy()
    )

    current = add_stage(
        f"After requiring statements ({has_statements_column})",
        current[current[has_statements_column]].copy(),
    )

    if subset.endswith("first"):
        current = add_stage(
            "After first-ECG per stay filter",
            current[current["ecg_no_within_stay"] == 0].copy(),
        )

    non_empty_mask = current[label_column].map(lambda x: len(x) > 0)
    summaries.append(
        {
            "stage": f"Informational: non-empty {label_column} after label filtering",
            "records": int(non_empty_mask.sum()),
            "patients": int(current[non_empty_mask]["subject_id"].nunique()),
        }
    )

    return summaries, current


def format_summary(summary: List[Dict[str, int]]) -> List[str]:
    lines = []
    prev_records = None
    prev_patients = None
    for row in summary:
        records = row["records"]
        patients = row["patients"]
        delta_records = (
            "" if prev_records is None else f"{records - prev_records:+d} ECGs"
        )
        delta_patients = (
            "" if prev_patients is None else f"{patients - prev_patients:+d} patients"
        )
        delta_info = " / ".join(
            part for part in [delta_records, delta_patients] if part
        )
        suffix = f" ({delta_info})" if delta_info else ""
        lines.append(f"- {row['stage']}: {records} ECGs, {patients} patients{suffix}")
        prev_records = records
        prev_patients = patients
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate inclusion/exclusion counts for MIMIC-IV-ECG preprocessing."
    )
    parser.add_argument(
        "--mimic-path",
        default="./mimic",
        help="Path to MIMIC-IV directory containing hosp and ed subfolders.",
    )
    parser.add_argument(
        "--zip-path",
        default="mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip",
        help="Path to MIMIC-IV-ECG zip file.",
    )
    parser.add_argument(
        "--target-path",
        default="./",
        help="Directory where intermediary pickles/csvs are stored or will be created.",
    )
    parser.add_argument(
        "--finetune-dataset",
        default="mimic_all_all_allfirst_all_2000_5A",
        help="Finetune dataset descriptor (same format used by training scripts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mimic_path = Path(args.mimic_path)
    zip_path = Path(args.zip_path)
    target_path = Path(args.target_path)
    table_rows: List[Dict[str, object]] = []

    print("=== Loading base records ===")
    df_records = ensure_records(zip_path, target_path)
    base_counts = count_unique(df_records)
    print(f"- Raw extracted ECGs: {base_counts[0]} ({base_counts[1]} patients)")
    table_rows.append(
        {
            "step": 1,
            "criteria": "ECG 压缩包中的总记录",
            "records": base_counts[0],
            "patients": base_counts[1],
        }
    )

    print("\n=== Loading diagnoses (ICD raw) ===")
    df_diag = ensure_records_with_diag(mimic_path, target_path, df_records)
    diag_counts = count_unique(df_diag)
    print(f"- ECGs with linked stay context: {diag_counts[0]} ({diag_counts[1]} patients)")
    table_rows.append(
        {
            "step": 2,
            "criteria": "链接到有效的 MIMIC-IV 访问",
            "records": diag_counts[0],
            "patients": diag_counts[1],
        }
    )

    print("\n=== Mapping to ICD10 and enriching metadata ===")
    df_icd10 = ensure_records_with_icd10(mimic_path, target_path, df_diag)
    icd_counts = count_unique(df_icd10)
    print(f"- Post-ICD10 mapping: {icd_counts[0]} ECGs ({icd_counts[1]} patients)")

    print("\n=== Preparing label sets for reporting ===")
    df_labels, lbl_itos = prepare_mimic_ecg(
        args.finetune_dataset, target_path, df_mapped=None, df_diags=df_icd10.copy()
    )
    print(f"- Distinct ICD labels retained (>= min count): {len(lbl_itos)}")

    mask_has_diag = df_labels["has_statements_train"] | df_labels["has_statements_test"]
    df_step3 = df_labels[mask_has_diag].copy()
    step3_counts = count_unique(df_step3)
    print(
        f"- Records with ≥1 ICD statement (before location filter): "
        f"{step3_counts[0]} ECGs ({step3_counts[1]} patients)"
    )
    table_rows.append(
        {
            "step": 3,
            "criteria": "至少有一个 ICD 诊断",
            "records": step3_counts[0],
            "patients": step3_counts[1],
        }
    )

    dataset_parts = args.finetune_dataset.split("_")
    if len(dataset_parts) < 6:
        raise ValueError("finetune_dataset string must follow expected pattern.")
    subset_train = dataset_parts[1]
    subset_test = dataset_parts[3]

    print(f"\n=== Train subset '{subset_train}' ===")
    train_summary, train_df = summarize_subset(
        df_labels, subset_train, "label_train", "has_statements_train"
    )
    for line in format_summary(train_summary):
        print(line)

    print(f"\n=== Test subset '{subset_test}' ===")
    test_summary, test_df = summarize_subset(
        df_labels, subset_test, "label_test", "has_statements_test"
    )
    for line in format_summary(test_summary):
        print(line)

    subset_mask = subset_location_mask(df_step3, subset_train) | subset_location_mask(
        df_step3, subset_test
    )
    df_step4 = df_step3[subset_mask].copy()
    step4_counts = count_unique(df_step4)
    table_rows.append(
        {
            "step": 4,
            "criteria": f"匹配目标子集 ({subset_train}/{subset_test})",
            "records": step4_counts[0],
            "patients": step4_counts[1],
        }
    )

    mask_highfreq = df_step4["label_train"].map(len) > 0
    mask_highfreq |= df_step4["label_test"].map(len) > 0
    df_step5 = df_step4[mask_highfreq].copy()
    step5_counts = count_unique(df_step5)
    table_rows.append(
        {
            "step": 5,
            "criteria": "至少有一个高频 ICD",
            "records": step5_counts[0],
            "patients": step5_counts[1],
        }
    )

    combined_df = pd.concat([train_df, test_df]).drop_duplicates(subset="study_id")
    combined_counts = count_unique(combined_df)
    table_rows.append(
        {
            "step": 6,
            "criteria": "最终分析队列",
            "records": combined_counts[0],
            "patients": combined_counts[1],
        }
    )
    print(
        "\n=== Final combined dataset ===\n"
        f"- Unique ECGs after subset filters: {combined_counts[0]} "
        f"({combined_counts[1]} patients)"
    )

    print(
        "\nNote: The informational steps report how many records retain at least one "
        "label after the frequency threshold; the main inclusion/exclusion flow matches "
        "the training pipeline logic without waveform preprocessing."
    )

    print("\n=== Summary Table ===")
    print("步骤,标准,剩余 ECG 数量,剩余患者数量")
    for row in sorted(table_rows, key=lambda x: x["step"]):
        print(f"{row['step']},{row['criteria']},{row['records']},{row['patients']}")


if __name__ == "__main__":
    main()
