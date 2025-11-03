#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from extract_headers import extract_and_open_files_in_zip

#requires icd-mappings
from icdmappings import Mapper
from ecg_utils import prepare_mimicecg
from clinical_ts.timeseries_utils import reformat_as_memmap
from utils.stratify import stratified_subsets
from mimic_ecg_preprocessing import prepare_mimic_ecg


def main():
    parser = argparse.ArgumentParser(description='A script to extract two paths from the command line.')
    
    # Add arguments for the two paths
    parser.add_argument('--mimic-path', help='path to mimic iv folder with subfolders hosp and ed',default="./mimic")
    parser.add_argument('--zip-path', help='path to mimic ecg zip',default="mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip")
    parser.add_argument('--target-path', help='desired output path',default="./")
    
    # you have to explicitly pass this argument to convert to numpy and memmapp
    parser.add_argument('--numpy-memmap', type=bool, default=True, help='Enable or disable numpy memmap (for fast access to waveforms)')
    parser.add_argument('--skip-folds', action='store_true', help='Skip stratified fold generation for faster preprocessing.')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Access the paths
    mimic_path = Path(args.mimic_path)
    zip_file_path = Path(args.zip_path)
    target_path = Path(args.target_path)
    
    numpy_memmap = args.numpy_memmap
    skip_folds = args.skip_folds
    
    print("mimic_path",mimic_path)
    print("zip_file_path",zip_file_path)
    print("target_path",target_path)
    ##################################################################################################
    print("Step 1: Extract available records from mimic-ecg-zip-path to create records.pkl")
    if((target_path/"records.pkl").exists()):
        print("Skipping: using existing records.pkl")
        df = pd.read_pickle(target_path/"records.pkl")
    else:
        print("Creating records.pkl")
        df = extract_and_open_files_in_zip(zip_file_path, ".hea")
        target_path.mkdir(parents=True, exist_ok=True)
        df.to_pickle(target_path/"records.pkl")
    
    #################################################################################################
    icd_mapping = None
    print("Step 2: Extract diagnoses for records in raw format to create records_w_diag.pkl")
    if((target_path/"records_w_diag.pkl").exists()):
        print("Skipping: using existing records_w_diag.pkl")
        df_full = pd.read_pickle(target_path/"records_w_diag.pkl")
    else:
        mapper = Mapper()
        
        df_hosp_icd_description = pd.read_csv(mimic_path/"hosp/d_icd_diagnoses.csv.gz")
        df_hosp_icd_diagnoses = pd.read_csv(mimic_path/"hosp/diagnoses_icd.csv.gz")
        df_hosp_admissions = pd.read_csv(mimic_path/"hosp/admissions.csv.gz")
        df_hosp_admissions["admittime"]=pd.to_datetime(df_hosp_admissions["admittime"])
        df_hosp_admissions["dischtime"]=pd.to_datetime(df_hosp_admissions["dischtime"])
        df_hosp_admissions["deathtime"]=pd.to_datetime(df_hosp_admissions["deathtime"])
        
        df_hosp_icd_description["icd10_code"] = df_hosp_icd_description.apply(
            lambda row: row["icd_code"] if row["icd_version"] == 10 else mapper.map(row["icd_code"], source="icd9", target="icd10"),
            axis=1
        )
        icd_mapping = {ic: ic10 for ic, ic10 in zip(df_hosp_icd_description["icd_code"], df_hosp_icd_description["icd10_code"])}
        df_ed_stays = pd.read_csv(mimic_path/"ed/edstays.csv.gz")
        df_ed_stays["intime"]=pd.to_datetime(df_ed_stays["intime"])
        df_ed_stays["outtime"]=pd.to_datetime(df_ed_stays["outtime"])
        df_ed_diagnosis = pd.read_csv(mimic_path/"ed/diagnosis.csv.gz")

        def get_diagnosis_hosp(subject_id, ecg_time):
            df_ecg_during_hosp= df_hosp_admissions[(df_hosp_admissions.subject_id==subject_id) & (df_hosp_admissions.admittime<ecg_time) & ((df_hosp_admissions.dischtime>ecg_time)|(df_hosp_admissions.deathtime>ecg_time))]
            if(len(df_ecg_during_hosp)==0):
                return [],np.nan
            else:
                if(len(df_ecg_during_hosp)>1):
                    print("Error in get_diagnosis_hosp: multiple entries for",subject_id,ecg_time,". Considering only the first one.")
                hadm_id=df_ecg_during_hosp.hadm_id.iloc[0]
                return list(df_hosp_icd_diagnoses[(df_hosp_icd_diagnoses.subject_id==subject_id)&(df_hosp_icd_diagnoses.hadm_id==hadm_id)].sort_values(by=['seq_num']).icd_code), hadm_id #diags_hosp, hadm_id

        def get_diagnosis_ed(subject_id, ecg_time,also_hosp_diag=True):
            df_ecg_during_ed = df_ed_stays[(df_ed_stays.subject_id==subject_id) & (df_ed_stays.intime<ecg_time) & (df_ed_stays.outtime>ecg_time)]
            if(len(df_ecg_during_ed)==0):
                return ([],[],np.nan,np.nan) if also_hosp_diag else ([],np.nan)
            else:
                if(len(df_ecg_during_ed)>1):
                    print("Error in get_diagnosis_ed: multiple entries for",subject_id,ecg_time,". Considering only the first one.")
                stay_id=df_ecg_during_ed.stay_id.iloc[0]
                hadm_id=df_ecg_during_ed.hadm_id.iloc[0]#potentially none
                res=list(df_ed_diagnosis[(df_ed_diagnosis.subject_id==subject_id)&(df_ed_diagnosis.stay_id==stay_id)].sort_values(by=['seq_num']).icd_code)
                if(also_hosp_diag):
                    res2=list(df_hosp_icd_diagnoses[(df_hosp_icd_diagnoses.subject_id==subject_id)&(df_hosp_icd_diagnoses.hadm_id==hadm_id)].sort_values(by=['seq_num']).icd_code)
                    return res, res2, stay_id, (np.nan if hadm_id is None else hadm_id) #diags_ed, diags_hosp, stay_id, hadm_id
                else:
                    return res, stay_id #diags_ed, stay_id


        result=[]

        for id,row in tqdm(df.iterrows(),total=len(df)):
            tmp={}
            tmp["file_name"]=row["file_name"]
            tmp["study_id"]=row["study_id"]
            tmp["subject_id"]=row["subject_id"]
            tmp["ecg_time"]=row["ecg_time"]
            hosp_diag_hosp, hosp_hadm_id =get_diagnosis_hosp(row["subject_id"], row["ecg_time"])
            tmp["hosp_diag_hosp"] = hosp_diag_hosp
            tmp["hosp_hadm_id"] =hosp_hadm_id
            ed_diag_ed,ed_diag_hosp,ed_stay_id,ed_hadm_id = get_diagnosis_ed(row["subject_id"], row["ecg_time"])
            tmp["ed_diag_ed"]=ed_diag_ed
            tmp["ed_diag_hosp"]=ed_diag_hosp
            tmp["ed_stay_id"]=ed_stay_id
            tmp["ed_hadm_id"]=ed_hadm_id
            result.append(tmp)
        df_full = pd.DataFrame(result)
        df_full["hosp_diag_hosp"]=df_full["hosp_diag_hosp"].apply(lambda x: [] if x is None else x)
        df_full.to_pickle(target_path/"records_w_diag.pkl")
        
    #################################################################################################
    print("Step 3: Map everything to ICD10 and enrich with more metadata to create output records_w_diag_icd10.pkl")
    icd10_output_path = target_path/"records_w_diag_icd10.pkl"
    icd10_csv_path = target_path/"records_w_diag_icd10.csv"
    regenerate_step3 = not icd10_output_path.exists()
    if icd10_output_path.exists():
        df_cached = pd.read_pickle(icd10_output_path)
        has_fold_columns = any(col in df_cached.columns for col in ("fold", "strat_fold"))
        if skip_folds:
            if has_fold_columns:
                print("Skipping heavy regeneration: cached records include folds but --skip-folds requested. Removing fold columns.")
                df_cached = df_cached.drop(columns=[c for c in ("fold", "strat_fold") if c in df_cached.columns])
                if "gender" in df_cached.columns:
                    df_cached["gender"] = df_cached["gender"].fillna("missing_gender")
                df_cached.to_pickle(icd10_output_path)
                df_cached.to_csv(icd10_csv_path, index=False)
            else:
                print("Skipping Step 3: using cached records_w_diag_icd10.pkl")
            regenerate_step3 = False
        else:
            if has_fold_columns:
                print("Skipping Step 3: using cached records_w_diag_icd10.pkl")
                regenerate_step3 = False
            else:
                print("Cached records_w_diag_icd10.pkl do not contain folds; regenerating to add them.")

    if regenerate_step3:
        if icd_mapping is None:
            mapper = Mapper()
            df_hosp_icd_description = pd.read_csv(mimic_path/"hosp/d_icd_diagnoses.csv.gz")
            df_hosp_icd_description["icd10_code"] = df_hosp_icd_description.apply(
                lambda row: row["icd_code"] if row["icd_version"] == 10 else mapper.map(row["icd_code"], source="icd9", target="icd10"),
                axis=1
            )
            icd_mapping = {ic: ic10 for ic, ic10 in zip(df_hosp_icd_description["icd_code"], df_hosp_icd_description["icd10_code"])}

        def map_codes(codes):
            if not isinstance(codes, (list, tuple)):
                return []
            mapped = [icd_mapping.get(code) for code in codes if code and code != "NoDx"]
            return list(set([code for code in mapped if code]))

        df_full["hosp_diag_hosp"]=df_full["hosp_diag_hosp"].apply(map_codes)
        df_full["ed_diag_hosp"]=df_full["ed_diag_hosp"].apply(map_codes)
        df_full["ed_diag_ed"]=df_full["ed_diag_ed"].apply(map_codes)
        #ed or hosp ecgs with discharge diagnosis
        df_full["all_diag_hosp"]=df_full.apply(lambda row: list(set(row["hosp_diag_hosp"]+row["ed_diag_hosp"])),axis=1)
        # 'all_diag_all': 'all_diag_hosp' if available otherwise 'ed_diag_ed'
        df_full['all_diag_all'] = df_full.apply(lambda row: row['all_diag_hosp'] if row['all_diag_hosp'] else row['ed_diag_ed'],axis=1)
        

        #add demographics
        df_hosp_patients = pd.read_csv(mimic_path/"hosp/patients.csv.gz")
        df_full=df_full.join(df_hosp_patients.set_index("subject_id"),on="subject_id")
        df_full["ecg_time"] = pd.to_datetime(df_full["ecg_time"])
        df_full["dod"] = pd.to_datetime(df_full["dod"])
        df_full["age"]=df_full["ecg_time"].dt.year - df_full["anchor_year"] + df_full["anchor_age"]

        #add ecg number within stay
        df_full["ecg_no_within_stay"]=-1
        df_full=df_full.sort_values(["subject_id","ecg_time"],ascending=True)

        df_full.loc[~df_full.ed_stay_id.isna(),"ecg_no_within_stay"]=df_full[~df_full.ed_stay_id.isna()].groupby("ed_stay_id",as_index=False).cumcount()
        df_full.loc[~df_full.hosp_hadm_id.isna(),"ecg_no_within_stay"]=df_full[~df_full.hosp_hadm_id.isna()].groupby("hosp_hadm_id",as_index=False).cumcount()

        df_full["ecg_taken_in_ed"]=df_full["ed_stay_id"].notnull()
        df_full["ecg_taken_in_hosp"]=df_full["hosp_hadm_id"].notnull()
        df_full["ecg_taken_in_ed_or_hosp"]=(df_full["ecg_taken_in_ed"]|df_full["ecg_taken_in_hosp"])
        

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

        if skip_folds:
            df_full["gender"] = df_full["gender"].fillna("missing_gender")
            df_to_save = df_full[base_columns].copy()
            print("Skipping stratified fold generation (requested). You can create folds later from the exported data.")
        else:
            # Fols used in the manuscript experiments, use them for reproducibility.
            df_full["fold"] = np.load('utils/folds.npy')
            
            # STRATIFIED FOLDS based on'all_diag'. folds not used in experiments, but provided for convenience
            df_full, _ = prepare_mimic_ecg('mimic_all_all_allfirst_all_2000_5A',target_path,df_mapped=None,df_diags=df_full)
            df_full['label_train'] = df_full['label_train'].apply(lambda x: x if x else ['outpatient'])
            df_full.rename(columns={'label_train':'label_strat_all2all'}, inplace=True)
            age_bins = pd.qcut(df_full['age'], q=4, duplicates='drop')
            unique_intervals = age_bins.cat.categories
            bin_labels = {interval: f'{interval.left}-{interval.right}' for interval in unique_intervals}
            df_full['age_bin'] = age_bins.map(bin_labels)
            df_full['age_bin'] = df_full['age_bin'].astype('category')
            df_full['age_bin'] = df_full['age_bin'].cat.add_categories(['missing_age']).fillna('missing_age')
            df_full['gender'] = df_full['gender'].fillna('missing_gender')
            
            df_full['merged_strat'] = df_full.apply(lambda row: row['label_strat_all2all'] + [row['age_bin'], row['gender']], axis=1)
            
            col_label = 'merged_strat'
            col_group = 'subject_id'
            
            res = stratified_subsets(df_full,
                           col_label,
                           [0.05]*20,
                           col_group=col_group,
                           label_multi_hot=False,
                           random_seed=42)
            
            df_full['strat_fold'] = res
            df_to_save = df_full[base_columns + ["fold", "strat_fold"]].copy()

        df_to_save.to_pickle(icd10_output_path)
        df_to_save.to_csv(icd10_csv_path, index=False)
        
        
    if numpy_memmap:
        
    
        processed_folder = target_path/"processed"
        processed_df_path = processed_folder/"df.pkl"
        recreate_data = not processed_df_path.exists()
        if recreate_data:
            print("Step 4: Convert signals into numpy in  target-path/processed")
        else:
            print("Skipping Step 4: using cached numpy arrays in target-path/processed")
        processed_folder.mkdir(parents=True, exist_ok=True)
        df,_,_,_=prepare_mimicecg(zip_file_path, target_folder=processed_folder, recreate_data=recreate_data)

        memmap_folder = target_path/"memmap"
        memmap_folder.mkdir(parents=True, exist_ok=True)
        memmap_data_path = memmap_folder/"memmap.npy"
        memmap_df_path = memmap_folder/"df_memmap.pkl"
        if memmap_data_path.exists() and memmap_df_path.exists():
            print("Skipping Step 5: memmap already available in target-path/memmap")
        else:
            print("Step 5: Reformat as memmap for fast access")
            reformat_as_memmap(df, memmap_data_path, data_folder=processed_folder, annotation=False, max_len=0, delete_npys=True,col_data="data",col_lbl=None, batch_length=0, skip_export_signals=False)
    
    
    print("Done.")
        
    

if __name__ == '__main__':
    main()
