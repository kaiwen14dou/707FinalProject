import zipfile
import os
import wfdb
from pathlib import Path
import datetime
import pandas as pd
from tqdm.auto import tqdm

def extract_and_open_files_in_zip(zip_file_path, extension):
    """
    extracts all headers from zip file and compiles information from them into records.pkl
    """
    ecg_records = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        header_infos = [info for info in zip_ref.infolist() if info.filename.endswith(extension)]
        for file_info in tqdm(header_infos):
            # Extract the file to a temporary directory
            extract_path = file_info.filename
            os.makedirs(os.path.dirname(extract_path), exist_ok=True)
            with open(extract_path, 'wb') as extracted_file:
                extracted_file.write(zip_ref.read(file_info.filename))
            
            # Open the extracted file using wfdb
            metadata = wfdb.rdheader(extract_path[:-len(extension)])
            file = Path(extract_path[:-len(extension)])
            
            tmp={}
            tmp["file_name"]=f'{file.parent}/{file.stem}'
            tmp["study_id"]=int(file.stem)
            tmp["subject_id"]=int(file.parent.parent.stem[1:])
            if metadata.base_date is not None and metadata.base_time is not None:
                tmp['ecg_time']= datetime.datetime.combine(metadata.base_date,metadata.base_time)
            else:
                tmp['ecg_time']= pd.NaT
            ecg_records.append(tmp)                     
            
            # Delete the extracted file after use
            os.remove(extract_path)
    return pd.DataFrame(ecg_records)
    
# Path to the zip file and extension
#zip_file_path = 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'
#extension = '.hea'

# Call the function
#df=extract_and_open_files_in_zip(zip_file_path, extension)
#df.to_pickle("records.pkl")
