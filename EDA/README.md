# ECG EDA Toolkit

`run_eda.py` keeps all exploratory analysis isolated from the preprocessing pipeline. It focuses on the standard checks described in recent MIMIC-IV-ECG papers:

1. **Demographics:** age histograms + gender bars that compare the *full* ECG corpus with the *analysis* cohort (first ECG per subject).
2. **Label engineering:** ICD-10 groupings for MI, arrhythmia/AFib, conduction disorders, heart failure, hypertension, mitral stenosis, and cardiomyopathy. Optionally reuse your custom label (e.g., `my_new_label_column`).
3. **Label prevalence:** prevalence tables/plots plus age- and gender-specific risk heatmaps, label co-occurrence matrices, and UpSet plots (if `upsetplot` is installed).
4. **Signal-level analysis:** optional extraction of heart rate, QRS/QT/PR intervals via `wfdb` + `neurokit2`, followed by violin/box plots and PCA/t-SNE. Falls back gracefully when WFDB files or dependencies are absent.
5. **12-lead visualisation:** randomly sampled ECGs (first-ECG cohort) saved as multi-panel PNGs to verify morphology vs. labels.

## Dependencies

Install the plotting + signal stack inside your preferred environment:

```bash
pip install pandas numpy matplotlib seaborn wfdb neurokit2 scikit-learn upsetplot
```

`wfdb` and `neurokit2` are only required for signal features/plots; the rest of the EDA runs without them.

## Usage

```bash
python EDA/run_eda.py \
  --metadata-path preprocess/src/records_w_diag_icd10_folds.pkl \
  --subject-column subject_id \
  --ecg-time-column ecg_time \
  --file-column file_name \
  --wfdb-root /path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
  --output-dir EDA/outputs \
  --max-feature-records 200 \
  --max-plot-records 12
```

- Optionally add `--label-column <your_binary_label>` if you already created a custom target during preprocessing.
- Set `--wfdb-root` to the directory that contains the `files/pXXXX/...` tree so that `wfdb.rdrecord` can find each `<record>.hea/.dat`.
- Omit `--wfdb-root` (or leave empty) if the raw ECG waveforms are not available; the script will still produce all demographic + label analyses.
- Adjust `--max-feature-records` and `--max-plot-records` if you want deeper or faster signal analyses.

All tables/figures land under `EDA/outputs`, ready to be copied into reports or notebooks.
