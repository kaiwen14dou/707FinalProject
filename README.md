# 707FinalProject
ECGModelSubgroupPerformance


| **Week**   | **Date Range**      | **Milestones / Tasks**                                            |
| ---------- | ------------------- | ----------------------------------------------------------------- |
| **Week 1** | **Oct 12 – Oct 17** | - Finalize project proposal<br>- Complete data preprocessing      |
| **Week 2** | **Oct 20 – Oct 25** | - Determine models to be tested<br>- Begin model application      |
| **Week 3** | **Oct 28 – Nov 1**  | - Conduct model verification and refinement                       |
| **Week 4** | **Nov 4 – Nov 9**   | - Verify and validate results<br>- Begin presentation preparation |
| **Week 5** | **Nov 11 – TBD**    | - Final presentation rehearsal and delivery                       |

## Aims

Aim 1: Benchmark subgroup performance of state-of-the-art ECG models on arrhythmia.
We will evaluate how current deep learning models trained on large ECG datasets perform across demographic subgroups (e.g., sex, age, race/ethnicity) and institutional cohorts. Performance will be assessed using established measures of discrimination (AUROC, AUPRC, sensitivity, specificity) and calibration (Brier score, subgroup calibration error, calibration curves). This aim will establish a systematic baseline of subgroup disparities in ECG modeling.

Aim 2: We will introduce new quantitative measures that have not yet been applied in the medical ECG domain. These metrics will provide a more nuanced understanding of discrimination and calibration disparities, particularly under subgroup imbalance and distribution shift. Comparisons with established metrics will highlight advantages and limitations of each approach.

Aim 3: We will evaluate whether advanced representation learning methods (e.g., self-supervised contrastive pretraining, multi-task learning, domain adversarial training) can improve downstream ECG classification performance while reducing subgroup disparities. Models will be trained with fairness-constrained objectives and tested under institutional distribution shift. This aim will generate actionable insights into how representational strategies contribute to equitable and reliable ECG modeling.

## Data 


## Model Pipeline Overview

Our project evaluates ECG arrhythmia classification performance under demographic subgroups using two types of models:

1. Predict-Only Baseline (Original CODE Model)

We first apply the original CODE model and its published weights directly to our MIMIC-IV ECG dataset.
This provides a distribution-shift baseline, allowing us to assess how well the original model generalizes across sex subgroups.

Script: predict_with_CODE.py

Output: predicted probabilities, binary predictions, subgroup performance tables, and AUROC curves.

2. Retrained Model on MIMIC-IV ECGs

We retrain a CODE-style CNN on our own dataset to better match the MIMIC-IV population and reduce subgroup disparities.

Key features:

Same architecture as CODE

Per-sample ECG standardization

Class-weighted BCE loss to address strong label imbalance

Early stopping and model checkpointing

Final model: model_fairness_weighted_best.h5

3. Threshold Optimization and Evaluation

Because ECG labels are highly imbalanced, each class receives its own F1-optimal decision threshold.

Script: evaluate_test.py

Output:

per_class_thresholds.csv

final test predictions (probabilities + binary)

sex-specific prediction files

4. Subgroup Analysis (Sex-based)

We compute per-class performance and sex-stratified AUROC, sensitivity, specificity, and F1.

Scripts:

make_confusion_tables.py

plot_auroc_female_male.py

Outputs include:

per_class_metrics_all.csv

per_class_metrics_female.csv

per_class_metrics_male.csv

Female vs Male ROC curve panels

This full pipeline allows us to compare:

Predict-only baseline performance, and

Retrained model performance,
along with subgroup fairness metrics.
