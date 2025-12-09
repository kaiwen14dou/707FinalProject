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

Our project evaluates ECG arrhythmia classification performance under demographic subgroups using two modeling approaches. The goal is to assess subgroup disparities, distribution shift, and whether retraining on domain‐specific data improves fairness and accuracy.

### 1. Predict-Only Baseline (Original CODE Model)

We first apply the original CODE model and pretrained weights directly to our dataset.  
This serves as an out-of-distribution baseline for evaluating subgroup disparities such as male vs female performance, AUROC differences, and calibration gaps.  
Predictions and subgroup metrics are generated using the `predict_with_CODE.py` script.

### 2. Retrained Model on MIMIC-IV ECG Data

To improve performance under our institutional distribution, we retrain a CODE-style model using standardized ECG inputs and class-weighted loss to address substantial label imbalance.  
Training is performed using the `train_fairness_tf.py` script with early stopping, learning-rate scheduling, and per-sample normalization.  
The resulting model achieves better alignment with MIMIC-IV and more balanced subgroup metrics.

### 3. Threshold Optimization

We compute an optimal decision threshold for each class by sweeping thresholds from 0.01 to 0.99 and selecting the value that maximizes F1.  
This produces the file `per_class_thresholds.csv`, which is used for all downstream subgroup evaluations.

### 4. Subgroup Analysis (Male vs Female)

Using both the original and retrained models, we compute AUROC, sensitivity, specificity, and F1 separately for male and female patients.  
Additional confusion matrices and ROC curves are produced to visualize subgroup disparities, using scripts in the `graph/` folder.

### 5. Exported Predictions for Analysis

To support reproducibility and external analysis, we export the final test-set predictions, thresholds, labels, and metadata using `export_test_predictions.py`.  
The output includes probability matrices, binary predictions, true labels, and patient metadata.


