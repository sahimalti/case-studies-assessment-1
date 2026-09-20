# Case Studies in Data Science

This repository contains the code, datasets and outputs used for the Case Studies in Data Science assessment.

## Repository Structure

```text
case-studies-assessment-1/
│
├── data/
│   ├── heart_disease_brfss.csv
│   └── heart_disease_cleveland.csv
│
├── src/
│   ├── compare_datasets.py
│   ├── heart_disease_brfss_analysis.py
│   ├── heart_disease_cleveland_analysis.py
│   ├── model_utils.py
│   ├── task2_common.py
│   ├── task2_cv_sampling.py
|   ├── task2_extra.py
│   ├── task2_fairness.py
│   ├── task2_learning_curves.py
│   ├── task2_make_tables.py
│   └── task2_plot_learning_curves.py
│
├── outputs/
│   ├── [Task 1 outputs]
│   └── task2/
│       ├── [Task 2 results]
│       └── tables/
│
├── requirements.txt
└── run_task2.sh

Datasets

Two heart-disease datasets are used in the analysis:

Cleveland Heart Disease dataset: used for the corrected label analysis, cross-validation, sampling analysis and learning curves.
BRFSS Heart Disease dataset: used for cross-validation, learning curves and the Fairlearn fairness analysis.
Analysis
Task 1

The original analysis includes:

Dataset exploration and comparison
Heart-disease classification
Logistic Regression and Random Forest models
Model evaluation
Feature and correlation analysis
Task 2

The extended analysis examines the reliability and fairness of the Task 1 results.

It includes:

-Repeated stratified cross-validation
-Nested cross-validation
-Repeated hold-out sampling
-Learning curves across different training set sizes
-Fairlearn group-level fairness analysis
-Recall and false-positive-rate comparisons
-Age-based fairness mitigation
- Wilson confidence intervals for group-level recall
- Corrected resampled t-tests for model comparison
- Imbalance-handling strategies applied inside cross-validation folds
- Supporting tables and figures


The Task 2 analysis also includes corrected handling of the Cleveland dataset label and encoded missing values.

Requirements

The analysis was developed using Python 3.12.

Install the required packages using: pip install -r requirements.txt

The main packages used include:

pandas
NumPy
scikit-learn
Fairlearn
matplotlib
seaborn
Running the Analysis

Run commands from the root directory of the repository.

To run the Task 1 reproduction: python src/task2_cv_sampling.py task1

For the Task 2 cross-validation and sampling analysis:
python src/task2_cv_sampling.py repeated
python src/task2_cv_sampling.py nested
python src/task2_cv_sampling.py holdout
python src/task2_cv_sampling.py brfss
python src/task2_cv_sampling.py collect

To generate the learning-curve results:

python src/task2_learning_curves.py cleveland
python src/task2_learning_curves.py brfss
python src/task2_plot_learning_curves.py

To run the fairness analysis: python src/task2_fairness.py
To generate the LaTeX tables: To generate the LaTeX tables:
The generated Task 2 results are saved under: outputs/task2/

Reproducibility

The repository contains the datasets, source code and generated outputs used for the analysis reported in the assessment.

The Task 2 scripts are designed to run from the repository root and use the datasets in the data/ directory. The outputs in outputs/task2/ provide the results and supporting figures used in the assessment.

Additional Task 2 analyses

`task2_extra.py` contains the supporting analyses for:

- 95% Wilson intervals for group-level recall
- Corrected resampled t-tests comparing Logistic Regression and Random Forest
- BRFSS imbalance strategies, including class weighting, random undersampling and SMOTE applied inside cross-validation folds

Notes

The analysis was conducted locally using the Python environment specified in requirements.txt. The final interpretation of the results is based on the analysis and outputs reported in the assessment.
