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

DATASETS

Two heart-disease datasets are used in the analysis:

-Cleveland Heart Disease dataset: used for the corrected label analysis, cross-validation, sampling analysis and learning curves.

-BRFSS Heart Disease dataset: used for cross-validation, learning curves and the Fairlearn fairness analysis.

ANALYSIS
Task 1

The original analysis includes:

-Dataset exploration and comparison
-Heart-disease classification
-Logistic Regression and Random Forest models
-Model evaluation
-Feature and correlation analysis

Task 2
Task 2 extends the original analysis by looking more closely at model reliability, performance variation and fairness.

The analysis includes:

-Repeated stratified cross-validation
-Nested cross-validation
-Repeated hold-out sampling
-Learning curves across different training set sizes
-Fairlearn group-level fairness analysis
-Recall and false-positive-rate comparisons
-Age-based fairness mitigation
-Wilson confidence intervals for group-level recall
-Corrected resampled t-tests for model comparison
-Imbalance-handling strategies applied inside cross-validation folds
-Supporting tables and figures


The Task 2 analysis also includes corrected handling of the Cleveland dataset label and encoded missing values.

REQUIREMENTS

The analysis was developed using Python 3.12.

Install the required packages using: pip install -r requirements.txt

The main packages used include:

-pandas
-NumPy
-SciPy
-scikit-learn
-Fairlearn
-matplotlib
-imbalanced-learn
-seaborn

RUNNING THE ANALYSIS

Run commands from the root directory of the repository.

Task 1 reproduction: python src/task2_cv_sampling.py task1

Task 2 cross-validation and sampling analysis:
python src/task2_cv_sampling.py repeated
python src/task2_cv_sampling.py nested
python src/task2_cv_sampling.py holdout
python src/task2_cv_sampling.py brfss
python src/task2_cv_sampling.py collect

Learning-curve results:
python src/task2_learning_curves.py cleveland
python src/task2_learning_curves.py brfss
python src/task2_plot_learning_curves.py

Fairness Analysis:
python src/task2_fairness.py

Additional Task 2 analyses

The task2_extra.py script contains additional analyses used to support the Task 2 results.

It includes:

-95% Wilson intervals for group-level recall
-Corrected resampled t-tests comparing Logistic Regression and Random Forest
-BRFSS imbalance strategies, including class weighting, random undersampling and SMOTE within cross-validation

Run all three analyses with: python src/task2_extra.py

Individual analyses can also be run with:
python src/task2_extra.py intervals
python src/task2_extra.py ttests
python src/task2_extra.py sampling

LaTeX tables

To generate the LaTeX tables:
python src/task2_make_tables.py

The generated Task 2 results are saved under:
outputs/task2/

Reproducibility

The repository contains the datasets, source code and outputs used for the analysis reported in the assessment.

The Task 2 scripts are designed to run from the repository root and use the datasets stored in the data/ directory. The results and supporting figures are saved in outputs/task2/.

The main analyses use random seed 42. The hold-out sensitivity analysis uses seeds 0–99.

Notes

The analysis was conducted locally using the Python environment specified in requirements.txt. The results and interpretations reported in the assessment are based on the analysis outputs generated from this repository.
