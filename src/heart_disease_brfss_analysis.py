"""
heart_disease_brfss_analysis.py

Dataset 2: Heart Disease Health Indicators, derived from the CDC's
Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey.
https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

Expects a CSV at data/heart_disease_brfss.csv with (at minimum) these
columns from the public dataset:
    HeartDiseaseorAttack, HighBP, HighChol, BMI, Smoker, Stroke,
    Diabetes, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
    AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth,
    DiffWalk, Sex, Age, Education, Income

The target column is HeartDiseaseorAttack (0/1). This dataset is
large (253,680 rows) and has a strong class imbalance (roughly
9:1 negative:positive), which is why stratified sampling and
class-weighted models are used throughout.

Run from the project root:
    python src/heart_disease_brfss_analysis.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model_utils import (
    train_and_evaluate,
    plot_confusion_matrix,
    get_feature_importance,
    print_results_table,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart_disease_brfss.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DATASET_NAME = "Heart Disease Health Indicators (BRFSS)"
TARGET_COL = "HeartDiseaseorAttack"


def load_and_clean(path):
    df = pd.read_csv(path)

    # Normalise column names in case of casing differences across mirrors
    df.columns = [c.strip() for c in df.columns]

    # This dataset is generally clean (pre-processed by the Kaggle
    # uploader from the raw BRFSS survey), but guard against missing
    # values regardless.
    df = df.dropna()

    # Ensure target is integer 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_and_clean(DATA_PATH)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    # This dataset is large, so a smaller test proportion still gives
    # a substantial held-out set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    all_results = []
    for model_name in ["random_forest", "logistic_regression"]:
        model, results = train_and_evaluate(
            X_train_scaled, X_test_scaled, y_train, y_test, model_name, DATASET_NAME,
            tune=True, cv_folds=3,
        )
        all_results.append(results)

        plot_confusion_matrix(
            results["confusion_matrix"],
            model_name,
            DATASET_NAME,
            OUTPUT_DIR / f"brfss_{model_name}_confusion_matrix.png",
        )

        importance = get_feature_importance(model, feature_cols, model_name)
        importance.to_csv(OUTPUT_DIR / f"brfss_{model_name}_feature_importance.csv")
        print(f"\nTop 5 features — {model_name} on {DATASET_NAME}:")
        print(importance.head(5))

    print(f"\n=== Results summary: {DATASET_NAME} ===")
    results_df = print_results_table(all_results)
    results_df.to_csv(OUTPUT_DIR / "brfss_results_summary.csv", index=False)

    return all_results


if __name__ == "__main__":
    run()
