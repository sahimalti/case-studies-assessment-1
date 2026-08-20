"""
heart_disease_cleveland_analysis.py

Dataset 1: Heart Disease (Cleveland), UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/45/heart+disease

Expects a CSV at data/heart_disease_cleveland.csv with the standard
14-column Cleveland schema:
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
    oldpeak, slope, ca, thal, target

`target` is 0 for no heart disease, and 1-4 for increasing severity in
the original UCI data (some Kaggle mirrors already binarise it). This
script binarises it to 0/1 if it finds values above 1.

Run from the project root:
    python src/heart_disease_cleveland_analysis.py
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

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart_disease_cleveland.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DATASET_NAME = "Heart Disease (Cleveland)"


def load_and_clean(path):
    df = pd.read_csv(path)

    # Some mirrors use different column casing / a 'num' column instead
    # of 'target' — normalise here if needed.
    df.columns = [c.strip().lower() for c in df.columns]
    if "num" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"num": "target"})

    # Binarise target: 0 = no disease, 1 = disease (any severity)
    df["target"] = (df["target"] > 0).astype(int)

    # ca and thal sometimes contain '?' for missing values in the raw
    # UCI files — coerce to numeric and impute with the median.
    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Drop any remaining rows with missing values in other columns
    df = df.dropna()

    return df


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_and_clean(DATA_PATH)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardise numeric features (helps Logistic Regression particularly)
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
            tune=True, cv_folds=5,
        )
        all_results.append(results)

        plot_confusion_matrix(
            results["confusion_matrix"],
            model_name,
            DATASET_NAME,
            OUTPUT_DIR / f"cleveland_{model_name}_confusion_matrix.png",
        )

        importance = get_feature_importance(model, feature_cols, model_name)
        importance.to_csv(OUTPUT_DIR / f"cleveland_{model_name}_feature_importance.csv")
        print(f"\nTop 5 features — {model_name} on {DATASET_NAME}:")
        print(importance.head(5))

    print(f"\n=== Results summary: {DATASET_NAME} ===")
    results_df = print_results_table(all_results)
    results_df.to_csv(OUTPUT_DIR / "cleveland_results_summary.csv", index=False)

    return all_results


if __name__ == "__main__":
    run()
