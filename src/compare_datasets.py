"""
compare_datasets.py

Runs both dataset analyses and produces the cross-dataset comparison
that is the core "insights" deliverable for Part 1.3: does the
blood-pressure-related signal carry similar predictive weight in
both the small clinical dataset (Cleveland) and the large,
self-reported survey dataset (BRFSS)?

Run from the project root:
    python src/compare_datasets.py
"""

import pandas as pd
from pathlib import Path

import heart_disease_cleveland_analysis as cleveland
import heart_disease_brfss_analysis as brfss
from model_utils import (
    print_results_table,
    plot_correlation_matrix,
    plot_combined_metrics_comparison,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DATA_DIR = Path(_file_).resolve().parent.parent / "data"

def compare_risk_factor_importance():
    """
    Load the saved feature importance CSVs from both datasets and
    report where BOTH blood pressure and diabetes rank in each,
    since both are established heart-disease risk factors and the
    goal is to see whether their predictive weight is consistent
    across two differently-collected datasets.
    """
    print("\n" + "=" * 60)
    print("RISK FACTOR IMPORTANCE — CROSS-DATASET COMPARISON")
    print("=" * 60)

    # Cleveland uses fbs (fasting blood sugar > 120 mg/dl) as the
    # standard clinical proxy for diabetes/prediabetes, since the
    # dataset has no direct diabetes diagnosis column.
    risk_factors = {
        "Blood pressure": {"cleveland": "trestbps", "brfss": "HighBP"},
        "Diabetes":        {"cleveland": "fbs",       "brfss": "Diabetes"},
    }

    for model_name in ["random_forest", "logistic_regression"]:
        cleveland_imp = pd.read_csv(
            OUTPUT_DIR / f"cleveland_{model_name}_feature_importance.csv",
            index_col=0,
        ).iloc[:, 0]
        brfss_imp = pd.read_csv(
            OUTPUT_DIR / f"brfss_{model_name}_feature_importance.csv",
            index_col=0,
        ).iloc[:, 0]

        print(f"\n{model_name.replace('_', ' ').title()}:")
        for factor_label, cols in risk_factors.items():
            cleveland_rank = list(cleveland_imp.index).index(cols["cleveland"]) + 1
            brfss_rank = list(brfss_imp.index).index(cols["brfss"]) + 1
            print(
                f"  {factor_label:16s} — Cleveland: #{cleveland_rank} of "
                f"{len(cleveland_imp)}   |   BRFSS: #{brfss_rank} of {len(brfss_imp)}"
            )
def generate_correlation_matrices():
    print("\n" + "=" * 60)
    print("GENERATING CORRELATION MATRICES")
    print("=" * 60)

    df1 = cleveland.load_and_clean(DATA_DIR / "heart_disease_cleveland.csv")
    corr1 = plot_correlation_matrix(
        df1, "target", "Heart Disease (Cleveland)",
        OUTPUT_DIR / "cleveland_correlation_matrix.png",
    )
    print("\nCleveland correlation with target (top 5):")
    print(corr1.head(5))

    df2 = brfss.load_and_clean(DATA_DIR / "heart_disease_brfss.csv")
    corr2 = plot_correlation_matrix(
        df2, "HeartDiseaseorAttack", "Heart Disease Health Indicators (BRFSS)",
        OUTPUT_DIR / "brfss_correlation_matrix.png",
    )
    print("\nBRFSS correlation with target (top 5):")
    print(corr2.head(5))

def main():
    print("Running Dataset 1: Heart Disease (Cleveland)...")
    cleveland_results = cleveland.run()

    print("\n\nRunning Dataset 2: Heart Disease Health Indicators (BRFSS)...")
    brfss_results = brfss.run()

    print("\n\n" + "=" * 60)
    print("COMBINED RESULTS — ALL FOUR MODEL RUNS")
    print("=" * 60)
    all_results = cleveland_results + brfss_results
    combined_df = print_results_table(all_results)
    combined_df.to_csv(OUTPUT_DIR / "combined_results_summary.csv", index=False)
    plot_combined_metrics_comparison(
        all_results, OUTPUT_DIR / "combined_metrics_comparison.png"
    )

    compare_risk_factor_importance()
generate_correlation_matrices()
    print(
        "\nDone. See outputs/ for confusion matrix plots, feature "
        "importance CSVs, and results summaries to write up in "
        "Part 1.3 (Insights and Comparison)."
    )


if __name__ == "__main__":
    main()
