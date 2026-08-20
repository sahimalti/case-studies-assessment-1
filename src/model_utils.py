"""
model_utils.py

Shared helper functions for training and evaluating Random Forest and
Logistic Regression on a binary health-risk classification task.

Used identically by heart_disease_cleveland_analysis.py and
heart_disease_brfss_analysis.py so that the two datasets are processed
through directly comparable modelling steps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def tune_hyperparameters(X_train, y_train, model_name, cv_folds=5, random_state=42,
                          max_tune_samples=20000):
    """
    Run GridSearchCV to find the best hyperparameters for the given model,
    scored on F1 rather than accuracy, since accuracy is misleading on
    imbalanced data (see README / report discussion).

    For large training sets (e.g. BRFSS, ~200K+ rows), searching the full
    grid on the full data is prohibitively slow, particularly for Random
    Forest. To keep tuning practical, the search itself is run on a
    stratified subsample of at most `max_tune_samples` rows; the resulting
    best hyperparameters are then used to fit a final model on the FULL
    training set. This is a standard, defensible practice for large-scale
    hyperparameter search and does not affect the final model's training
    data, only which hyperparameters are searched over.

    Returns the fitted best estimator (trained on the FULL training set
    with the best found parameters) and a dict of the best parameters.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if model_name == "random_forest":
        base_model = RandomForestClassifier(
            random_state=random_state, class_weight="balanced"
        )
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [5, 10],
        }
    elif model_name == "logistic_regression":
        base_model = LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight="balanced"
        )
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Subsample (stratified) for the search itself, if the training set is large
    if len(X_train) > max_tune_samples:
        X_search, _, y_search, _ = train_test_split(
            X_train, y_train,
            train_size=max_tune_samples,
            stratify=y_train,
            random_state=random_state,
        )
    else:
        X_search, y_search = X_train, y_train

    search = GridSearchCV(
        base_model,
        param_grid,
        scoring="f1",   # F1, not accuracy — see class-imbalance discussion
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_search, y_search)

    # Refit the best-found hyperparameters on the FULL training set
    final_model = search.best_estimator_.__class__(**search.best_estimator_.get_params())
    final_model.fit(X_train, y_train)

    return final_model, search.best_params_


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, dataset_name,
                        random_state=42, tune=False, cv_folds=5):
    """
    Train either a Random Forest or Logistic Regression model, evaluate it
    on the held-out test set, and return a dict of results.

    Parameters
    ----------
    model_name : str
        Either "random_forest" or "logistic_regression".
    dataset_name : str
        Human-readable dataset label, used for labelling outputs/plots.
    tune : bool
        If True, run GridSearchCV (scored on F1) to select hyperparameters
        before fitting the final model. If False, use sensible defaults.
    cv_folds : int
        Number of cross-validation folds used during tuning.
    """
    best_params = None

    if tune:
        model, best_params = tune_hyperparameters(
            X_train, y_train, model_name, cv_folds=cv_folds, random_state=random_state
        )
    else:
        if model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200, random_state=random_state, class_weight="balanced"
            )
        elif model_name == "logistic_regression":
            model = LogisticRegression(
                max_iter=1000, random_state=random_state, class_weight="balanced"
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Probability of the positive class, needed for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred  # fallback, shouldn't happen for these two models

    results = {
        "dataset": dataset_name,
        "model": model_name,
        "tuned": tune,
        "best_params": best_params,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    results["confusion_matrix"] = cm

    return model, results


def plot_confusion_matrix(cm, model_name, dataset_name, output_path):
    """Save a confusion matrix plot to outputs/."""
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name.replace('_', ' ').title()}\n{dataset_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def get_feature_importance(model, feature_names, model_name):
    """
    Return a sorted pandas Series of feature importances.
    Random Forest: uses .feature_importances_
    Logistic Regression: uses absolute value of coefficients
    """
    if model_name == "random_forest":
        importances = model.feature_importances_
    elif model_name == "logistic_regression":
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def print_results_table(results_list):
    """Print a tidy summary table of results across all model runs."""
    df = pd.DataFrame(
        [
            {
                "Dataset": r["dataset"],
                "Model": r["model"],
                "Accuracy": round(r["accuracy"], 3),
                "Precision": round(r["precision"], 3),
                "Recall": round(r["recall"], 3),
                "F1": round(r["f1"], 3),
                "ROC-AUC": round(r["roc_auc"], 3),
            }
            for r in results_list
        ]
    )
    print(df.to_string(index=False))
    return df
def plot_correlation_matrix(df, target_col, dataset_name, output_path, figsize=(10, 8)):
    """
    Generate and save a correlation heatmap for all features (including the
    target) in a dataset. Used to compare raw univariate correlation with
    the outcome against model-based feature importance rankings.
    """
    plt.figure(figsize=figsize)
    corr = df.corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
    )
    plt.title(f"Correlation Matrix \u2014 {dataset_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return corr[target_col].sort_values(ascending=False)


def plot_combined_metrics_comparison(results_list, output_path):
    """
    Generate a grouped bar chart comparing accuracy, precision, recall, F1
    and ROC-AUC across all model/dataset combinations in results_list.
    """
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results_list):
        values = [r[m] for m in metrics]
        label = f"{r['dataset']} \u2014 {r['model'].replace('_', ' ').title()}"
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xticks(x + width * (len(results_list) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across Both Datasets")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
