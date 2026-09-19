"""
task2_common.py

Shared helpers for the Individual Task 2 analyses: data loading (including the
Cleveland label correction), leakage-free scikit-learn pipelines (impute ->
scale -> model, so every preprocessing step is learned from training folds
only), the hyper-parameter grids used in Task 1, and the sensitive-feature
groupings used in the Fairlearn audit.

Run any task2_*.py script from the project root, e.g.
    python src/task2_cv_sampling.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "task2"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
MODELS = ["random_forest", "logistic_regression"]
SHORT = {"random_forest": "RF", "logistic_regression": "LR"}

# Same grids as Task 1 (src/model_utils.py), scored on F1.
GRIDS = {
    "random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20],
        "model__min_samples_split": [5, 10],
    },
    "logistic_regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_cleveland(corrected=True):
    """Cleveland data.

    corrected=False reproduces the Task 1 data exactly as loaded there (Kaggle
    mirror, label as supplied, missing-value codes left in place).
    corrected=True (i) flips the label, because the mirror codes healthy
    patients as 1 (165 vs 138 patients; the original UCI file has 164 without
    and 139 with disease) and (ii) recodes the mirror's missing-value codes
    (ca = 4, thal = 0) as NaN so that imputation actually runs.
    """
    df = pd.read_csv(DATA / "heart_disease_cleveland.csv", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df["target"] = (df["target"] > 0).astype(int)
    if corrected:
        df["target"] = 1 - df["target"]
        df.loc[df["ca"] == 4, "ca"] = np.nan
        df.loc[df["thal"] == 0, "thal"] = np.nan
    return df


def load_brfss():
    df = pd.read_csv(DATA / "heart_disease_brfss.csv").dropna()
    df["HeartDiseaseorAttack"] = df["HeartDiseaseorAttack"].astype(int)
    return df


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_pipeline(model_name, **params):
    """impute (median) -> standardise -> class-weighted model."""
    if model_name == "random_forest":
        est = RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 200),
            random_state=SEED, class_weight="balanced", n_jobs=-1, **params)
    elif model_name == "logistic_regression":
        est = LogisticRegression(max_iter=1000, random_state=SEED,
                                 class_weight="balanced", **params)
    else:
        raise ValueError(model_name)
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", est)])


# --------------------------------------------------------------------------- #
# Sensitive-feature groupings for the Fairlearn audit
# --------------------------------------------------------------------------- #
def cleveland_groups(df):
    return {
        "Sex": df["sex"].map({0: "Female", 1: "Male"}),
        "Age": pd.cut(df["age"], [0, 49, 59, 200], labels=["<50", "50-59", "60+"]),
    }


def brfss_groups(df):
    """BRFSS coded categories: Age 1-13 (1 = 18-24, 5 = 40-44, 8 = 55-59, 13 = 80+), Income 1-8, Education 1-6."""
    return {
        "Sex": df["Sex"].map({0.0: "Female", 1.0: "Male"}),
        "Age": pd.cut(df["Age"], [0, 5, 8, 13], labels=["18-44", "45-59", "60+"]),
        "Income": pd.cut(df["Income"], [0, 4, 6, 8],
                         labels=["<$25k", "$25-50k", ">=$50k"]),
        "Education": pd.cut(df["Education"], [0, 4, 5, 6],
                            labels=["HS or less", "Some college", "College grad"]),
    }


# --------------------------------------------------------------------------- #
# Fixed (untuned) configurations used for the repeated / k-fold CV rows, so that
# differences between protocols are not confounded with re-tuning.
# Nested CV (task2_cv_sampling.py) tunes inside each outer training fold instead.
# --------------------------------------------------------------------------- #
FIXED_PARAMS = {
    "cleveland": {
        "random_forest": dict(n_estimators=200, max_depth=20, min_samples_split=5),
        "logistic_regression": dict(C=1.0),
    },
    # BRFSS: the hyper-parameters selected by the Task 1 grid search
    # (see src/model_utils.py; reproduced by task2_cv_sampling.py).
    "brfss": {
        "random_forest": dict(n_estimators=200, max_depth=10, min_samples_split=10),
        "logistic_regression": dict(C=0.01),
    },
}


def fixed_pipeline(dataset, model_name):
    return make_pipeline(model_name, **dict(FIXED_PARAMS[dataset][model_name]))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mean_sd(values):
    """Mean and (population) SD across folds, as reported in the tables."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std())


def confusion_rates(y_true, y_pred):
    """Return (recall/TPR, FPR, precision) with zero-division guards."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return tpr, fpr, prec


# --------------------------------------------------------------------------- #
# Out-of-fold predictions on BRFSS (cached: the RF fits take a few minutes)
# --------------------------------------------------------------------------- #
def brfss_oof(model_name, force=False):
    """Stratified 5-fold out-of-fold class predictions and probabilities.

    Returns dict with y, pred, proba (one entry per respondent, in file order)
    and fold_metrics (recall, f1, roc_auc per fold).
    """
    from sklearn.metrics import f1_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cache = OUT / f"oof_brfss_{model_name}.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    df = load_brfss()
    X = df.drop(columns="HeartDiseaseorAttack")
    y = df["HeartDiseaseorAttack"].to_numpy()
    pred = np.zeros(len(y), dtype=int)
    proba = np.zeros(len(y))
    folds = []
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(X, y), 1):
        m = fixed_pipeline("brfss", model_name).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
        proba[te] = m.predict_proba(X.iloc[te])[:, 1]
        folds.append([recall_score(y[te], pred[te]), f1_score(y[te], pred[te]),
                      roc_auc_score(y[te], proba[te])])
        print(f"  BRFSS {model_name} fold {k}/5 done", flush=True)
    out = dict(y=y, pred=pred, proba=proba, fold_metrics=np.array(folds))
    np.savez(cache, **out)
    return out
