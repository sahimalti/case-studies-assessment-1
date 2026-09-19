"""
task2_extra.py - additional Part 2 analyses for Individual Task 2.

  (1) intervals : 95% Wilson intervals for group-level recall (Fairlearn audit)
  (2) ttests    : corrected resampled t-tests (Nadeau & Bengio 2003) comparing
                  Logistic Regression and Random Forest on the same CV folds
  (3) sampling  : imbalance strategies (none / class weights / undersampling /
                  SMOTE) applied INSIDE each CV training fold on BRFSS, plus a
                  deliberately leaky SMOTE-before-splitting contrast

Run from the project root:
    python src/task2_extra.py               # all three stages
    python src/task2_extra.py intervals     # one stage only (intervals | ttests | sampling)

Outputs (outputs/task2/): recall_intervals.csv, model_comparison_ttests.csv,
sampling_strategies.csv. The sampling stage saves after every row, so an
interrupted run resumes where it stopped.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import task2_common as t

SCORING = ["recall", "precision", "f1", "roc_auc"]
MODELS = ["logistic_regression", "random_forest"]
STRATEGIES = ["none", "class_weight", "undersample", "smote"]


# --------------------------------------------------------------------------- #
# (1) Wilson intervals for group recall
# --------------------------------------------------------------------------- #
def wilson(p, n, z=1.96):
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def recall_intervals():
    print("== intervals", flush=True)
    g = pd.read_csv(t.OUT / "fairness_groups.csv")
    g = g[g["model"] == "logistic_regression"].copy()
    g["positives"] = (g["n"] * g["base_rate"]).round().astype(int)
    ci = [wilson(r, p) for r, p in zip(g["recall"], g["positives"])]
    g["recall_lo"] = [c[0] for c in ci]
    g["recall_hi"] = [c[1] for c in ci]
    g.to_csv(t.OUT / "recall_intervals.csv", index=False)
    print(g[["dataset", "attribute", "group", "n", "positives",
             "recall", "recall_lo", "recall_hi"]].round(3).to_string(index=False), flush=True)


# --------------------------------------------------------------------------- #
# (2) Corrected resampled t-tests
# --------------------------------------------------------------------------- #
def corrected_ttest(a, b, n_train, n_test):
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    t_stat = d.mean() / np.sqrt((1 / k + n_test / n_train) * d.var(ddof=1))
    return d.mean(), t_stat, 2 * stats.t.sf(abs(t_stat), df=k - 1)


def model_comparison():
    print("== ttests", flush=True)
    rows = []
    configs = [
        ("Cleveland (corrected)", lambda: t.load_cleveland(corrected=True), "target",
         RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=t.SEED), "cleveland"),
        ("BRFSS", t.load_brfss, "HeartDiseaseorAttack",
         StratifiedKFold(5, shuffle=True, random_state=t.SEED), "brfss"),
    ]
    for name, loader, target, cv, key in configs:
        X = loader()
        y = X.pop(target)
        scores = {m: cross_validate(t.fixed_pipeline(key, m), X, y, cv=cv, scoring=SCORING)
                  for m in MODELS}
        n_test = len(y) / 5
        for k in SCORING:
            diff, ts, p = corrected_ttest(scores["logistic_regression"][f"test_{k}"],
                                          scores["random_forest"][f"test_{k}"],
                                          len(y) - n_test, n_test)
            rows.append({"dataset": name, "metric": k, "LR_minus_RF": diff, "t": ts, "p": p})
        print(f"  {name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(t.OUT / "model_comparison_ttests.csv", index=False)
    print(out.round(4).to_string(index=False), flush=True)


# --------------------------------------------------------------------------- #
# (3) Imbalance strategies inside CV folds (BRFSS)
# --------------------------------------------------------------------------- #
def strategy_pipeline(model_name, strategy):
    """Task 1 fixed BRFSS configuration with the given imbalance strategy.
    Resampling sits inside an imblearn Pipeline, so it only ever sees the
    training fold."""
    impute, scale, model = t.fixed_pipeline("brfss", model_name).steps
    if strategy != "class_weight":
        model[1].set_params(class_weight=None)
    steps = [impute, scale]
    if strategy == "smote":
        steps.append(("resample", SMOTE(random_state=t.SEED)))
    elif strategy == "undersample":
        steps.append(("resample", RandomUnderSampler(random_state=t.SEED)))
    steps.append(model)
    return ImbPipeline(steps)


def _summarise(res, model, strategy):
    r = {"model": model, "strategy": strategy}
    for k in SCORING:
        r[k], r[k + "_sd"] = t.mean_sd(res[f"test_{k}"])
    return r


def sampling_experiment():
    print("== sampling (the random_forest + smote run is slow)", flush=True)
    path = t.OUT / "sampling_strategies.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model", "strategy"])
    rows = done.to_dict("records")
    have = {(r["model"], r["strategy"]) for r in rows}

    df = t.load_brfss()
    y = df.pop("HeartDiseaseorAttack")
    X = df
    cv = StratifiedKFold(5, shuffle=True, random_state=t.SEED)

    for m in MODELS:
        for s in STRATEGIES:
            if (m, s) in have:
                print(f"  {m} / {s}: cached", flush=True)
                continue
            res = cross_validate(strategy_pipeline(m, s), X, y, cv=cv, scoring=SCORING)
            rows.append(_summarise(res, m, s))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  {m} / {s}:", {k: round(rows[-1][k], 3) for k in SCORING}, flush=True)

    # Leaky contrast, for illustration only: SMOTE on the WHOLE dataset before
    # splitting, so synthetic points built from test-fold patients leak into training.
    key = ("logistic_regression", "smote_before_split_LEAKY")
    if key not in have:
        prep = t.fixed_pipeline("brfss", "logistic_regression")[:2]
        Xs, ys = SMOTE(random_state=t.SEED).fit_resample(prep.fit_transform(X), y)
        lr = t.fixed_pipeline("brfss", "logistic_regression")
        lr.steps[-1][1].set_params(class_weight=None)
        res = cross_validate(lr[-1:], Xs, ys, cv=cv, scoring=SCORING)
        rows.append(_summarise(res, *key))
        pd.DataFrame(rows).to_csv(path, index=False)
        print("  LEAKY smote-before-split:", {k: round(rows[-1][k], 3) for k in SCORING}, flush=True)

    out = pd.DataFrame(rows)
    print(out[["model", "strategy"] + SCORING].round(3).to_string(index=False), flush=True)


STAGES = {"intervals": recall_intervals, "ttests": model_comparison, "sampling": sampling_experiment}

if __name__ == "__main__":
    for stage in (sys.argv[1:] or list(STAGES)):
        STAGES[stage]()
