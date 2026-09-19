"""
task2_fairness.py

Part 2, "Bias analysis with Microsoft Fairlearn".

1. Audit: Fairlearn MetricFrame on out-of-fold predictions (every person is scored
   by a model that never saw them), by sex and age (both datasets) and income and
   education (BRFSS). Cleveland uses repeated 5-fold CV (10 repeats); metrics are
   computed per repeat and averaged. BRFSS uses one stratified 5-fold pass.
2. Gaps: largest between-group differences (TPR, FPR, equalized odds, demographic parity)
   for Logistic Regression and Random Forest.
3. Mitigation on BRFSS age: (a) drop age from the inputs ("fairness through
   unawareness"), (b) Fairlearn ThresholdOptimizer with group-specific thresholds
   that equalise recall across age groups. Stratified 80/20 split, seed 42.

Writes to outputs/task2/: fairness_groups.csv, fairness_gaps.csv, mitigation.csv,
fairness_by_group.png.
"""
import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               equalized_odds_difference, false_positive_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from task2_common import (MODELS, OUT, SEED, SHORT, brfss_groups, brfss_oof,
                          cleveland_groups, fixed_pipeline, load_brfss,
                          load_cleveland)

METRICS = {
    "recall": lambda yt, yp: recall_score(yt, yp, zero_division=np.nan),
    "fpr": false_positive_rate,
    "precision": lambda yt, yp: precision_score(yt, yp, zero_division=np.nan),
    "base_rate": lambda yt, yp: float(np.mean(yt)),
}
LR, RF = "logistic_regression", "random_forest"


def audit(dataset, y, pred, groups):
    """One MetricFrame per attribute -> (per-group table, per-attribute gaps)."""
    tabs, gaps = [], []
    for attr, g in groups.items():
        g = g.astype(str)
        mf = MetricFrame(metrics=METRICS, y_true=y, y_pred=pred, sensitive_features=g)
        t = mf.by_group.copy()
        t["n"] = g.value_counts()
        t = t.reset_index().rename(columns={t.index.name or "index": "group"})
        t.insert(0, "attribute", attr)
        t.insert(0, "dataset", dataset)
        tabs.append(t)
        gaps.append(dict(dataset=dataset, attribute=attr,
                         tpr_gap=float(mf.by_group["recall"].max() - mf.by_group["recall"].min()),
                         fpr_gap=float(mf.by_group["fpr"].max() - mf.by_group["fpr"].min()),
                         eq_odds_diff=float(equalized_odds_difference(y, pred, sensitive_features=g)),
                         dem_parity_diff=float(demographic_parity_difference(y, pred, sensitive_features=g))))
    return pd.concat(tabs), pd.DataFrame(gaps)


def cleveland_oof_repeats(model_name):
    df = load_cleveland(True)
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    preds = np.zeros((10, len(y)), dtype=int)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m = fixed_pipeline("cleveland", model_name).fit(X.iloc[tr], y[tr])
        preds[i // 5, te] = m.predict(X.iloc[te])
    return df, y, preds


def run_audit():
    groups_tabs, gaps_tabs = [], []
    for m in MODELS:
        # ---- Cleveland: average the per-repeat audits
        df, y, preds = cleveland_oof_repeats(m)
        groups = cleveland_groups(df)
        rep_t, rep_g = zip(*[audit("Cleveland", y, p, groups) for p in preds])
        t = pd.concat(rep_t).groupby(["dataset", "attribute", "group"], sort=False).mean(numeric_only=True).reset_index()
        g = pd.concat(rep_g).groupby(["dataset", "attribute"], sort=False).mean(numeric_only=True).reset_index()
        t.insert(0, "model", m); g.insert(0, "model", m)
        groups_tabs.append(t); gaps_tabs.append(g)
        print(f"  Cleveland {m} audited", flush=True)

        # ---- BRFSS: single out-of-fold pass
        oof = brfss_oof(m)
        b = load_brfss()
        t, g = audit("BRFSS", oof["y"], oof["pred"], brfss_groups(b))
        t.insert(0, "model", m); g.insert(0, "model", m)
        groups_tabs.append(t); gaps_tabs.append(g)
        print(f"  BRFSS {m} audited", flush=True)

    groups = pd.concat(groups_tabs, ignore_index=True)
    gaps = pd.concat(gaps_tabs, ignore_index=True)
    groups.to_csv(OUT / "fairness_groups.csv", index=False)
    gaps.to_csv(OUT / "fairness_gaps.csv", index=False)
    return groups, gaps


# --------------------------------------------------------------------------- #
def _age_report(name, y, pred, age):
    age = age.astype(str)
    mf = MetricFrame(metrics={"recall": recall_score, "fpr": false_positive_rate},
                     y_true=y, y_pred=pred, sensitive_features=age)
    row = {"variant": name,
           "overall_recall": recall_score(y, pred),
           "overall_precision": precision_score(y, pred),
           "overall_f1": f1_score(y, pred),
           "balanced_accuracy": balanced_accuracy_score(y, pred),
           "recall_gap": mf.by_group["recall"].max() - mf.by_group["recall"].min(),
           "fpr_gap": mf.by_group["fpr"].max() - mf.by_group["fpr"].min()}
    for grp in ["18-44", "45-59", "60+"]:
        row[f"recall_{grp}"] = mf.by_group.loc[grp, "recall"]
        row[f"fpr_{grp}"] = mf.by_group.loc[grp, "fpr"]
    return row


def run_mitigation():
    df = load_brfss()
    X, y = df.drop(columns="HeartDiseaseorAttack"), df["HeartDiseaseorAttack"]
    age = brfss_groups(df)["Age"].astype(str)
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(X, y, age, test_size=0.2,
                                                    random_state=SEED, stratify=y)
    rows = []

    base = fixed_pipeline("brfss", LR).fit(Xtr, ytr)
    rows.append(_age_report("Baseline LR", yte, base.predict(Xte), ate))

    noage = fixed_pipeline("brfss", LR).fit(Xtr.drop(columns="Age"), ytr)
    rows.append(_age_report("Age removed", yte, noage.predict(Xte.drop(columns="Age")), ate))

    to = ThresholdOptimizer(estimator=base, constraints="true_positive_rate_parity",
                            objective="balanced_accuracy_score", prefit=True,
                            predict_method="predict_proba")
    to.fit(Xtr, ytr, sensitive_features=atr)
    rows.append(_age_report("ThresholdOptimizer", yte,
                            to.predict(Xte, sensitive_features=ate, random_state=SEED), ate))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mitigation.csv", index=False)
    print(out.round(3).T.to_string())
    return out


# --------------------------------------------------------------------------- #
def plot(groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = {"Sex": ["Female", "Male"], "Age": ["<50", "50-59", "60+", "18-44", "45-59"],
             "Income": ["<$25k", "$25-50k", ">=$50k"],
             "Education": ["HS or less", "Some college", "College grad"]}
    age_order = {"Cleveland": ["<50", "50-59", "60+"], "BRFSS": ["18-44", "45-59", "60+"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [5, 11]})
    for ax, ds, title in zip(axes, ["Cleveland", "BRFSS"],
                             ["Cleveland (corrected) — Logistic Regression",
                              "BRFSS — Logistic Regression"]):
        t = groups[(groups.dataset == ds) & (groups.model == LR)]
        labels, rec, fpr = [], [], []
        for attr in ["Sex", "Age", "Income", "Education"]:
            for grp in (age_order[ds] if attr == "Age" else order[attr]):
                r = t[(t.attribute == attr) & (t.group == grp)]
                if len(r):
                    labels.append(f"{attr}: {grp}"); rec.append(float(r.recall.iloc[0]))
                    fpr.append(float(r.fpr.iloc[0]))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, rec, 0.4, label="Recall (TPR)")
        ax.bar(x + 0.2, fpr, 0.4, label="False positive rate")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ds == "BRFSS":
            ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fairness_by_group.png", dpi=200)


if __name__ == "__main__":
    g, _ = run_audit()
    run_mitigation()
    plot(g)
