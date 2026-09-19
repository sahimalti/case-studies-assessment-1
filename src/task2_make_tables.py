"""
task2_make_tables.py

Writes the LaTeX tables used in the report from the CSV results, so that every
number in the tables comes straight from the code:

    tab_cv.tex, tab_fair.tex, tab_gaps.tex, tab_labelcheck.tex, tab_mitigation.tex

Output goes to outputs/task2/tables/ (copy them next to the .tex files in Overleaf).

Rows marked "as reported" are the Task 1 hold-out figures as printed in the Task 1
report (kept for comparison with the re-run of the Task 1 code).
"""
import numpy as np
import pandas as pd

from task2_common import OUT, load_cleveland

TAB = OUT / "tables"
TAB.mkdir(exist_ok=True)

# Figures printed in the Task 1 report (recall, F1, ROC-AUC)
TASK1_AS_REPORTED = {
    ("Cleveland (Task 1 label)", "RF"): (0.860, 0.827, 0.876),
    ("Cleveland (Task 1 label)", "LR"): (0.840, 0.792, 0.866),
    ("BRFSS", "RF"): (0.662, 0.399, 0.841),
    ("BRFSS", "LR"): (0.797, 0.378, 0.847),
}


def f3(x):
    return f"{x:.3f}"


def pm(m, s):
    return f"{m:.3f} $\\pm$ {s:.3f}"


def write(name, text):
    (TAB / name).write_text(text)
    print("wrote", TAB / name)


# --------------------------------------------------------------------------- #
def tab_cv():
    cv = pd.read_csv(OUT / "cv_results.csv")
    lines = []

    def emit(dataset, protocol, model, rec, f1, auc):
        lines.append(f"{dataset} & {protocol} & {model} & {rec} & {f1} & {auc}\\\\")

    def reported(ds, model):
        r, f, a = TASK1_AS_REPORTED[(ds, model)]
        emit(ds, "Task 1 hold-out (as reported)", model, f3(r), f3(f), f3(a))

    def computed(ds, protocol, model):
        row = cv[(cv.dataset == ds) & (cv.protocol == protocol) & (cv.model == model)].iloc[0]
        if pd.isna(row.recall_sd):
            emit(ds, protocol, model, f3(row.recall), f3(row.f1), f3(row.auc))
        else:
            emit(ds, protocol, model, pm(row.recall, row.recall_sd), pm(row.f1, row.f1_sd),
                 pm(row.auc, row.auc_sd))

    c1, c2 = "Cleveland (Task 1 label)", "Cleveland (corrected)"
    for m in ["RF", "LR"]:
        reported(c1, m)
    for m in ["RF", "LR"]:
        computed(c1, "Repeated CV (5-fold, 10x)", m)
    for m in ["RF", "LR"]:
        computed(c2, "Repeated CV (5-fold, 10x)", m)
    for m in ["RF", "LR"]:
        computed(c2, "Nested CV", m)
    lines.append("\\midrule")
    for m in ["RF", "LR"]:
        reported("BRFSS", m)
    for m in ["RF", "LR"]:
        computed("BRFSS", "Task 1 hold-out (80/20), code re-run", m)
    for m in ["RF", "LR"]:
        computed("BRFSS", "Stratified 5-fold CV", m)
    body = "\n".join(lines)
    write("tab_cv.tex", f"""\\begin{{table}}[h]
\\caption{{Task 1 hold-out results versus repeated, nested and k-fold cross-validation (mean $\\pm$ SD across folds). RF = Random Forest, LR = Logistic Regression. ``As reported'' rows are the hold-out figures printed in the Task 1 report; ``code re-run'' rows come from re-running the Task 1 code.}}
\\label{{tab:cv}}
\\small\\setlength\\tabcolsep{{4pt}}
\\begin{{tabular}}{{lllccc}}
\\toprule
Dataset & Protocol & Model & Recall & F1 & ROC-AUC\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


# --------------------------------------------------------------------------- #
def tab_fair():
    g = pd.read_csv(OUT / "fairness_groups.csv")
    g = g[g.model == "logistic_regression"]
    order = {"Sex": ["Female", "Male"], "Income": ["<$25k", "$25-50k", ">=$50k"],
             "Age": {"Cleveland": ["<50", "50-59", "60+"], "BRFSS": ["18-44", "45-59", "60+"]},
             "Education": ["HS or less", "Some college", "College grad"]}
    def esc(grp):
        grp = grp.replace("<$", "$<$\\$").replace(">=$", "$\\geq$\\$")
        if grp.startswith("<") and not grp.startswith("$<$"):
            grp = "$<$" + grp[1:]
        if grp.startswith("$25"):
            grp = "\\" + grp
        return grp
    lines = []
    for ds, attrs in [("Cleveland", ["Sex", "Age"]), ("BRFSS", ["Sex", "Age", "Income", "Education"])]:
        first = True
        for attr in attrs:
            grps = order[attr][ds] if attr == "Age" else order[attr]
            for grp in grps:
                r = g[(g.dataset == ds) & (g.attribute == attr) & (g.group == grp)].iloc[0]
                label = ds if first else ""
                first = False
                lines.append(f"{label} & {attr} & {esc(grp)} & {int(round(r.n)):,} & {f3(r.base_rate)} & "
                             f"{f3(r.recall)} & {f3(r.fpr)} & {f3(r.precision)}\\\\")
        if ds == "Cleveland":
            lines.append("\\midrule")
    body = "\n".join(lines)
    write("tab_fair.tex", f"""\\begin{{table}}[h]
\\caption{{Fairlearn MetricFrame results by group for Logistic Regression, using out-of-fold predictions (Cleveland averaged over 10 repeats of 5-fold CV; BRFSS 5-fold CV).}}
\\label{{tab:fair}}
\\small\\setlength\\tabcolsep{{4pt}}
\\begin{{tabular}}{{lllrcccc}}
\\toprule
Dataset & Attribute & Group & $n$ & Base rate & Recall & FPR & Precision\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


def tab_gaps():
    g = pd.read_csv(OUT / "fairness_gaps.csv")
    short = {"logistic_regression": "LR", "random_forest": "RF"}
    lines = []
    for ds, attrs in [("Cleveland", ["Sex", "Age"]), ("BRFSS", ["Sex", "Age", "Income", "Education"])]:
        for m in ["logistic_regression", "random_forest"]:
            for a in attrs:
                r = g[(g.dataset == ds) & (g.model == m) & (g.attribute == a)].iloc[0]
                lines.append(f"{ds} & {short[m]} & {a} & {f3(r.tpr_gap)} & {f3(r.fpr_gap)} & "
                             f"{f3(r.eq_odds_diff)} & {f3(r.dem_parity_diff)}\\\\")
    body = "\n".join(lines)
    write("tab_gaps.tex", f"""\\begin{{table}}[h]
\\caption{{Largest between-group differences computed with Fairlearn. The equalized odds difference is the larger of the recall (TPR) and FPR gaps.}}
\\label{{tab:gaps}}
\\small\\setlength\\tabcolsep{{4pt}}
\\begin{{tabular}}{{lllcccc}}
\\toprule
Dataset & Model & Attribute & TPR gap & FPR gap & Eq.\\ odds diff. & Dem.\\ parity diff.\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


def tab_labelcheck():
    df = load_cleveland(corrected=False)          # the file exactly as used in Task 1
    df.loc[df.ca == 4, "ca"] = np.nan             # ca = 4 is the mirror's code for "missing"
    lines = []
    for t in [0, 1]:
        d = df[df.target == t]
        lines.append(f"target = {t} & {len(d)} & {d.thalach.mean():.1f} & {d.oldpeak.mean():.2f} & "
                     f"{d.exang.mean():.2f} & {d.ca.mean():.2f} & {(1 - d.sex.mean()):.2f}\\\\")
    body = "\n".join(lines)
    write("tab_labelcheck.tex", f"""\\begin{{table}}[h]
\\caption{{Mean clinical profile by label in the Kaggle mirror of the Cleveland data used in Task 1.}}
\\label{{tab:labelcheck}}
\\small\\setlength\\tabcolsep{{4pt}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
Label in file & $n$ & Max HR & ST depression & Exercise angina & Major vessels & Female share\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


def tab_mitigation():
    m = pd.read_csv(OUT / "mitigation.csv").set_index("variant")
    cols = ["Baseline LR", "Age removed", "ThresholdOptimizer"]
    rows = [("Overall recall", "overall_recall"), ("Overall precision", "overall_precision"),
            ("Overall F1", "overall_f1"), ("Balanced accuracy", "balanced_accuracy"),
            ("Recall gap across age groups", "recall_gap"), ("FPR gap across age groups", "fpr_gap"),
            ("Recall, 18-44", "recall_18-44"), ("Recall, 45-59", "recall_45-59"), ("Recall, 60+", "recall_60+"),
            ("FPR, 18-44", "fpr_18-44"), ("FPR, 45-59", "fpr_45-59"), ("FPR, 60+", "fpr_60+")]
    body = "\n".join(f"{lab} & " + " & ".join(f3(m.loc[c, key]) for c in cols) + "\\\\" for lab, key in rows)
    write("tab_mitigation.tex", f"""\\begin{{table}}[h]
\\caption{{Mitigating the age disparity on BRFSS (Logistic Regression, stratified 80/20 split). ThresholdOptimizer applies group-specific thresholds to equalise recall across age groups.}}
\\label{{tab:mitigation}}
\\small\\setlength\\tabcolsep{{4pt}}
\\begin{{tabular}}{{lccc}}
\\toprule
 & Baseline LR & Age removed & ThresholdOptimizer\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


if __name__ == "__main__":
    tab_labelcheck()
    for f in (tab_cv, tab_fair, tab_gaps, tab_mitigation):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
