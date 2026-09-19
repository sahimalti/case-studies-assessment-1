"""
task2_cv_sampling.py

Part 2, "Did my evaluation give an unbiased estimate?"

Stages (run all, or name the ones you want; each stage caches its result in
outputs/task2/ so an interrupted run can be resumed):

    task1     Task 1 hold-out results re-run with the Task 1 code
    repeated  repeated stratified 5-fold CV (10 repeats), Task 1 label vs corrected label
    nested    nested CV (inner grid search, outer repeated 5-fold) + the non-nested
              best_score_ that nesting is compared against
    holdout   Task 1 protocol repeated over 100 random 70/30 seeds (figure)
    brfss     stratified 5-fold CV on BRFSS (RF and LR)
    collect   write outputs/task2/cv_results.csv, holdout_summary.csv and the figure

Usage (from the project root):
    python src/task2_cv_sampling.py                # everything
    python src/task2_cv_sampling.py repeated nested collect
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.metrics import f1_score

from task2_common import (FIXED_PARAMS, GRIDS, MODELS, OUT, SEED, SHORT,
                          brfss_oof, fixed_pipeline, load_cleveland,
                          make_pipeline, mean_sd)

SCORING = {"recall": "recall", "f1": "f1", "auc": "roc_auc"}


def _cache(name):
    return OUT / f"cache_{name}.json"


def _summarise(res):
    row = {}
    for k in SCORING:
        m, s = mean_sd(res["test_" + k])
        row[k], row[k + "_sd"] = m, s
    return row


# --------------------------------------------------------------------------- #
def stage_task1():
    """Task 1 hold-out numbers, produced by the *Task 1* code (src/model_utils.py)."""
    import heart_disease_brfss_analysis as B
    import heart_disease_cleveland_analysis as C
    from model_utils import train_and_evaluate
    from sklearn.preprocessing import StandardScaler

    rows = []
    for name, mod, tgt, ts, folds in [
        ("Cleveland (Task 1 label)", C, "target", 0.3, 5),
        ("BRFSS", B, "HeartDiseaseorAttack", 0.2, 3),
    ]:
        df = mod.load_and_clean(mod.DATA_PATH)
        fc = [c for c in df.columns if c != tgt]
        Xtr, Xte, ytr, yte = train_test_split(df[fc], df[tgt], test_size=ts,
                                              random_state=42, stratify=df[tgt])
        sc = StandardScaler()
        Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=fc, index=Xtr.index)
        Xte = pd.DataFrame(sc.transform(Xte), columns=fc, index=Xte.index)
        for m in MODELS:
            _, r = train_and_evaluate(Xtr, Xte, ytr, yte, m, name, tune=True, cv_folds=folds)
            rows.append(dict(dataset=name, model=m, recall=r["recall"], f1=r["f1"],
                             auc=r["roc_auc"], accuracy=r["accuracy"],
                             precision=r["precision"], best_params=str(r["best_params"])))
            print("  task1", name, m, round(r["recall"], 3), round(r["f1"], 3),
                  round(r["roc_auc"], 3), r["best_params"], flush=True)
    json.dump(rows, open(_cache("task1"), "w"))


def stage_repeated():
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    rows = []
    for label, corrected in [("Cleveland (Task 1 label)", False), ("Cleveland (corrected)", True)]:
        df = load_cleveland(corrected)
        X, y = df.drop(columns="target"), df["target"]
        for m in MODELS:
            res = cross_validate(fixed_pipeline("cleveland", m), X, y, cv=cv, scoring=SCORING)
            rows.append(dict(dataset=label, model=m, **_summarise(res)))
            print("  repeated", label, m, {k: round(v, 3) for k, v in rows[-1].items() if k in SCORING},
                  flush=True)
    json.dump(rows, open(_cache("repeated"), "w"))


def stage_nested():
    df = load_cleveland(True)
    X, y = df.drop(columns="target"), df["target"]
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for m in MODELS:
        t0 = time.time()
        est = GridSearchCV(make_pipeline(m), GRIDS[m], scoring="f1", cv=inner)
        res = cross_validate(est, X, y, cv=outer, scoring=SCORING)
        # the optimistic alternative: best mean F1 of the grid search on one
        # 5-fold split, reported as if it were a test score
        gs = GridSearchCV(make_pipeline(m), GRIDS[m], scoring="f1", cv=inner).fit(X, y)
        rows.append(dict(dataset="Cleveland (corrected)", model=m, **_summarise(res),
                         non_nested_f1=float(gs.best_score_),
                         non_nested_params=str(gs.best_params_)))
        print(f"  nested {m} ({time.time() - t0:.0f}s)",
              {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    json.dump(rows, open(_cache("nested"), "w"))


def stage_holdout(n_seeds=100):
    """Task 1 protocol (stratified 70/30 split, grid search scored on F1) over many seeds."""
    df = load_cleveland(True)
    X, y = df.drop(columns="target"), df["target"]
    path = _cache("holdout")
    rows = json.load(open(path)) if path.exists() else []
    done = {r["seed"] for r in rows}
    for seed in range(n_seeds):
        if seed in done:
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        row = {"seed": seed}
        for m in MODELS:
            gs = GridSearchCV(make_pipeline(m), GRIDS[m], scoring="f1",
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
            gs.fit(Xtr, ytr)
            row[m] = float(f1_score(yte, gs.predict(Xte)))
        rows.append(row)
        json.dump(rows, open(path, "w"))
        if seed % 10 == 0:
            print(f"  holdout seed {seed}", row, flush=True)


def stage_brfss():
    rows = []
    for m in MODELS:
        oof = brfss_oof(m)
        fm = oof["fold_metrics"]
        row = dict(dataset="BRFSS", model=m)
        for i, k in enumerate(SCORING):
            row[k], row[k + "_sd"] = mean_sd(fm[:, i])
        rows.append(row)
        print("  brfss 5-fold", m, {k: round(v, 3) for k, v in row.items() if k in SCORING}, flush=True)
    json.dump(rows, open(_cache("brfss"), "w"))


# --------------------------------------------------------------------------- #
def stage_collect():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(n):
        p = _cache(n)
        return json.load(open(p)) if p.exists() else []

    task1, repeated, nested, brfss = load("task1"), load("repeated"), load("nested"), load("brfss")
    out = []

    def add(dataset, protocol, model, r, sd=True):
        out.append(dict(dataset=dataset, protocol=protocol, model=SHORT[model],
                        recall=r["recall"], recall_sd=r.get("recall_sd") if sd else None,
                        f1=r["f1"], f1_sd=r.get("f1_sd") if sd else None,
                        auc=r["auc"], auc_sd=r.get("auc_sd") if sd else None))

    t1 = {(r["dataset"], r["model"]): r for r in task1}
    rep = {(r["dataset"], r["model"]): r for r in repeated}
    for m in MODELS:
        if ("Cleveland (Task 1 label)", m) in t1:
            add("Cleveland (Task 1 label)", "Task 1 hold-out (70/30)", m,
                t1[("Cleveland (Task 1 label)", m)], sd=False)
    for m in MODELS:
        if ("Cleveland (Task 1 label)", m) in rep:
            add("Cleveland (Task 1 label)", "Repeated CV (5-fold, 10x)", m, rep[("Cleveland (Task 1 label)", m)])
    for m in MODELS:
        if ("Cleveland (corrected)", m) in rep:
            add("Cleveland (corrected)", "Repeated CV (5-fold, 10x)", m, rep[("Cleveland (corrected)", m)])
    for r in nested:
        add("Cleveland (corrected)", "Nested CV", r["model"], r)
    for m in MODELS:
        if ("BRFSS", m) in t1:
            add("BRFSS", "Task 1 hold-out (80/20), code re-run", m, t1[("BRFSS", m)], sd=False)
    for r in brfss:
        add("BRFSS", "Stratified 5-fold CV", r["model"], r)
    pd.DataFrame(out).to_csv(OUT / "cv_results.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))

    if nested:
        pd.DataFrame(nested).to_csv(OUT / "nested_vs_non_nested.csv", index=False)

    hold = load("holdout")
    if hold:
        h = pd.DataFrame(hold)
        summ = h[MODELS].agg(["mean", "std", "min", "max"]).T
        summ["n_seeds"] = len(h)
        summ.to_csv(OUT / "holdout_summary.csv")
        print(summ.round(3))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist([h["random_forest"], h["logistic_regression"]], bins=15, alpha=0.8,
                label=["Random Forest", "Logistic Regression"])
        ax.set_xlabel("F1 on a single 70/30 hold-out split")
        ax.set_ylabel("Number of seeds")
        ax.set_title(f"Cleveland: F1 across {len(h)} random hold-out splits")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "cleveland_holdout_variability.png", dpi=200)
        plt.close(fig)


STAGES = {"task1": stage_task1, "repeated": stage_repeated, "nested": stage_nested,
          "holdout": stage_holdout, "brfss": stage_brfss, "collect": stage_collect}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STAGES)
    for s in todo:
        print(f"== {s}", flush=True)
        STAGES[s]()
