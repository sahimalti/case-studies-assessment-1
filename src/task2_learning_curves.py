"""
task2_learning_curves.py

Part 2, "Performance under varying training set sizes".

For each dataset and model, trains on growing stratified subsets of each
training fold and records F1 and recall on (a) the subset it was trained on and
(b) the untouched validation fold.

    Cleveland (corrected label): repeated stratified 5-fold CV (5 repeats),
                                 training sizes 10%..100% of a fold's training set
    BRFSS:                       stratified 5-fold CV, sizes 500 .. full fold (log scale)

Writes outputs/task2/learning_curves.csv; task2_plot_learning_curves.py draws it.

Usage (from the project root):
    python src/task2_learning_curves.py            # both datasets
    python src/task2_learning_curves.py cleveland
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split)

from task2_common import (MODELS, OUT, SEED, fixed_pipeline, load_brfss,
                          load_cleveland)

BRFSS_SIZES = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]


def curves(dataset, X, y, cv, sizes_fn):
    recs = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        sizes = sizes_fn(len(tr))
        for n in sizes:
            if n < len(tr) - 1:   # top size = the whole fold (242 or 243 rows)
                sub, _ = train_test_split(tr, train_size=int(n), stratify=y.iloc[tr],
                                          random_state=SEED + fold)
            else:
                sub = tr
            for m in MODELS:
                t0 = time.time()
                model = fixed_pipeline(dataset, m).fit(X.iloc[sub], y.iloc[sub])
                for split, idx in [("train", sub), ("val", te)]:
                    pred = model.predict(X.iloc[idx])
                    recs.append(dict(dataset=dataset, model=m, fold=fold, size=int(n), split=split,
                                     f1=f1_score(y.iloc[idx], pred),
                                     recall=recall_score(y.iloc[idx], pred)))
        print(f"  {dataset} fold {fold + 1}/{cv.get_n_splits()} done", flush=True)
    return recs


def run_cleveland():
    df = load_cleveland(True)
    X, y = df.drop(columns="target"), df["target"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    # 10 sizes from 10% to 100% of the smallest fold's training set (24 ... 242 patients).
    # Sizes are fixed across folds (folds hold 242 or 243 patients) so every point is
    # averaged over all 25 folds.
    n_min = int(len(X) * 0.8)
    sizes = [int(f * n_min) for f in np.linspace(0.1, 1.0, 10)]
    return curves("cleveland", X, y, cv, lambda n: sizes)


def run_brfss():
    df = load_brfss()
    X, y = df.drop(columns="HeartDiseaseorAttack"), df["HeartDiseaseorAttack"]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return curves("brfss", X, y, cv, lambda n: BRFSS_SIZES + [n])


if __name__ == "__main__":
    which = sys.argv[1:] or ["cleveland", "brfss"]
    for w in which:
        print(f"== {w}", flush=True)
        recs = run_cleveland() if w == "cleveland" else run_brfss()
        pd.DataFrame(recs).to_csv(OUT / f"learning_curves_{w}.csv", index=False)
