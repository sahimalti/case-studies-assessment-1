"""
task2_plot_learning_curves.py

Draws Figure "Learning curves" (F1 solid, recall dashed, shaded band +/-1 SD across
folds) and prints/saves the summary table quoted in Part 2.

Reads outputs/task2/learning_curves_{cleveland,brfss}.csv, written by
task2_learning_curves.py.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from task2_common import OUT

TITLES = {("cleveland", "random_forest"): "Cleveland (corrected) — Random Forest",
          ("cleveland", "logistic_regression"): "Cleveland (corrected) — Logistic Regression",
          ("brfss", "random_forest"): "BRFSS — Random Forest",
          ("brfss", "logistic_regression"): "BRFSS — Logistic Regression"}
COL = {"train": "tab:blue", "val": "tab:orange"}


def main():
    df = pd.concat([pd.read_csv(OUT / f"learning_curves_{d}.csv") for d in ["cleveland", "brfss"]])
    g = (df.melt(id_vars=["dataset", "model", "fold", "size", "split"], value_vars=["f1", "recall"],
                 var_name="metric")
           .groupby(["dataset", "model", "size", "split", "metric"])["value"]
           .agg(["mean", "std"]).reset_index())
    g["std"] = g["std"].fillna(0)
    g.to_csv(OUT / "learning_curves_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, ((ds, m), title) in zip(axes.ravel(), TITLES.items()):
        sub = g[(g.dataset == ds) & (g.model == m)]
        for split in ["train", "val"]:
            for metric, ls in [("f1", "-"), ("recall", "--")]:
                s = sub[(sub.split == split) & (sub.metric == metric)].sort_values("size")
                name = f"{'Train' if split == 'train' else 'Validation'} {metric}"
                ax.plot(s["size"], s["mean"], ls, color=COL[split], label=name)
                ax.fill_between(s["size"], s["mean"] - s["std"], s["mean"] + s["std"],
                                color=COL[split], alpha=0.15)
        if ds == "brfss":
            ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=200)
    print("Saved", OUT / "learning_curves.png")

    # numbers quoted in the text
    last = g[g.groupby(["dataset", "model", "split", "metric"])["size"].transform("max") == g["size"]]
    print("\nAt full training size:")
    print(last.pivot_table(index=["dataset", "model"], columns=["split", "metric"], values="mean").round(3))


if __name__ == "__main__":
    main()
