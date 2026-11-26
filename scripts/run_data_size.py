"""How does the Bayesian model behave when training data is scarce? (learning curve + interval width)

Sub-samples the 5,625-customer training split (stratified, several repeats), always evaluates on the same
1,407-customer test split, and compares the Bayesian model with unregularised-ish sklearn logistic regression.

    python scripts/run_data_size.py --out results/data_size
"""
import argparse
import os

import numpy as np
import pandas as pd
from _common import ROOT, cached, save_json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data import load_dataset
from evaluate import evaluate
from features import FeaturePipeline, split
from models import BayesianLogisticRegression


def one(n, rep, Xtr, ytr, Xte, yte, args):
    if n < len(ytr):
        Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=n, stratify=ytr, random_state=1000 + rep)
    else:
        Xs, ys = Xtr, ytr
    fp = FeaturePipeline(0.95).fit(Xs)
    Zs, Zte = fp.transform(Xs), fp.transform(Xte)
    b = BayesianLogisticRegression(draws=args.draws, tune=args.tune, chains=args.chains, seed=rep).fit(Zs, ys)
    s = b.predict_summary(Zte)
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(Zs, ys)                # ~maximum likelihood
    out = {"n": int(n), "rep": rep, "n_components": int(fp.n_components),
           "bayes": evaluate(yte, s["mean"]), "mle_lr": evaluate(yte, lr.predict_proba(Zte)[:, 1]),
           "hdi_width_mean": float(np.mean(s["hdi_high"] - s["hdi_low"])),
           "coverage_of_test_rate": None}
    print(f"  n={n:<5} rep={rep}: Bayes AUC {out['bayes']['ROC-AUC']:.3f} LL {out['bayes']['LogLoss']:.3f} | "
          f"MLE AUC {out['mle_lr']['ROC-AUC']:.3f} LL {out['mle_lr']['LogLoss']:.3f} | HDI {out['hdi_width_mean']:.3f}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/data_size"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 250, 500, 1000, 2500, 5625])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()
    X, y = load_dataset(os.path.join(ROOT, "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    Xtr, Xte, ytr, yte = split(X, y)
    runs = []
    for n in args.sizes:
        for r in range(1 if n >= len(ytr) else args.repeats):
            runs.append(cached(os.path.join(args.out, "cache", f"n{n}_r{r}.json"),
                               lambda: one(n, r, Xtr, ytr, Xte, yte, args)))
    rows = [{"n": r["n"], "rep": r["rep"], "bayes_auc": r["bayes"]["ROC-AUC"], "mle_auc": r["mle_lr"]["ROC-AUC"],
             "bayes_logloss": r["bayes"]["LogLoss"], "mle_logloss": r["mle_lr"]["LogLoss"],
             "bayes_ece": r["bayes"]["ECE"], "mle_ece": r["mle_lr"]["ECE"], "hdi_width_mean": r["hdi_width_mean"]}
            for r in runs]
    df = pd.DataFrame(rows)
    agg = df.groupby("n").agg(["mean", "std"]).drop(columns="rep")
    save_json({"runs": runs}, os.path.join(args.out, "metrics.json"))
    flat = df.groupby("n").mean().drop(columns="rep").reset_index()
    with open(os.path.join(args.out, "table.md"), "w") as f:
        f.write(f"Mean over {args.repeats} stratified sub-samples (1 for the full training set); fixed 1,407-customer test set\n\n")
        f.write(flat.to_markdown(index=False, floatfmt=".4f") + "\n")
    print(flat.round(4).to_string(index=False))

    import plots
    fig, axes = plots.plt.subplots(1, 3, figsize=(14, 3.9))
    m = agg
    for ax, (col, lab) in zip(axes[:2], [("auc", "Test ROC-AUC"), ("logloss", "Test log-loss (lower is better)")]):
        for key, name, c in [("bayes", "Bayesian LR (N(0,1) prior)", plots.BLUE), ("mle", "Logistic Regression (≈MLE, C=1e4)", plots.ORANGE)]:
            mu, sd = m[(f"{key}_{col}", "mean")], m[(f"{key}_{col}", "std")].fillna(0)
            ax.plot(m.index, mu, marker="o", ms=5, color=c, label=name)
            ax.fill_between(m.index, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
        ax.set_xscale("log"); ax.set_xlabel("Training customers (log scale)"); ax.set_title(lab)
    axes[0].legend(fontsize=8)
    axes[1].set_ylim(top=min(axes[1].get_ylim()[1], 1.2))
    ax = axes[2]
    ax.plot(m.index, m[("hdi_width_mean", "mean")], marker="o", ms=5, color=plots.BLUE)
    ax.set_xscale("log"); ax.set_xlabel("Training customers (log scale)"); ax.set_ylabel("Mean 94% HDI width of P(churn)")
    ax.set_title("Epistemic uncertainty shrinks with data")
    fig.tight_layout()
    plots.save(fig, os.path.join(args.out, "data_size.png"))


if __name__ == "__main__":
    main()
