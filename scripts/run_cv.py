"""Stratified 5-fold cross-validation of every model (scaler/PCA re-fitted inside each fold).

    python scripts/run_cv.py --folds 5 --out results/cv

Per-fold results are cached in ``<out>/folds/`` so an interrupted run resumes where it stopped.
"""
import argparse
import os

import numpy as np
import pandas as pd
from _common import ROOT, cached, save_json
from sklearn.model_selection import StratifiedKFold

from data import load_dataset
from evaluate import METRICS, evaluate
from features import FeaturePipeline
from models import BayesianLogisticRegression, train_logistic, train_random_forest

MODELS = ["Logistic Regression", "Random Forest", "Bayesian LR (HMC)",
          "Logistic Regression (full, no PCA)", "Random Forest (full, no PCA)", "Bayesian LR (one-hot, no PCA)"]


def run_fold(k, tr, te, X, Xo, y, args):
    out = {}
    fp = FeaturePipeline(0.95, seed=args.seed).fit(X.iloc[tr])
    Ztr, Zte = fp.transform(X.iloc[tr]), fp.transform(X.iloc[te])
    out["Logistic Regression"] = evaluate(y[te], train_logistic(Ztr, y[tr]).predict_proba(Zte)[:, 1])
    out["Random Forest"] = evaluate(y[te], train_random_forest(Ztr, y[tr], seed=args.seed).predict_proba(Zte)[:, 1])
    b = BayesianLogisticRegression(draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed + k).fit(Ztr, y[tr])
    out["Bayesian LR (HMC)"] = evaluate(y[te], b.predict_proba(Zte)[:, 1])
    out["Bayesian LR (HMC)"].update({f"diag_{d}": v for d, v in b.diagnostics().items()
                                     if d in ("max_rhat", "min_ess_bulk", "divergences")})
    fs = FeaturePipeline(None).fit(X.iloc[tr])
    Str, Ste = fs.transform(X.iloc[tr]), fs.transform(X.iloc[te])
    out["Logistic Regression (full, no PCA)"] = evaluate(y[te], train_logistic(Str, y[tr]).predict_proba(Ste)[:, 1])
    out["Random Forest (full, no PCA)"] = evaluate(
        y[te], train_random_forest(Str, y[tr], seed=args.seed).predict_proba(Ste)[:, 1])
    fo = FeaturePipeline(None).fit(Xo.iloc[tr])
    bo = BayesianLogisticRegression(draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed + k).fit(
        fo.transform(Xo.iloc[tr]), y[tr])
    out["Bayesian LR (one-hot, no PCA)"] = evaluate(y[te], bo.predict_proba(fo.transform(Xo.iloc[te]))[:, 1])
    out["n_pca_components"] = int(fp.n_components)
    print(f"  fold {k}: " + ", ".join(f"{m.split(' (')[0][:12]} AUC {out[m]['ROC-AUC']:.4f}" for m in MODELS[:3]),
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/cv"))
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    X, y = load_dataset(os.path.join(ROOT, "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    Xo, yo = load_dataset(os.path.join(ROOT, "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"), encoding="onehot")
    assert np.array_equal(y, yo)
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    folds = []
    for k, (tr, te) in enumerate(skf.split(X, y)):
        folds.append(cached(os.path.join(args.out, "folds", f"fold{k}.json"),
                            lambda: run_fold(k, tr, te, X, Xo, y, args), resume=not args.no_resume))

    summary = {}
    for m in MODELS:
        summary[m] = {met: {"mean": float(np.mean([f[m][met] for f in folds])),
                            "std": float(np.std([f[m][met] for f in folds], ddof=1))} for met in METRICS}
    save_json({"folds": args.folds, "summary": summary, "per_fold": folds}, os.path.join(args.out, "metrics.json"))
    cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier", "ECE"]
    lines = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in MODELS:
        lines.append(f"| {m} | " + " | ".join(f"{summary[m][c]['mean']:.4f} ± {summary[m][c]['std']:.4f}" for c in cols) + " |")
    with open(os.path.join(args.out, "cv_table.md"), "w") as f:
        f.write(f"Stratified {args.folds}-fold CV on all 7,032 customers (mean ± SD across folds)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    import plots
    fig, axes = plots.plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, met in zip(axes, ["ROC-AUC", "F1"]):
        data = [[f[m][met] for f in folds] for m in MODELS]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55, medianprops=dict(color="#191919"))
        for patch, m in zip(bp["boxes"], MODELS):
            patch.set_facecolor(plots.model_color(m)); patch.set_alpha(0.55)
        for i, d in enumerate(data):
            ax.scatter(d, np.full(len(d), i + 1), color=plots.model_color(MODELS[i]), s=14, zorder=3)
        ax.set_yticks(range(1, len(MODELS) + 1), MODELS if met == "ROC-AUC" else [""] * len(MODELS), fontsize=8.5)
        ax.set_xlabel(met); ax.set_title(f"{met} across {args.folds} CV folds")
    fig.tight_layout()
    plots.save(fig, os.path.join(args.out, "cv_boxplot.png"))


if __name__ == "__main__":
    main()
