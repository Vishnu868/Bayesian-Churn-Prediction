"""SYNTHETIC sanity check (not real customers): can the sampler recover KNOWN coefficients?

Generates logistic data with known (intercept, beta), fits the same Bayesian model, and measures how often
the 94% HDI contains the true value (should be close to 94%) across replicate datasets.

    python scripts/run_synthetic.py --reps 10 --out results/synthetic_sanity_check
"""
import argparse
import os

import numpy as np
from _common import ROOT, cached, save_json

from data import make_synthetic
from evaluate import evaluate
from models import BayesianLogisticRegression, hdi


def one(rep, args):
    Xa, ya, truth = make_synthetic(n=args.n + 2000, d=args.d, seed=rep)   # one draw, then split
    X, y, Xte, yte = Xa[: args.n], ya[: args.n], Xa[args.n:], ya[args.n:]
    p_true = 1 / (1 + np.exp(-(truth["intercept"] + Xte @ truth["beta"])))
    b = BayesianLogisticRegression(draws=args.draws, tune=args.tune, chains=args.chains, seed=rep).fit(X, y)
    b0, beta = b.posterior_samples()
    samples = np.column_stack([b0, beta])
    true = np.r_[truth["intercept"], truth["beta"]]
    lo, hi = hdi(samples, 0.94)
    ps = b.prob_samples(Xte, max_draws=1000)
    plo, phi = hdi(ps, 0.94)
    out = {"rep": rep, "true": true, "post_mean": samples.mean(0), "hdi_low": lo, "hdi_high": hi,
           "covered": ((true >= lo) & (true <= hi)).astype(int),
           "p_covered_frac": float(np.mean((p_true >= plo) & (p_true <= phi))),
           "rmse": float(np.sqrt(np.mean((samples.mean(0) - true) ** 2))),
           "test": evaluate(yte, ps.mean(0)), "oracle_test": evaluate(yte, p_true),
           **{f"diag_{k}": v for k, v in b.diagnostics().items() if k in ("max_rhat", "min_ess_bulk", "divergences")}}
    print(f"  rep {rep}: coverage {out['covered'].mean():.2f}, RMSE {out['rmse']:.3f}, AUC {out['test']['ROC-AUC']:.3f}"
          f" (oracle {out['oracle_test']['ROC-AUC']:.3f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/synthetic_sanity_check"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()
    runs = [cached(os.path.join(args.out, "cache", f"rep{r}.json"), lambda: one(r, args)) for r in range(args.reps)]
    cov = np.concatenate([np.asarray(r["covered"]) for r in runs])
    summary = {
        "DATA": "SYNTHETIC (known coefficients) - sanity check only, NOT real customers",
        "reps": args.reps, "n_train": args.n, "d": args.d,
        "param_hdi94_coverage": float(cov.mean()), "n_params_checked": int(cov.size),
        "prob_hdi94_coverage_mean": float(np.mean([r["p_covered_frac"] for r in runs])),
        "rmse_mean": float(np.mean([r["rmse"] for r in runs])),
        "test_auc_mean": float(np.mean([r["test"]["ROC-AUC"] for r in runs])),
        "oracle_auc_mean": float(np.mean([r["oracle_test"]["ROC-AUC"] for r in runs])),
        "max_rhat": float(max(r["diag_max_rhat"] for r in runs)),
        "divergences_total": int(sum(r["diag_divergences"] for r in runs)),
    }
    save_json({"summary": summary, "runs": runs}, os.path.join(args.out, "metrics.json"))
    print(summary)

    import plots
    r0 = runs[0]
    fig, axes = plots.plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    k = np.arange(len(r0["true"]))
    ax.hlines(k, r0["hdi_low"], r0["hdi_high"], color=plots.BLUE, lw=2, label="94% HDI")
    ax.scatter(r0["post_mean"], k, color=plots.BLUE, s=22, zorder=3, label="posterior mean")
    ax.scatter(r0["true"], k, marker="x", color=plots.RED, s=45, zorder=4, label="true value")
    ax.set_yticks(k, ["β0"] + [f"β{j + 1}" for j in range(len(k) - 1)])
    ax.set_xlabel("Coefficient"); ax.set_title("SYNTHETIC data, replicate 0: recovery of known coefficients")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    tr = np.concatenate([r["true"] for r in runs]); pm = np.concatenate([r["post_mean"] for r in runs])
    ax.scatter(tr, pm, s=14, color=plots.BLUE, alpha=0.7)
    lim = [min(tr.min(), pm.min()) - 0.1, max(tr.max(), pm.max()) + 0.1]
    ax.plot(lim, lim, ls="--", lw=1, color=plots.GREY)
    ax.set_xlabel("True coefficient"); ax.set_ylabel("Posterior mean")
    ax.set_title(f"All {args.reps} replicates: 94% HDI coverage = {summary['param_hdi94_coverage']:.1%}")
    fig.tight_layout()
    plots.save(fig, os.path.join(args.out, "synthetic_recovery.png"))


if __name__ == "__main__":
    main()
