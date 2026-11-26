"""Prior sensitivity (beta ~ N(0, s^2) for several s) and PSIS-LOO model comparison.

    python scripts/run_prior_sensitivity.py --out results/prior_sensitivity
"""
import argparse
import os

import arviz as az
import numpy as np
import pandas as pd
from _common import ROOT, cached, save_json

from data import load_dataset
from evaluate import evaluate
from features import FeaturePipeline, split
from models import BayesianLogisticRegression

DATA = os.path.join(ROOT, "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def fit_one(Ztr, ytr, Zte, yte, sigma, args, names):
    b = BayesianLogisticRegression(prior_sigma=sigma, draws=args.draws, tune=args.tune, chains=args.chains,
                                   seed=args.seed, coords=names).fit(Ztr, ytr, log_likelihood=True)
    s = b.predict_summary(Zte)
    loo = az.loo(b.idata_, pointwise=False)
    r = evaluate(yte, s["mean"])
    r.update({"prior_sigma": sigma, "elpd_loo": float(loo.elpd_loo), "elpd_loo_se": float(loo.se),
              "p_loo": float(loo.p_loo), "hdi_width_mean": float(np.mean(s["hdi_high"] - s["hdi_low"])),
              "beta_abs_mean": float(np.abs(b.posterior_samples()[1].mean(0)).mean()),
              "intercept_mean": float(b.posterior_samples()[0].mean()), **{f"diag_{k}": v for k, v in
              b.diagnostics().items() if k in ("max_rhat", "min_ess_bulk", "divergences")}})
    print(f"  {names[0][:2]} sigma={sigma:<6} AUC {r['ROC-AUC']:.4f} logloss {r['LogLoss']:.4f} "
          f"elpd_loo {r['elpd_loo']:.1f} ± {r['elpd_loo_se']:.1f}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/prior_sensitivity"))
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y = load_dataset(DATA)
    Xtr, Xte, ytr, yte = split(X, y)
    fp = FeaturePipeline(0.95).fit(Xtr)
    Xo, _ = load_dataset(DATA, encoding="onehot")
    Xotr, Xote, _, _ = split(Xo, y)
    fo = FeaturePipeline(None).fit(Xotr)

    rows = []
    for s in args.sigmas:
        rows.append({"features": "PCA (16 comps)", **cached(os.path.join(args.out, "cache", f"pca_{s}.json"),
                    lambda: fit_one(fp.transform(Xtr), ytr, fp.transform(Xte), yte, s, args, fp.component_names()))})
    rows.append({"features": "one-hot (23 cols)", **cached(os.path.join(args.out, "cache", "onehot_1.0.json"),
                lambda: fit_one(fo.transform(Xotr), ytr, fo.transform(Xote), yte, 1.0, args, list(Xo.columns)))})
    df = pd.DataFrame(rows)
    save_json({"rows": rows}, os.path.join(args.out, "metrics.json"))
    cols = ["features", "prior_sigma", "ROC-AUC", "LogLoss", "Brier", "F1", "elpd_loo", "elpd_loo_se", "p_loo",
            "hdi_width_mean", "beta_abs_mean"]
    with open(os.path.join(args.out, "table.md"), "w") as f:
        f.write(df[cols].to_markdown(index=False, floatfmt=".4f") + "\n")
    print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
