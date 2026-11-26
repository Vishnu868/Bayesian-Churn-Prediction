"""End-to-end pipeline: clean -> encode -> z-score -> PCA -> {baselines, LR, RF, Bayesian HMC} -> results.

    python src/main.py                                   # corrected pipeline -> results/main/
    python src/main.py --legacy-keep-id --out results/legacy_with_customerID   # reproduces the original run

Everything (metrics, bootstrap CIs, posterior summary, diagnostics, per-customer predictions, figures,
models and an HTML report) is written to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plots  # noqa: E402
from data import DEFAULT_PATH, load_dataset  # noqa: E402
from evaluate import bootstrap_ci, evaluate  # noqa: E402
from features import FeaturePipeline, split  # noqa: E402
from models import (BayesianLogisticRegression, train_logistic, train_majority, train_prior,  # noqa: E402
                    train_random_forest)

logging.getLogger("pymc").setLevel(logging.ERROR)


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def markdown_table(results: dict, cols=("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier", "ECE")) -> str:
    lines = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m, r in results.items():
        lines.append(f"| {m} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_PATH)
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--variance", type=float, default=0.95, help="PCA explained-variance threshold")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--legacy-keep-id", action="store_true",
                    help="keep customerID as a feature (reproduces the original README run; a bug)")
    ap.add_argument("--skip-full", action="store_true", help="skip the no-PCA reference models")
    ap.add_argument("--no-save-models", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    out, figdir = args.out, os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)

    # ------------------------------------------------------------------ data
    X, y = load_dataset(args.data, encoding="label", keep_id=args.legacy_keep_id)
    X_tr, X_te, y_tr, y_te = split(X, y, seed=args.seed)
    fp = FeaturePipeline(variance=args.variance, seed=args.seed).fit(X_tr)
    Z_tr, Z_te = fp.transform(X_tr), fp.transform(X_te)
    info = {
        "data_source": "real: IBM Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)",
        "legacy_keep_customerID": args.legacy_keep_id,
        "n_customers": int(len(y)), "n_features": int(X.shape[1]), "feature_names": list(X.columns),
        "churn_rate": float(y.mean()), "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "train_churn_rate": float(y_tr.mean()), "test_churn_rate": float(y_te.mean()),
        "pca_components": int(fp.n_components), "pca_variance_retained": fp.explained_variance,
        "pca_variance_ratio": fp.pca_.explained_variance_ratio_.tolist(),
    }
    print(f"[data] {info['n_customers']} customers, {info['n_features']} features, churn {info['churn_rate']:.1%};"
          f" train {info['n_train']} / test {info['n_test']}; PCA -> {fp.n_components} comps "
          f"({fp.explained_variance:.1%})", flush=True)

    # ------------------------------------------------------------------ models on PCA features
    probs, models = {}, {}
    models["Majority class"] = train_majority(Z_tr, y_tr)
    models["Prior (base rate)"] = train_prior(Z_tr, y_tr)
    models["Logistic Regression"] = train_logistic(Z_tr, y_tr)
    models["Random Forest"] = train_random_forest(Z_tr, y_tr, seed=args.seed)
    for k, m in models.items():
        probs[k] = m.predict_proba(Z_te)[:, 1]

    print(f"[bayes] NUTS: {args.chains} chains x {args.draws} draws (tune {args.tune}), "
          f"prior N(0, {args.prior_sigma}^2)", flush=True)
    bayes = BayesianLogisticRegression(prior_sigma=args.prior_sigma, draws=args.draws, tune=args.tune,
                                       chains=args.chains, seed=args.seed, coords=fp.component_names())
    bayes.fit(Z_tr, y_tr)
    summ_te = bayes.predict_summary(Z_te)
    probs["Bayesian LR (HMC)"] = summ_te["mean"]
    diag = bayes.diagnostics()
    print(f"[bayes] max R-hat {diag['max_rhat']:.4f}, min ESS bulk {diag['min_ess_bulk']:.0f}, "
          f"divergences {diag['divergences']}", flush=True)

    # ------------------------------------------------------------------ reference models without PCA
    probs_full = {}
    bayes_oh = None
    if not args.skip_full:
        fs = FeaturePipeline(variance=None).fit(X_tr)
        S_tr, S_te = fs.transform(X_tr), fs.transform(X_te)
        probs_full["Logistic Regression (full, no PCA)"] = train_logistic(S_tr, y_tr).predict_proba(S_te)[:, 1]
        probs_full["Random Forest (full, no PCA)"] = train_random_forest(S_tr, y_tr, seed=args.seed).predict_proba(S_te)[:, 1]
        Xo, yo = load_dataset(args.data, encoding="onehot")
        Xo_tr, Xo_te, yo_tr, yo_te = split(Xo, yo, seed=args.seed)
        assert np.array_equal(yo_te, y_te)
        fo = FeaturePipeline(variance=None).fit(Xo_tr)
        print(f"[bayes] one-hot model: {Xo.shape[1]} indicator/numeric features", flush=True)
        bayes_oh = BayesianLogisticRegression(prior_sigma=args.prior_sigma, draws=args.draws, tune=args.tune,
                                              chains=args.chains, seed=args.seed, coords=list(Xo.columns))
        bayes_oh.fit(fo.transform(Xo_tr), yo_tr)
        probs_full["Bayesian LR (one-hot, no PCA)"] = bayes_oh.predict_summary(fo.transform(Xo_te))["mean"]
        diag_oh = bayes_oh.diagnostics()

    # ------------------------------------------------------------------ metrics
    results = {k: evaluate(y_te, p) for k, p in probs.items()}
    results_full = {k: evaluate(y_te, p) for k, p in probs_full.items()}
    all_probs = {**probs, **probs_full}
    ci = bootstrap_ci(y_te, all_probs, n_boot=args.n_boot, seed=args.seed, reference="Bayesian LR (HMC)")
    print("\n" + pd.DataFrame({**results, **results_full}).T.round(4).to_string(), flush=True)

    # ------------------------------------------------------------------ posterior summaries
    import arviz as az
    post_summary = az.summary(bayes.idata_, var_names=["intercept", "beta"], hdi_prob=0.94, round_to=6)
    post_summary.to_csv(os.path.join(out, "posterior_summary.csv"))
    rate_tr = bayes.posterior_predictive_rate(Z_tr)
    rate_te = bayes.posterior_predictive_rate(Z_te)
    w = summ_te["hdi_high"] - summ_te["hdi_low"]
    unc = {
        "hdi_prob": 0.94,
        "hdi_width_mean": float(w.mean()), "hdi_width_median": float(np.median(w)),
        "hdi_width_p90": float(np.quantile(w, 0.9)), "hdi_width_max": float(w.max()),
        "pred_sd_mean": float(summ_te["std"].mean()),
        "frac_hdi_straddles_0.5": float(np.mean((summ_te["hdi_low"] < 0.5) & (summ_te["hdi_high"] > 0.5))),
        "ppc_train_observed": float(y_tr.mean()), "ppc_train_interval94": np.quantile(rate_tr, [0.03, 0.97]).tolist(),
        "ppc_test_observed": float(y_te.mean()), "ppc_test_interval94": np.quantile(rate_te, [0.03, 0.97]).tolist(),
    }
    # accuracy on confident vs uncertain halves
    conf = np.abs(summ_te["mean"] - 0.5) / np.maximum(summ_te["std"], 1e-9)
    pred = (summ_te["mean"] >= 0.5).astype(int)
    half = conf >= np.median(conf)
    unc["acc_most_confident_half"] = float(np.mean(pred[half] == y_te[half]))
    unc["acc_least_confident_half"] = float(np.mean(pred[~half] == y_te[~half]))
    straddle = (summ_te["hdi_low"] < 0.5) & (summ_te["hdi_high"] > 0.5)
    unc["acc_hdi_straddles_0.5"] = float(np.mean(pred[straddle] == y_te[straddle])) if straddle.any() else None
    unc["acc_hdi_clear_of_0.5"] = float(np.mean(pred[~straddle] == y_te[~straddle]))

    # MAP vs posterior mean vs sklearn LR coefficients (Bayesian ~ penalised MLE check)
    lr = models["Logistic Regression"]
    coef_cmp = pd.DataFrame({
        "posterior_mean": np.r_[post_summary.loc["intercept", "mean"],
                                post_summary.loc[[i for i in post_summary.index if i.startswith("beta")], "mean"]],
        "posterior_sd": np.r_[post_summary.loc["intercept", "sd"],
                              post_summary.loc[[i for i in post_summary.index if i.startswith("beta")], "sd"]],
        "sklearn_LR": np.r_[lr.intercept_, lr.coef_.ravel()],
    }, index=["intercept"] + fp.component_names())
    coef_cmp.to_csv(os.path.join(out, "coefficients_bayes_vs_lr.csv"))

    # PCA loadings (what each component means)
    load = pd.DataFrame(fp.pca_.components_.T, index=X.columns, columns=fp.component_names())
    load.to_csv(os.path.join(out, "pca_loadings.csv"))

    # per-customer predictions
    pred_df = pd.DataFrame({"churn_actual": y_te, **{f"p_{k}": v for k, v in all_probs.items()},
                            "bayes_sd": summ_te["std"], "bayes_hdi94_low": summ_te["hdi_low"],
                            "bayes_hdi94_high": summ_te["hdi_high"]}, index=X_te.index)
    pred_df.index.name = "row"
    pred_df.round(6).to_csv(os.path.join(out, "predictions_test.csv"))

    # ------------------------------------------------------------------ figures
    print("[figures] writing", flush=True)
    main_probs = {k: v for k, v in probs.items() if k in ("Logistic Regression", "Random Forest", "Bayesian LR (HMC)")}
    plots.pca_variance(fp.pca_, os.path.join(figdir, "pca_variance.png"), args.variance)
    plots.roc_pr(y_te, {**main_probs, **probs_full}, os.path.join(figdir, "roc_pr.png"))
    plots.calibration(y_te, main_probs, os.path.join(figdir, "calibration.png"))
    plots.confusion_matrices(y_te, main_probs, os.path.join(figdir, "confusion_matrices.png"))
    plots.trace(bayes.idata_, os.path.join(figdir, "trace.png"))
    plots.rank(bayes.idata_, os.path.join(figdir, "rank.png"))
    plots.forest(bayes.idata_, os.path.join(figdir, "forest.png"))
    plots.energy(bayes.idata_, os.path.join(figdir, "energy.png"))
    plots.posterior_corr(bayes.idata_, os.path.join(figdir, "posterior_corr.png"))
    plots.ppc(rate_tr, y_tr.mean(), os.path.join(figdir, "ppc_train.png"))
    plots.uncertainty_band(summ_te, y_te, os.path.join(figdir, "uncertainty_band.png"))
    plots.uncertainty_analysis(summ_te, y_te, os.path.join(figdir, "uncertainty_analysis.png"))
    if bayes_oh is not None:
        plots.coefficient_forest_named(bayes_oh.idata_, list(Xo.columns), os.path.join(figdir, "onehot_coefficients.png"),
                                       "Bayesian LR on original (one-hot) features: what drives churn")
        az.summary(bayes_oh.idata_, var_names=["intercept", "beta"], hdi_prob=0.94, round_to=6).to_csv(
            os.path.join(out, "posterior_summary_onehot.csv"))

    # ------------------------------------------------------------------ save
    payload = {
        "info": info, "test_metrics": results, "test_metrics_no_pca": results_full, "bootstrap_95ci": ci,
        "bayes_diagnostics": diag, "bayes_onehot_diagnostics": diag_oh if bayes_oh is not None else None,
        "uncertainty": unc,
        "config": {k: v for k, v in vars(args).items()}, "threshold": 0.5, "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    with open(os.path.join(out, "metrics_table.md"), "w") as f:
        f.write("## PCA features\n\n" + markdown_table(results) + "\n")
        if results_full:
            f.write("\n## Without PCA (reference)\n\n" + markdown_table(results_full) + "\n")

    if not args.no_save_models:
        import joblib
        md = os.path.join(out, "models")
        os.makedirs(md, exist_ok=True)
        joblib.dump(fp, os.path.join(md, "feature_pipeline.joblib"), compress=3)
        joblib.dump(models["Logistic Regression"], os.path.join(md, "logistic_regression.joblib"), compress=3)
        joblib.dump(models["Random Forest"], os.path.join(md, "random_forest.joblib"), compress=3)
        az.InferenceData(posterior=bayes.idata_.posterior, sample_stats=bayes.idata_.sample_stats).to_netcdf(
            os.path.join(md, "bayes_posterior.nc"))
        if bayes_oh is not None:
            az.InferenceData(posterior=bayes_oh.idata_.posterior).to_netcdf(
                os.path.join(md, "bayes_onehot_posterior.nc"))

    from report import build_report
    build_report(out)
    print(f"[done] {time.time() - t0:.0f}s -> {out}/ (metrics.json, figures/, report.html)", flush=True)
    return payload


if __name__ == "__main__":
    main()
