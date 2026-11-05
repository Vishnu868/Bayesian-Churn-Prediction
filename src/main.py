"""End-to-end pipeline: preprocess -> PCA -> LR / RF / Bayesian HMC -> HTML report."""
import argparse
import os

import pandas as pd

from models import (bayesian_predict, evaluate, train_bayesian_hmc, train_logistic,
                    train_random_forest)
from preprocess import build_features, load_and_clean
from report import build_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_and_clean(args.data)
    d = build_features(df)
    print(f"Samples: {len(df)} | Features: {len(d['feature_names'])} | "
          f"PCA components: {d['pca'].n_components_} "
          f"({d['pca'].explained_variance_ratio_.sum():.1%} variance)")

    results = {}
    lr = train_logistic(d["X_train"], d["y_train"])
    results["Logistic Regression"] = evaluate(d["y_test"], lr.predict_proba(d["X_test"])[:, 1])
    rf = train_random_forest(d["X_train"], d["y_train"])
    results["Random Forest"] = evaluate(d["y_test"], rf.predict_proba(d["X_test"])[:, 1])

    _, idata = train_bayesian_hmc(d["X_train"], d["y_train"], draws=args.draws)
    mean, std = bayesian_predict(idata, d["X_test"])
    results["Bayesian HMC"] = evaluate(d["y_test"], mean)

    print(pd.DataFrame(results).T.round(4))
    meta = {
        "Total Samples": len(df), "Features": len(d["feature_names"]),
        "PCA Components": d["pca"].n_components_, "Churn Rate": f"{df['Churn'].mean():.1%}",
        "Models": 3, "HMC Samples": args.draws * 2,
    }
    out_file = os.path.join(args.out, "report.html")
    build_report(results, d["pca"], idata, mean, std, out_file, meta)
    print(f"Report written to {out_file}")


if __name__ == "__main__":
    main()
