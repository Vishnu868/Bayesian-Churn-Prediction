"""Dataset (EDA) figures for the report: churn by contract / tenure / charges / services.

    python scripts/make_figures.py      # -> docs/figures/dataset_*.png
"""
import os

import numpy as np
import pandas as pd
from _common import ROOT

import plots
from data import clean, load_raw

OUT = os.path.join(ROOT, "docs", "figures")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = clean(load_raw(os.path.join(ROOT, "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")))
    plt = plots.plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    ax = axes[0, 0]
    counts = df["Churn"].value_counts().sort_index()
    ax.bar(["Stayed (0)", "Churned (1)"], counts.values, color=[plots.GREY, plots.RED], width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 60, f"{v:,} ({v / len(df):.1%})", ha="center", fontsize=9)
    ax.set_title(f"Target: {len(df):,} customers"); ax.set_ylabel("Customers"); ax.set_ylim(0, counts.max() * 1.15)

    for ax, col in zip([axes[0, 1], axes[0, 2], axes[1, 0]], ["Contract", "InternetService", "PaymentMethod"]):
        r = df.groupby(col)["Churn"].agg(["mean", "size"]).sort_values("mean")
        ax.barh([s.replace(" (automatic)", " (auto)") for s in r.index], r["mean"] * 100, color=plots.BLUE, height=0.6)
        for i, (m, n) in enumerate(zip(r["mean"], r["size"])):
            ax.text(m * 100 + 0.8, i, f"{m:.0%}  (n={n:,})", va="center", fontsize=8.5)
        ax.set_xlabel("Churn rate (%)"); ax.set_title(f"Churn rate by {col}"); ax.set_xlim(0, 62)
        ax.axvline(df["Churn"].mean() * 100, ls="--", lw=1, color=plots.GREY)

    ax = axes[1, 1]
    bins = np.arange(0, 75, 3)
    ax.hist(df.loc[df.Churn == 0, "tenure"], bins=bins, color=plots.GREY, alpha=0.7, label="stayed")
    ax.hist(df.loc[df.Churn == 1, "tenure"], bins=bins, color=plots.RED, alpha=0.75, label="churned")
    ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Customers"); ax.set_title("Tenure: churners are new customers")
    ax.legend()
    ax = axes[1, 2]
    bins = np.linspace(18, 120, 30)
    ax.hist(df.loc[df.Churn == 0, "MonthlyCharges"], bins=bins, color=plots.GREY, alpha=0.7, label="stayed")
    ax.hist(df.loc[df.Churn == 1, "MonthlyCharges"], bins=bins, color=plots.RED, alpha=0.75, label="churned")
    ax.set_xlabel("Monthly charges ($)"); ax.set_title("Monthly charges"); ax.legend()
    fig.tight_layout()
    plots.save(fig, os.path.join(OUT, "dataset_overview.png"))

    # correlation of label-encoded features
    from data import label_encode
    enc = label_encode(df)
    C = enc.corr()
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C)), C.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(C)), C.columns, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Correlation of label-encoded features (why PCA helps)")
    plots.save(fig, os.path.join(OUT, "dataset_correlation.png"))

    summary = {
        "rows_raw": 7043, "rows_clean": len(df), "dropped_blank_TotalCharges": 7043 - len(df),
        "churn_rate": float(df.Churn.mean()),
        "churn_by_contract": df.groupby("Contract")["Churn"].mean().round(4).to_dict(),
        "churn_by_internet": df.groupby("InternetService")["Churn"].mean().round(4).to_dict(),
        "tenure_median_churn_vs_stay": [float(df[df.Churn == 1].tenure.median()), float(df[df.Churn == 0].tenure.median())],
    }
    pd.Series(summary).to_json(os.path.join(ROOT, "results", "dataset_summary.json"), indent=2)
    print(summary)


if __name__ == "__main__":
    main()
