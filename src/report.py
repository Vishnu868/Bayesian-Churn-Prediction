"""Self-contained HTML report (figures embedded as base64) built from a results directory.

    python src/report.py results/main        # -> results/main/report.html
"""
from __future__ import annotations

import base64
import html
import json
import os
import sys

import pandas as pd

SECTIONS = [
    ("PCA feature analysis", ["pca_variance.png"]),
    ("Model performance", ["roc_pr.png", "confusion_matrices.png"]),
    ("Calibration", ["calibration.png"]),
    ("Prediction uncertainty", ["uncertainty_band.png", "uncertainty_analysis.png"]),
    ("Posterior predictive check", ["ppc_train.png"]),
    ("HMC diagnostics", ["trace.png", "rank.png", "energy.png", "forest.png", "posterior_corr.png"]),
    ("Interpretable model (one-hot features)", ["onehot_coefficients.png"]),
]

CSS = """
:root{--bg:#ffffff;--ink:#191919;--muted:#666;--card:#f3f6fb;--line:#d8dee8}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#191919;--ink:#efefef;--muted:#9c9c9c;
--card:#232830;--line:#3a404a}}
:root[data-theme="dark"]{--bg:#191919;--ink:#efefef;--muted:#9c9c9c;--card:#232830;--line:#3a404a}
body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink);max-width:1100px;
margin:auto;padding:16px 16px 48px}
.kpi{display:inline-block;margin:4px;padding:10px 16px;border-radius:8px;background:var(--card)}
.kpi b{display:block;font-size:12px;color:var(--muted);font-weight:600}
table{border-collapse:collapse;font-size:13px;margin:8px 0;display:block;overflow-x:auto}
td,th{border:1px solid var(--line);padding:5px 9px;text-align:right}th:first-child,td:first-child{text-align:left}
img{max-width:100%;background:#fff;border-radius:6px;margin:6px 0}
.note{color:var(--muted);font-size:13px}
"""


def _img(path):
    with open(path, "rb") as f:
        return f'<img alt="{html.escape(os.path.basename(path))}" src="data:image/png;base64,{base64.b64encode(f.read()).decode()}"/>'


def build_report(out_dir: str) -> str:
    with open(os.path.join(out_dir, "metrics.json")) as f:
        M = json.load(f)
    info, d, u = M["info"], M["bayes_diagnostics"], M["uncertainty"]
    kpis = {
        "Customers": f"{info['n_customers']:,}", "Features": info["n_features"],
        "PCA components": f"{info['pca_components']} ({info['pca_variance_retained']:.1%})",
        "Churn rate": f"{info['churn_rate']:.1%}", "Train / test": f"{info['n_train']:,} / {info['n_test']:,}",
        "Posterior draws": f"{d['total_draws']:,}", "Max R-hat": f"{d['max_rhat']:.3f}",
        "Divergences": d["divergences"],
    }
    tbl = pd.DataFrame({**M["test_metrics"], **(M.get("test_metrics_no_pca") or {})}).T.round(4)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Bayesian Churn Report</title><style>{CSS}</style></head><body>",
             "<h1>Bayesian Churn Prediction</h1>",
             f"<p class='note'>Data: {html.escape(info['data_source'])}"
             + (" · <b>LEGACY run: customerID kept as a feature (reproduces original README; not valid)</b>"
                if info.get("legacy_keep_customerID") else "") + "</p>",
             "".join(f"<div class='kpi'><b>{k}</b>{v}</div>" for k, v in kpis.items()),
             "<h2>Test-set metrics (threshold 0.5)</h2>", tbl.to_html(),
             "<h2>Posterior summary</h2>",
             pd.read_csv(os.path.join(out_dir, "posterior_summary.csv"), index_col=0).round(4).to_html(),
             f"<p class='note'>Mean 94% HDI width of P(churn): {u['hdi_width_mean']:.3f} "
             f"(median {u['hdi_width_median']:.3f}); {u['frac_hdi_straddles_0.5']:.1%} of test customers have an HDI "
             f"that straddles 0.5.</p>"]
    fig = os.path.join(out_dir, "figures")
    for title, files in SECTIONS:
        present = [p for p in (os.path.join(fig, f) for f in files) if os.path.exists(p)]
        if present:
            parts.append(f"<h2>{title}</h2>" + "".join(_img(p) for p in present))
    parts.append("</body></html>")
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


if __name__ == "__main__":
    print(build_report(sys.argv[1] if len(sys.argv) > 1 else "results/main"))
