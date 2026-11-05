"""Generates diagnostic plots and a self-contained HTML report."""
import base64
import io

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig) -> str:
    return f'<img src="data:image/png;base64,{_fig_to_b64(fig)}"/>'


def pca_plot(pca):
    fig, ax = plt.subplots(figsize=(7, 4))
    cum = np.cumsum(pca.explained_variance_ratio_)
    ax.bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, label="per component")
    ax.plot(range(1, len(cum) + 1), cum, marker="o", color="crimson", label="cumulative")
    ax.axhline(0.95, ls="--", color="grey")
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance"); ax.legend()
    return fig


def uncertainty_plot(mean, std):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(mean)
    ax.plot(mean[order], color="navy", label="posterior mean")
    ax.fill_between(range(len(mean)), (mean - 2 * std)[order], (mean + 2 * std)[order],
                    alpha=0.3, label="±2 SD")
    ax.set_xlabel("Test customers (sorted)"); ax.set_ylabel("P(churn)"); ax.legend()
    return fig


def build_report(results: dict, pca, idata, mean, std, out_path: str, meta: dict):
    metrics = pd.DataFrame(results).T.round(4).to_html(classes="tbl")
    summary = az.summary(idata, var_names=["intercept", "beta"], round_to=4).to_html(classes="tbl")
    trace = az.plot_trace(idata, var_names=["intercept", "beta"], compact=True)
    forest = az.plot_forest(idata, var_names=["intercept", "beta"], combined=True)
    energy = az.plot_energy(idata)

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Bayesian Churn Report</title>
<style>body{{font-family:Segoe UI,Arial;max-width:1100px;margin:auto;padding:24px;color:#222}}
.kpi{{display:inline-block;margin:6px;padding:12px 18px;border-radius:8px;background:#eef3ff}}
.tbl{{border-collapse:collapse}} .tbl td,.tbl th{{border:1px solid #ccc;padding:6px 10px}}
img{{max-width:100%}}</style></head><body>
<h1>Bayesian Churn Prediction</h1><p>Probabilistic reasoning with HMC parameter learning</p>
{''.join(f'<div class="kpi"><b>{k}</b><br>{v}</div>' for k, v in meta.items())}
<h2>PCA Feature Analysis</h2>{_img(pca_plot(pca))}
<h2>Model Performance</h2>{metrics}
<h2>Posterior Parameter Summary</h2>{summary}
<h2>Prediction Uncertainty</h2>{_img(uncertainty_plot(mean, std))}
<h2>HMC Diagnostics</h2>
<h3>Trace plots</h3>{_img(trace.ravel()[0].figure)}
<h3>Forest plot</h3>{_img(forest.ravel()[0].figure)}
<h3>Energy</h3>{_img(energy.figure)}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
