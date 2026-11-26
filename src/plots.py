"""Matplotlib figures for evaluation, posterior diagnostics and uncertainty analysis.

Colour follows the model, never its rank: Bayesian = blue, logistic regression = orange,
random forest = aqua, baselines = grey (fixed categorical order of a CVD-validated palette).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from evaluate import reliability_curve  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
GREY, INK, MUTED = "#8c8c8c", "#191919", "#666666"


def model_color(name: str) -> str:
    n = name.lower()
    if "bayes" in n:
        return VIOLET if "one-hot" in n or "onehot" in n else BLUE
    if "logistic" in n:
        return YELLOW if "full" in n else ORANGE
    if "forest" in n:
        return GREEN if "full" in n else AQUA
    return GREY


plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
})


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pca_variance(pca, path, threshold=0.95):
    r = pca.explained_variance_ratio_
    c = np.cumsum(r)
    k = len(r)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(np.arange(1, k + 1), r, color=BLUE, width=0.75, label="per component")
    ax.plot(np.arange(1, k + 1), c, marker="o", ms=5, color=ORANGE, label="cumulative")
    ax.axhline(threshold, ls="--", lw=1, color=GREY)
    ax.text(1, threshold + 0.015, f"{threshold:.0%} threshold", color=MUTED, fontsize=8.5)
    ax.annotate(f"{k} components\n{c[-1]:.1%} variance", (k, c[-1]), xytext=(k - 4.5, 0.62),
                arrowprops=dict(arrowstyle="-", color=MUTED), fontsize=9, color=INK)
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, 1.05); ax.set_xticks(np.arange(1, k + 1)); ax.legend(loc="center right")
    ax.set_title("PCA on standardised features (fit on training split)")
    return save(fig, path)


def roc_pr(y, probs: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import auc
        axes[0].plot(fpr, tpr, color=model_color(name), label=f"{name} ({auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, color=model_color(name), label=name)
    axes[0].plot([0, 1], [0, 1], ls="--", lw=1, color=GREY)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (test set, AUC in legend)"); axes[0].legend(fontsize=8.5, loc="lower right")
    axes[1].axhline(np.mean(y), ls="--", lw=1, color=GREY)
    axes[1].text(0.02, np.mean(y) + 0.02, f"churn rate {np.mean(y):.1%}", color=MUTED, fontsize=8.5)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-recall curve")
    axes[1].legend(fontsize=8.5, loc="upper right")
    return save(fig, path)


def calibration(y, probs: dict, path, n_bins=10):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GREY, label="perfect calibration")
    for name, p in probs.items():
        rc = reliability_curve(y, p, n_bins)
        ax.plot(rc[:, 0], rc[:, 1], marker="o", ms=5, color=model_color(name), label=name)
    ax.set_xlabel("Mean predicted P(churn) in bin"); ax.set_ylabel("Observed churn rate in bin")
    ax.set_title("Reliability diagram (10 equal-width bins)"); ax.legend(fontsize=8.5)
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    for name, p in probs.items():
        ax.hist(p, bins=bins, histtype="step", lw=2, color=model_color(name), label=name)
    ax.set_xlabel("Predicted P(churn)"); ax.set_ylabel("Test customers"); ax.set_title("Prediction histogram")
    ax.legend(fontsize=8.5)
    return save(fig, path)


def confusion_matrices(y, probs: dict, path, threshold=0.5):
    from sklearn.metrics import confusion_matrix
    n = len(probs)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (name, p) in zip(axes, probs.items()):
        cm = confusion_matrix(y, (np.asarray(p) >= threshold).astype(int))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["stay", "churn"]); ax.set_yticks([0, 1], ["stay", "churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=9.5); ax.grid(False)
    fig.tight_layout()
    return save(fig, path)


def trace(idata, path):
    import arviz as az
    axes = az.plot_trace(idata, var_names=["intercept", "beta"], compact=True, figsize=(11, 5.5),
                         backend_kwargs={"constrained_layout": True})
    axes[0, 0].figure.suptitle("Trace plots: marginal posteriors (left) and chains over iterations (right)",
                               fontsize=11, fontweight="bold")
    return save(axes[0, 0].figure, path)


def rank(idata, path, max_params=6):
    import arviz as az
    nb = idata.posterior["beta"].shape[-1]
    coords = {"feature": list(idata.posterior["feature"].values[: min(max_params - 1, nb)])}
    axes = az.plot_rank(idata, var_names=["intercept", "beta"], coords=coords, kind="vlines",
                        figsize=(11, 4.2))
    for a in np.ravel(axes):
        a.set_xlabel(a.get_xlabel(), fontsize=9); a.tick_params(labelsize=8); a.title.set_fontsize(9.5)
    fig = np.ravel(axes)[0].figure
    fig.suptitle("Rank plots (uniform = chains mixed well)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, path)


def forest(idata, path, title="94% HDI of every coefficient (4 chains combined)", labels=None):
    import arviz as az
    nb = idata.posterior["beta"].shape[-1]
    axes = az.plot_forest(idata, var_names=["intercept", "beta"], combined=True, hdi_prob=0.94,
                          figsize=(7, 0.32 * nb + 1.6), colors=BLUE)
    ax = np.ravel(axes)[0]
    ax.axvline(0, ls="--", lw=1, color=GREY)
    ax.set_title(title)
    return save(ax.figure, path)


def energy(idata, path):
    import arviz as az
    ax = az.plot_energy(idata, figsize=(6.5, 3.8))
    ax.set_title("HMC energy: marginal vs transition (overlap = good exploration)")
    return save(ax.figure, path)


def posterior_corr(idata, path):
    post = idata.posterior
    b0 = post["intercept"].values.reshape(-1, 1)
    beta = post["beta"].values.reshape(b0.shape[0], -1)
    M = np.hstack([b0, beta])
    names = ["β0"] + [f"β{j + 1}" for j in range(beta.shape[1])]
    C = np.corrcoef(M.T)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="posterior correlation")
    ax.set_title("Posterior parameter correlations")
    return save(fig, path)


def ppc(rate_samples, observed, path, label="training"):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(rate_samples, bins=40, color=BLUE, alpha=0.85)
    ax.axvline(observed, color=RED, lw=2)
    ax.text(observed, ax.get_ylim()[1] * 0.92, f"  observed {observed:.3f}", color=INK, fontsize=9)
    lo, hi = np.quantile(rate_samples, [0.03, 0.97])
    ax.set_title(f"Posterior predictive check: churn rate on {label} data")
    ax.set_xlabel(f"Simulated churn rate (94% interval {lo:.3f}–{hi:.3f})"); ax.set_ylabel("Posterior draws")
    return save(fig, path)


def uncertainty_band(summary, y, path):
    m, lo, hi = summary["mean"], summary["hdi_low"], summary["hdi_high"]
    o = np.argsort(m)
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, lo[o], hi[o], color=BLUE, alpha=0.25, lw=0, label="94% HDI")
    ax.plot(x, m[o], color=BLUE, lw=1.6, label="posterior mean P(churn)")
    yy = np.asarray(y)[o]
    ax.scatter(x[yy == 1], np.full((yy == 1).sum(), -0.04), marker="|", s=18, color=RED, lw=0.8,
               label="actually churned")
    ax.axhline(0.5, ls="--", lw=1, color=GREY)
    ax.set_ylim(-0.08, 1.0); ax.set_xlabel("Test customers, sorted by posterior mean")
    ax.set_ylabel("P(churn)"); ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Per-customer predictive uncertainty (Bayesian logistic regression)")
    return save(fig, path)


def uncertainty_analysis(summary, y, path, threshold=0.5):
    m, lo, hi = summary["mean"], summary["hdi_low"], summary["hdi_high"]
    w = hi - lo
    y = np.asarray(y)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.scatter(m, w, s=8, color=BLUE, alpha=0.5, lw=0)
    ax.set_xlabel("Posterior mean P(churn)"); ax.set_ylabel("94% HDI width"); ax.set_title("Interval width vs prediction")
    ax = axes[1]
    ax.hist(w, bins=40, color=BLUE)
    ax.axvline(np.median(w), color=RED, lw=2)
    ax.text(np.median(w), ax.get_ylim()[1] * 0.9, f"  median {np.median(w):.3f}", fontsize=9)
    ax.set_xlabel("94% HDI width"); ax.set_ylabel("Test customers"); ax.set_title("Distribution of interval widths")
    ax = axes[2]
    straddle = (lo < threshold) & (hi > threshold)
    pred = (m >= threshold).astype(int)
    order = np.argsort(-np.abs(m - threshold) / np.maximum(summary["std"], 1e-9))   # most certain first
    cov = np.linspace(0.1, 1.0, 19)
    acc = [np.mean(pred[order[: int(c * len(m))]] == y[order[: int(c * len(m))]]) for c in cov]
    ax.plot(cov * 100, np.array(acc) * 100, marker="o", ms=4, color=BLUE)
    ax.set_xlabel("Customers kept (% most confident, by |mean−0.5|/sd)"); ax.set_ylabel("Accuracy on kept (%)")
    ax.set_title(f"Selective prediction ({straddle.mean():.1%} of HDIs straddle 0.5)")
    fig.tight_layout()
    return save(fig, path)


def coefficient_forest_named(idata, names, path, title, top=None):
    """Horizontal HDI plot with human-readable feature names, sorted by posterior mean."""
    from models import hdi
    beta = idata.posterior["beta"].values.reshape(-1, len(names))
    mean = beta.mean(0)
    lo, hi = hdi(beta, 0.94)
    o = np.argsort(mean)
    if top:
        keep = np.argsort(-np.abs(mean))[:top]
        o = [i for i in o if i in set(keep)]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(o) + 1.4))
    yy = np.arange(len(o))
    col = [RED if lo[i] > 0 else (BLUE if hi[i] < 0 else GREY) for i in o]
    ax.hlines(yy, lo[o], hi[o], color=col, lw=2)
    ax.scatter(mean[o], yy, color=col, s=22, zorder=3)
    ax.axvline(0, ls="--", lw=1, color=GREY)
    ax.set_yticks(yy, [names[i] for i in o], fontsize=8)
    ax.set_xlabel("Coefficient (log-odds per 1 SD of feature); 94% HDI")
    ax.set_title(title)
    ax.text(0.99, 0.01, "red: raises churn · blue: lowers churn · grey: HDI covers 0", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=MUTED)
    return save(fig, path)
