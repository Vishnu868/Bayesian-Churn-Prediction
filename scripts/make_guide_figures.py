"""Explanatory diagrams for docs/Project_Guide.pdf (pipeline, Bayes update, sigmoid, HMC, PCA, HDI).

    python scripts/make_guide_figures.py      # -> docs/figures/guide_*.png
"""
import os

import numpy as np
from _common import ROOT
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import plots
from data import clean, load_raw
from models import hdi

plt = plots.plt
OUT = os.path.join(ROOT, "docs", "figures")


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec=color,
                                alpha=0.16, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc="none", ec=color, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#191919")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color="#555", lw=1.2))


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    steps = [("Telco CSV\n7,043 customers\n21 columns", plots.GREY), ("Clean\ndrop 11 blank\nTotalCharges", plots.GREY),
             ("Encode\nlabel-encode\n19 features", plots.GREY), ("Split 80/20\nstratified\n5,625 / 1,407", plots.GREY),
             ("z-score\n(fit on train)", plots.GREY), ("PCA ≥95%\n16 components", plots.ORANGE)]
    for i, (t, c) in enumerate(steps):
        box(ax, 0.1 + i * 1.98, 3.0, 1.7, 1.3, t, c, 8.6)
        if i:
            arrow(ax, 0.1 + i * 1.98 - 0.27, 3.65, 0.1 + i * 1.98 - 0.02, 3.65)
    models = [("Majority / prior\nbaselines", plots.GREY), ("Logistic\nRegression", plots.ORANGE),
              ("Random Forest\n300 trees", plots.AQUA), ("Bayesian LR\nNUTS · 4×1000 draws", plots.BLUE)]
    for i, (t, c) in enumerate(models):
        box(ax, 0.6 + i * 2.35, 1.35, 2.0, 1.05, t, c, 8.8)
        arrow(ax, 10.85, 3.0, 1.6 + i * 2.35, 2.42)
    box(ax, 0.6, 0.1, 4.3, 0.85, "Evaluation: Acc · P · R · F1 · ROC-AUC · Brier · ECE\nbootstrap CIs · 5-fold CV", plots.VIOLET, 8.6)
    box(ax, 5.3, 0.1, 4.0, 0.85, "Uncertainty: posterior mean ± 94% HDI\nR-hat · ESS · trace · energy · PPC", plots.BLUE, 8.6)
    box(ax, 9.7, 0.1, 2.2, 0.85, "results/ +\nHTML report", plots.GREY, 8.6)
    for x in (2.75, 7.3, 10.8):
        arrow(ax, x, 1.33, x, 0.98)
    plots.save(fig, os.path.join(OUT, "guide_pipeline.png"))


def bayes_update():
    """Exact grid posterior of the churn RATE (intercept-only model) as more customers are seen."""
    df = clean(load_raw(os.path.join(ROOT, "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")))
    y = df["Churn"].sample(frac=1, random_state=0).to_numpy()
    b = np.linspace(-7, 7, 8001)
    rate = 1 / (1 + np.exp(-b))
    logprior = -0.5 * b ** 2
    fig, ax = plt.subplots(figsize=(9, 3.9))
    cols = [plots.GREY, plots.YELLOW, plots.ORANGE, plots.AQUA, plots.BLUE]
    for n, c in zip([0, 10, 100, 1000, 7032], cols):
        k = y[:n].sum()
        lp = logprior + k * np.log(rate) + (n - k) * np.log1p(-rate)
        dens_b = np.exp(lp - lp.max())
        dens_r = dens_b / (rate * (1 - rate))               # change of variables to the rate scale
        dens_r /= np.trapezoid(dens_r, rate)
        ax.plot(rate, dens_r, color=c, label=("prior N(0,1) on log-odds" if n == 0 else f"after {n:,} customers ({k} churned)"))
    ax.set_xlim(0, 1); ax.set_xlabel("Churn rate"); ax.set_ylabel("Density (belief)")
    ax.set_yscale("log"); ax.set_ylim(0.05, 200)
    ax.set_title("Bayesian updating: belief about the churn rate sharpens as data arrives (real data)")
    ax.legend(fontsize=8.5)
    plots.save(fig, os.path.join(OUT, "guide_bayes_update.png"))


def sigmoid():
    z = np.linspace(-7, 7, 400)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(z, 1 / (1 + np.exp(-z)), color=plots.BLUE)
    ax.axhline(0.5, ls="--", lw=1, color=plots.GREY); ax.axvline(0, ls="--", lw=1, color=plots.GREY)
    ax.annotate("β0 = −1.62  →  P = 0.165\n(average customer, all PCs = 0)", (-1.62, 0.165), xytext=(-6.8, 0.6),
                arrowprops=dict(arrowstyle="-", color=plots.MUTED), fontsize=8.5)
    ax.scatter([-1.62], [0.165], color=plots.RED, zorder=3)
    ax.set_xlabel("Linear score  η = β0 + Σ βj zj  (log-odds)"); ax.set_ylabel("P(churn) = σ(η)")
    ax.set_title("The logistic (sigmoid) link")
    plots.save(fig, os.path.join(OUT, "guide_sigmoid.png"))


def hmc_vs_rw():
    rng = np.random.default_rng(3)
    S = np.array([[1.0, 0.95], [0.95, 1.0]])
    P = np.linalg.inv(S)
    U = lambda q: 0.5 * q @ P @ q
    gU = lambda q: P @ q
    # random-walk Metropolis
    q = np.array([-2.0, -2.0]); rw = [q]
    for _ in range(60):
        prop = q + rng.normal(0, 0.25, 2)
        if np.log(rng.uniform()) < U(q) - U(prop):
            q = prop
        rw.append(q)
    rw = np.array(rw)
    # HMC with leapfrog
    q = np.array([-2.0, -2.0]); hm = [q]; paths = []
    eps, L = 0.12, 20
    for _ in range(8):
        p = rng.normal(size=2); q0, p0 = q.copy(), p.copy(); path = [q.copy()]
        p = p - eps / 2 * gU(q)
        for i in range(L):
            q = q + eps * p
            if i < L - 1:
                p = p - eps * gU(q)
            path.append(q.copy())
        p = p - eps / 2 * gU(q)
        if np.log(rng.uniform()) < (U(q0) + 0.5 * p0 @ p0) - (U(q) + 0.5 * p @ p):
            hm.append(q.copy()); paths.append(np.array(path))
        else:
            q = q0; hm.append(q0)
    hm = np.array(hm)
    g = np.linspace(-3, 3, 200); X, Y = np.meshgrid(g, g)
    Z = 0.5 * (P[0, 0] * X ** 2 + 2 * P[0, 1] * X * Y + P[1, 1] * Y ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, title in zip(axes, ["Random-walk Metropolis: 60 small blind steps",
                                "Hamiltonian Monte Carlo: 8 trajectories (leapfrog)"]):
        ax.contour(X, Y, Z, levels=[0.5, 2, 4.5, 8], colors=plots.GREY, linewidths=0.8)
        ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_title(title, fontsize=10)
        ax.set_xlabel("β1"); ax.set_ylabel("β2")
    axes[0].plot(rw[:, 0], rw[:, 1], "-o", ms=3, lw=1, color=plots.ORANGE)
    for pth in paths:
        axes[1].plot(pth[:, 0], pth[:, 1], lw=1, color=plots.BLUE, alpha=0.6)
    axes[1].plot(hm[:, 0], hm[:, 1], "o", ms=5, color=plots.BLUE)
    fig.suptitle("Exploring a correlated posterior (contours = −log posterior)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    plots.save(fig, os.path.join(OUT, "guide_hmc.png"))


def pca_toy():
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, 0.85], [0.85, 1]], 300)
    w, V = np.linalg.eigh(np.cov(X.T))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(X[:, 0], X[:, 1], s=8, color=plots.GREY, alpha=0.6)
    for k, c, lab, off in [(1, plots.BLUE, "PC1", (0.15, -0.55)), (0, plots.ORANGE, "PC2", (-1.6, 0.25))]:
        v = V[:, k] * 2.2 * np.sqrt(w[k])
        if v[0] < 0:
            v = -v
        if k == 0 and v[1] < 0:
            v = -v
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5))
        ax.text(v[0] + off[0], v[1] + off[1], f"{lab}: {w[k] / w.sum():.0%} of variance", color="#191919", fontsize=9,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_aspect("equal"); ax.set_xlabel("tenure (z-score)"); ax.set_ylabel("TotalCharges (z-score)")
    ax.set_title("PCA: rotate to uncorrelated directions", fontsize=10)
    plots.save(fig, os.path.join(OUT, "guide_pca.png"))


def hdi_fig():
    rng = np.random.default_rng(1)
    s = rng.gamma(2.0, 0.06, 200_000)
    lo, hi = hdi(s, 0.94)
    q = np.quantile(s, [0.03, 0.97])
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    ax.hist(s, bins=200, density=True, color=plots.BLUE, alpha=0.35)
    ax.axvspan(lo, hi, ymin=0, ymax=0.06, color=plots.BLUE, label=f"94% HDI  [{lo:.3f}, {hi:.3f}]  width {hi - lo:.3f}")
    ax.axvspan(q[0], q[1], ymin=0.07, ymax=0.13, color=plots.ORANGE, label=f"94% equal-tailed  [{q[0]:.3f}, {q[1]:.3f}]  width {q[1] - q[0]:.3f}")
    ax.set_xlim(0, 0.6); ax.set_xlabel("P(churn) for one customer (illustrative skewed posterior)")
    ax.set_yticks([]); ax.legend(fontsize=8.5); ax.set_title("Highest-density interval = shortest interval holding 94%")
    plots.save(fig, os.path.join(OUT, "guide_hdi.png"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline(); bayes_update(); sigmoid(); hmc_vs_rw(); pca_toy(); hdi_fig()
    print("guide figures written to", OUT)
