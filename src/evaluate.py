"""Classification and calibration metrics, bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier", "LogLoss", "ECE"]


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE with equal-width bins: sum_b (n_b / N) |mean(y_b) - mean(p_b)|."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y, p, n_bins: int = 10):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return np.array(rows)


def evaluate(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-12, 1 - 1e-12)
    pred = (prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob) if len(np.unique(prob)) > 1 else 0.5,
        "PR-AUC": average_precision_score(y_true, prob),
        "Brier": brier_score_loss(y_true, prob),
        "LogLoss": log_loss(y_true, prob, labels=[0, 1]),
        "ECE": expected_calibration_error(y_true, prob),
    }


def bootstrap_ci(y_true, probs: dict, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                 reference: str | None = None) -> dict:
    """Percentile bootstrap CIs (test set resampled with replacement, same indices for every model).

    If ``reference`` is given, also returns the CI of ROC-AUC(model) - ROC-AUC(reference) (paired).
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {m: {k: [] for k in METRICS} for m in probs}
    diffs = {m: [] for m in probs}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        aucs = {}
        for m, p in probs.items():
            r = evaluate(y_true[idx], np.asarray(p)[idx])
            for k in METRICS:
                stats[m][k].append(r[k])
            aucs[m] = r["ROC-AUC"]
        if reference:
            for m in probs:
                diffs[m].append(aucs[m] - aucs[reference])
    out = {}
    for m in probs:
        out[m] = {k: [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
                  for k, v in stats[m].items()}
        if reference and m != reference:
            d = np.array(diffs[m])
            out[m][f"dAUC_vs_{reference}"] = [float(np.quantile(d, alpha / 2)), float(d.mean()),
                                               float(np.quantile(d, 1 - alpha / 2))]
    return out
