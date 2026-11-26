"""pytest suite: data cleaning, encodings, feature pipeline, metrics, HDI, Bayesian model on synthetic data.

The real-data tests are skipped automatically when data/WA_Fn-UseC_-Telco-Customer-Churn.csv is absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data import clean, label_encode, load_dataset, make_synthetic, onehot_encode  # noqa: E402
from evaluate import bootstrap_ci, evaluate, expected_calibration_error  # noqa: E402
from features import FeaturePipeline, split  # noqa: E402
from models import BayesianLogisticRegression, hdi, sigmoid  # noqa: E402

DATA = os.path.join(ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
needs_data = pytest.mark.skipif(not os.path.exists(DATA), reason="run scripts/download_data.py first")


def tiny_raw():
    return pd.DataFrame({
        "customerID": ["a", "b", "c", "d"], "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0], "tenure": [1, 0, 30, 60],
        "InternetService": ["DSL", "No", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "No internet service", "Yes", "Yes"],
        "MonthlyCharges": [20.0, 19.5, 90.0, 60.0], "TotalCharges": ["20", " ", "2700", "3600"],
        "Churn": ["Yes", "No", "No", "No"],
    })


# ------------------------------------------------------------------ data
def test_clean_drops_blank_totalcharges_and_maps_target():
    df = clean(tiny_raw())
    assert len(df) == 3 and "customerID" not in df
    assert df["Churn"].tolist() == [1, 0, 0]
    assert df["TotalCharges"].dtype.kind == "f"


def test_clean_keep_id_legacy():
    assert "customerID" in clean(tiny_raw(), keep_id=True)


def test_label_encoding_is_sorted_integer_codes():
    enc = label_encode(clean(tiny_raw()))
    assert enc["InternetService"].tolist() == [0, 1, 0]          # DSL < Fiber optic
    assert all(pd.api.types.is_numeric_dtype(enc[c]) for c in enc)


def test_onehot_folds_redundant_levels():
    oh = onehot_encode(clean(tiny_raw()))
    assert not any("No internet service" in c for c in oh.columns)
    assert "InternetService_Fiber optic" in oh.columns and "OnlineSecurity_Yes" in oh.columns


@needs_data
def test_real_dataset_shape_and_rate():
    X, y = load_dataset(DATA)
    assert X.shape == (7032, 19) and abs(y.mean() - 0.2658) < 1e-3
    Xl, _ = load_dataset(DATA, keep_id=True)
    assert Xl.shape[1] == 20
    Xo, yo = load_dataset(DATA, encoding="onehot")
    assert Xo.shape == (7032, 23) and np.linalg.matrix_rank(Xo.values) == 23
    assert np.array_equal(y, yo)


@needs_data
def test_split_and_pca_match_reported_sizes():
    X, y = load_dataset(DATA)
    Xtr, Xte, ytr, yte = split(X, y)
    assert len(ytr) == 5625 and len(yte) == 1407
    fp = FeaturePipeline(0.95).fit(Xtr)
    assert fp.n_components == 16 and fp.explained_variance >= 0.95
    Xl, yl = load_dataset(DATA, keep_id=True)
    Xltr, *_ = split(Xl, yl)
    assert FeaturePipeline(0.95).fit(Xltr).n_components == 17       # the original README's "17 components"


# ------------------------------------------------------------------ features
def test_feature_pipeline_fit_on_train_only():
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(5, 2, (200, 6)), rng.normal(50, 2, (50, 6))
    fp = FeaturePipeline(None).fit(Xtr)
    Z = fp.transform(Xtr)
    assert np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1, atol=1e-9)
    assert fp.transform(Xte).mean() > 10                                # test stats NOT used


def test_pca_threshold_respected():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 10)) @ rng.normal(size=(10, 10))
    fp = FeaturePipeline(0.9).fit(X)
    r = fp.pca_.explained_variance_ratio_
    assert r.sum() >= 0.9 and r[:-1].sum() < 0.9
    assert fp.transform(X).shape == (300, fp.n_components)


# ------------------------------------------------------------------ metrics
def test_evaluate_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    r = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert r["Accuracy"] == 1 and r["ROC-AUC"] == 1 and r["F1"] == 1
    r = evaluate(y, np.full(4, 0.5))
    assert r["ROC-AUC"] == 0.5


def test_ece_zero_when_calibrated():
    y = np.array([0] * 8 + [1] * 2)
    assert expected_calibration_error(y, np.full(10, 0.2)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(10, 0.7)) == pytest.approx(0.5)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.uniform(0, 0.4, 300), 0, 1)
    ci = bootstrap_ci(y, {"a": p, "b": rng.uniform(size=300)}, n_boot=200, reference="a")
    auc = evaluate(y, p)["ROC-AUC"]
    assert ci["a"]["ROC-AUC"][0] <= auc <= ci["a"]["ROC-AUC"][1]
    assert ci["b"]["dAUC_vs_a"][2] < 0                                  # random model is worse


# ------------------------------------------------------------------ Bayesian helpers
def test_hdi_of_normal():
    s = np.random.default_rng(0).normal(size=(200_000, 1))
    lo, hi = hdi(s, 0.94)
    assert lo[0] == pytest.approx(-1.881, abs=0.03) and hi[0] == pytest.approx(1.881, abs=0.03)


def test_hdi_skewed_is_shorter_than_equal_tailed():
    s = np.random.default_rng(0).exponential(size=(50_000,))
    lo, hi = hdi(s, 0.9)
    q = np.quantile(s, [0.05, 0.95])
    assert hi - lo < q[1] - q[0] and lo < q[0]


def test_sigmoid_stable():
    assert sigmoid(np.array([0.0]))[0] == 0.5
    assert np.all(np.isfinite(sigmoid(np.array([-500.0, 500.0]))))


def test_synthetic_generator_is_reproducible():
    X1, y1, t1 = make_synthetic(500, 4, seed=3)
    X2, y2, t2 = make_synthetic(500, 4, seed=3)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2) and np.allclose(t1["beta"], t2["beta"])


@pytest.mark.slow
def test_bayesian_model_recovers_synthetic_coefficients():
    """SYNTHETIC data with known betas: posterior mean close to the truth, sampler converges."""
    X, y, truth = make_synthetic(n=3000, d=3, seed=7)
    m = BayesianLogisticRegression(draws=300, tune=300, chains=2, cores=1, seed=0).fit(X, y)
    b0, beta = m.posterior_samples()
    assert abs(b0.mean() - truth["intercept"]) < 0.2
    assert np.max(np.abs(beta.mean(0) - truth["beta"])) < 0.2
    d = m.diagnostics()
    assert d["max_rhat"] < 1.05 and d["divergences"] == 0
    s = m.predict_summary(X[:20])
    assert np.all(s["hdi_low"] <= s["mean"]) and np.all(s["mean"] <= s["hdi_high"])
    assert m.predict_proba(X[:5]).shape == (5, 2)
