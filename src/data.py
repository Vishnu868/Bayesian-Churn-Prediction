"""Loading and cleaning the IBM Telco Customer Churn data, plus a synthetic generator for sanity checks.

Two encodings are provided:

* ``label``  – every categorical column is label-encoded to one integer column (the original project's
  choice; 19 features). This is what the PCA + Bayesian pipeline uses.
* ``onehot`` – categorical columns are one-hot encoded (drop-first, redundant "No internet/phone service"
  levels folded into "No"), giving 23 interpretable columns for the extra "Bayesian logistic regression on
  original features" model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COL = "customerID"
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(path: str = DEFAULT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame, keep_id: bool = False) -> pd.DataFrame:
    """Fix ``TotalCharges`` (blank strings for 11 brand-new customers), drop those rows, map Churn to 0/1.

    ``keep_id=True`` keeps ``customerID`` as a column (it is label-encoded later). That reproduces the
    original project's run (20 "features", 17 PCA components) but is a bug: an ID carries no information.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
                                       errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)          # 7,043 -> 7,032 rows
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    if not keep_id:
        df = df.drop(columns=[ID_COL], errors="ignore")
    return df


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]


def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode every non-numeric column (sorted category order, same as sklearn's LabelEncoder)."""
    df = df.copy()
    for c in categorical_columns(df):
        cats = sorted(df[c].astype(str).unique())
        df[c] = df[c].astype(str).map({v: i for i, v in enumerate(cats)}).astype(int)
    return df


def onehot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot (drop-first) encoding of categorical columns; numeric columns are kept as-is."""
    df = df.drop(columns=[ID_COL], errors="ignore").copy()
    # "No internet service" / "No phone service" duplicate InternetService_No / PhoneService_Yes exactly
    # (6 perfectly collinear dummies); fold them into "No" so the design matrix has full column rank.
    for c in categorical_columns(df):
        df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    cats = categorical_columns(df)
    return pd.get_dummies(df, columns=cats, drop_first=True, dtype=float)


def load_dataset(path: str = DEFAULT_PATH, encoding: str = "label", keep_id: bool = False):
    """Return ``(X: DataFrame, y: ndarray)`` ready for splitting."""
    df = clean(load_raw(path), keep_id=keep_id)
    if encoding == "label":
        enc = label_encode(df)
    elif encoding == "onehot":
        enc = onehot_encode(df)
    else:
        raise ValueError(f"unknown encoding {encoding!r}")
    X = enc.drop(columns=[TARGET])
    y = enc[TARGET].to_numpy()
    return X, y


def make_synthetic(n: int = 5000, d: int = 8, seed: int = 0, intercept: float = -1.0):
    """Synthetic logistic data with KNOWN coefficients (sanity check only, not real customers).

    Features are correlated Gaussians; ``beta`` ~ N(0, 0.75^2). Returns ``X, y, true_params`` where
    ``true_params = {"intercept": ..., "beta": ...}``.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    cov = A @ A.T / d + np.eye(d) * 0.5
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    X = (X - X.mean(0)) / X.std(0)
    beta = rng.normal(0, 0.75, size=d)
    p = 1 / (1 + np.exp(-(intercept + X @ beta)))
    y = rng.binomial(1, p)
    return X, y, {"intercept": intercept, "beta": beta}
