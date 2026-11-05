"""Classical baselines and the Bayesian logistic regression model sampled with HMC (NUTS)."""
import numpy as np
import pymc as pm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate(y_true, prob, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, prob),
    }


def train_logistic(X, y):
    return LogisticRegression(max_iter=1000).fit(X, y)


def train_random_forest(X, y, seed: int = 42):
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, y)


def train_bayesian_hmc(X, y, draws: int = 1000, tune: int = 1000, chains: int = 2, seed: int = 42):
    """Bayesian logistic regression: beta ~ N(0, 1), y ~ Bernoulli(sigmoid(X beta + b0))."""
    with pm.Model() as model:
        X_data = pm.Data("X", X)
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        p = pm.Deterministic("p", pm.math.sigmoid(intercept + pm.math.dot(X_data, beta)))
        pm.Bernoulli("y", p=p, observed=y, shape=X_data.shape[0])
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9,
                          random_seed=seed, progressbar=True)
    return model, idata


def bayesian_predict(idata, X):
    """Return posterior predictive mean and std of churn probability for each row of X."""
    post = idata.posterior
    b0 = post["intercept"].values.reshape(-1)              # (S,)
    beta = post["beta"].values.reshape(-1, X.shape[1])     # (S, D)
    probs = 1 / (1 + np.exp(-(b0[:, None] + beta @ X.T)))  # (S, N)
    return probs.mean(axis=0), probs.std(axis=0)
