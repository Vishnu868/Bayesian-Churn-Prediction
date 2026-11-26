"""Baselines (majority class, logistic regression, random forest) and Bayesian logistic regression (NUTS).

Bayesian model (as in the original project):

    beta_0 ~ N(0, s^2),  beta_j ~ N(0, s^2)   (s = 1 by default)
    p_i    = sigmoid(beta_0 + sum_j beta_j z_ij)
    y_i    ~ Bernoulli(p_i)

The posterior p(beta | D) is sampled with the No-U-Turn Sampler (adaptive Hamiltonian Monte Carlo) in PyMC.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------- baselines
def train_majority(X, y):
    return DummyClassifier(strategy="most_frequent").fit(X, y)


def train_prior(X, y):
    """Predicts the training churn rate for everyone: the 'no-skill' probabilistic baseline."""
    return DummyClassifier(strategy="prior").fit(X, y)


def train_logistic(X, y):
    return LogisticRegression(max_iter=2000).fit(X, y)


def train_random_forest(X, y, seed: int = 42, n_estimators: int = 300, n_jobs: int = 4):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs).fit(X, y)


# ----------------------------------------------------------------------------- Bayesian model
def hdi(samples: np.ndarray, prob: float = 0.94) -> np.ndarray:
    """Highest-density interval along axis 0 (vectorised). ``samples``: (S, ...) -> (2, ...)."""
    s = np.sort(np.asarray(samples), axis=0)
    n = s.shape[0]
    k = int(np.floor(prob * n))
    if k < 1 or k >= n:
        return np.stack([s[0], s[-1]])
    widths = s[k:] - s[: n - k]
    i = np.argmin(widths, axis=0)
    lo = np.take_along_axis(s, i[None], axis=0)[0]
    hi = np.take_along_axis(s, (i + k)[None], axis=0)[0]
    return np.stack([lo, hi])


class BayesianLogisticRegression:
    """Bayesian logistic regression with Gaussian priors, sampled with NUTS (PyMC)."""

    def __init__(self, prior_sigma: float = 1.0, draws: int = 1000, tune: int = 1000, chains: int = 4,
                 cores: int | None = None, target_accept: float = 0.9, seed: int = 42, progressbar: bool = False,
                 coords: list[str] | None = None):
        self.prior_sigma = prior_sigma
        self.draws, self.tune, self.chains = draws, tune, chains
        self.cores = cores if cores is not None else min(chains, 4)
        self.target_accept, self.seed, self.progressbar = target_accept, seed, progressbar
        self.coords = coords

    def build(self, X, y):
        import pymc as pm

        X = np.asarray(X, dtype=float)
        names = self.coords or [f"x{j + 1}" for j in range(X.shape[1])]
        with pm.Model(coords={"feature": names}) as model:
            Xd = pm.Data("X", X)
            b0 = pm.Normal("intercept", mu=0.0, sigma=self.prior_sigma)
            beta = pm.Normal("beta", mu=0.0, sigma=self.prior_sigma, dims="feature")
            logit_p = b0 + pm.math.dot(Xd, beta)
            pm.Bernoulli("y", logit_p=logit_p, observed=np.asarray(y, dtype=int))
        return model

    def fit(self, X, y, log_likelihood: bool = False):
        import pymc as pm

        self.model_ = self.build(X, y)
        with self.model_:
            self.idata_ = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains, cores=self.cores,
                                    target_accept=self.target_accept, random_seed=self.seed,
                                    progressbar=self.progressbar, compute_convergence_checks=False,
                                    idata_kwargs={"log_likelihood": log_likelihood})
        return self

    # -- posterior access ---------------------------------------------------
    def posterior_samples(self):
        post = self.idata_.posterior
        b0 = post["intercept"].values.reshape(-1)
        beta = post["beta"].values.reshape(len(b0), -1)
        return b0, beta

    def prob_samples(self, X, max_draws: int | None = None) -> np.ndarray:
        """Posterior samples of P(churn) for each row: shape (S, N)."""
        b0, beta = self.posterior_samples()
        if max_draws is not None and len(b0) > max_draws:
            idx = np.linspace(0, len(b0) - 1, max_draws).astype(int)
            b0, beta = b0[idx], beta[idx]
        return sigmoid(b0[:, None] + beta @ np.asarray(X, dtype=float).T)

    def predict_proba(self, X) -> np.ndarray:
        """Posterior predictive mean of P(churn) (sklearn-style 2-column output)."""
        p = self.prob_samples(X).mean(axis=0)
        return np.column_stack([1 - p, p])

    def predict_summary(self, X, prob: float = 0.94) -> dict:
        ps = self.prob_samples(X)
        lo, hi = hdi(ps, prob)
        return {"mean": ps.mean(0), "std": ps.std(0), "hdi_low": lo, "hdi_high": hi}

    def posterior_predictive_rate(self, X, rng=None) -> np.ndarray:
        """Simulated churn rate on ``X`` for each posterior draw (posterior predictive check)."""
        rng = np.random.default_rng(self.seed) if rng is None else rng
        ps = self.prob_samples(X, max_draws=2000)
        return rng.binomial(1, ps).mean(axis=1)

    def diagnostics(self) -> dict:
        import arviz as az

        summ = az.summary(self.idata_, var_names=["intercept", "beta"], round_to=6)
        ss = self.idata_.sample_stats
        bfmi = az.bfmi(self.idata_)
        return {
            "max_rhat": float(summ["r_hat"].max()),
            "min_ess_bulk": float(summ["ess_bulk"].min()),
            "min_ess_tail": float(summ["ess_tail"].min()),
            "divergences": int(ss["diverging"].values.sum()),
            "bfmi_min": float(np.min(bfmi)),
            "mean_tree_depth": float(ss["tree_depth"].values.mean()),
            "mean_step_size": float(ss["step_size"].values.mean()),
            "sampling_time_s": float(self.idata_.posterior.attrs.get("sampling_time", np.nan)),
            "chains": self.chains, "draws_per_chain": self.draws, "tune": self.tune,
            "total_draws": self.chains * self.draws,
        }
