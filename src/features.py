"""Train/test splitting, z-score standardisation and PCA (fitted on the training split only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class FeaturePipeline:
    """z = (x - mu) / sigma, then PCA keeping the fewest components with >= ``variance`` explained variance.

    ``variance=None`` disables PCA (standardised features are returned).
    """
    variance: float | None = 0.95
    seed: int = 42

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        self.pca_ = None
        if self.variance is not None:
            self.pca_ = PCA(n_components=self.variance, svd_solver="full", random_state=self.seed)
            self.pca_.fit(self.scaler_.transform(X))
        return self

    def transform(self, X):
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.pca_.transform(Z) if self.pca_ is not None else Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def n_components(self) -> int:
        return self.pca_.n_components_ if self.pca_ is not None else len(self.scaler_.mean_)

    @property
    def explained_variance(self) -> float:
        return float(self.pca_.explained_variance_ratio_.sum()) if self.pca_ is not None else 1.0

    def component_names(self) -> list[str]:
        return [f"PC{i + 1}" for i in range(self.n_components)]


def split(X, y, test_size: float = 0.2, seed: int = 42):
    """Stratified 80/20 split (5,625 train / 1,407 test on the cleaned data)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
