"""Extreme Learning Machine (ELM) with sklearn-compatible API.

Single hidden-layer feedforward network with random input weights
and analytically solved output weights via Moore-Penrose pseudoinverse.

References:
    Huang, G.-B., Zhu, Q.-Y., & Siew, C.-K. (2006).
    Extreme learning machine: Theory and applications.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class ELMClassifier(BaseEstimator, ClassifierMixin):
    """ELM for classification tasks.

    Args:
        n_hidden: Number of hidden neurons.
        activation: Activation function ('tanh', 'sigmoid', or 'relu').
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_hidden: int = 100,
        activation: str = "tanh",
        random_state: int | None = None,
    ) -> None:
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state

    def _activate(self, X: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(X)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
        if self.activation == "relu":
            return np.maximum(0, X)
        msg = f"Unknown activation: {self.activation}"
        raise ValueError(msg)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ELMClassifier:
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Encode targets
        if n_classes == 2:
            Y = (y == self.classes_[1]).astype(float).reshape(-1, 1)
        else:
            Y = np.zeros((len(y), n_classes))
            for i, c in enumerate(self.classes_):
                Y[y == c, i] = 1.0

        # Random hidden layer
        self.W_ = rng.randn(X.shape[1], self.n_hidden)
        self.b_ = rng.randn(self.n_hidden)
        H = self._activate(X @ self.W_ + self.b_)

        # Solve output weights analytically
        self.beta_ = np.linalg.pinv(H) @ Y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = self._activate(X @ self.W_ + self.b_)
        output = H @ self.beta_
        if len(self.classes_) == 2:
            return self.classes_[(output.ravel() > 0.5).astype(int)]
        return self.classes_[np.argmax(output, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        H = self._activate(X @ self.W_ + self.b_)
        output = H @ self.beta_
        if len(self.classes_) == 2:
            p1 = 1.0 / (1.0 + np.exp(-np.clip(output.ravel(), -500, 500)))
            return np.column_stack([1 - p1, p1])
        # Softmax
        exp_out = np.exp(output - np.max(output, axis=1, keepdims=True))
        return exp_out / exp_out.sum(axis=1, keepdims=True)


class ELMRegressor(BaseEstimator, RegressorMixin):
    """ELM for regression tasks.

    Args:
        n_hidden: Number of hidden neurons.
        activation: Activation function ('tanh', 'sigmoid', or 'relu').
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_hidden: int = 100,
        activation: str = "tanh",
        random_state: int | None = None,
    ) -> None:
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state

    def _activate(self, X: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(X)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
        if self.activation == "relu":
            return np.maximum(0, X)
        msg = f"Unknown activation: {self.activation}"
        raise ValueError(msg)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ELMRegressor:
        rng = np.random.RandomState(self.random_state)
        self.W_ = rng.randn(X.shape[1], self.n_hidden)
        self.b_ = rng.randn(self.n_hidden)
        H = self._activate(X @ self.W_ + self.b_)
        self.beta_ = np.linalg.pinv(H) @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = self._activate(X @ self.W_ + self.b_)
        return H @ self.beta_
