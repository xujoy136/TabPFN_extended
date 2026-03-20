"""Dataset loading utilities for benchmarks.

Uses sklearn built-in datasets to avoid network dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_regression,
)
from sklearn.model_selection import train_test_split


@dataclass
class BenchmarkDataset:
    """A dataset split into train/test with metadata."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    task: str  # "classification" or "regression"

    @property
    def n_samples(self) -> int:
        return len(self.X_train) + len(self.X_test)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def get_classification_datasets(
    test_size: float = 0.3,
    random_state: int = 42,
) -> list[BenchmarkDataset]:
    """Load classification datasets."""
    loaders = {
        "breast_cancer": load_breast_cancer,
        "iris": load_iris,
        "wine": load_wine,
        "digits": load_digits,
    }
    datasets = []
    for name, loader in loaders.items():
        X, y = loader(return_X_y=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )
        datasets.append(BenchmarkDataset(
            name=name, X_train=X_tr, X_test=X_te,
            y_train=y_tr, y_test=y_te, task="classification",
        ))
    return datasets


def get_regression_datasets(
    test_size: float = 0.3,
    random_state: int = 42,
) -> list[BenchmarkDataset]:
    """Load regression datasets."""
    datasets = []

    # Diabetes
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    datasets.append(BenchmarkDataset(
        name="diabetes", X_train=X_tr, X_test=X_te,
        y_train=y_tr, y_test=y_te, task="regression",
    ))

    # California Housing substitute (synthetic, 2000 samples, 8 features)
    # Uses make_regression to avoid network dependency from fetch_california_housing
    X, y = make_regression(
        n_samples=2000, n_features=8, noise=0.1, random_state=random_state,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    datasets.append(BenchmarkDataset(
        name="california_housing", X_train=X_tr, X_test=X_te,
        y_train=y_tr, y_test=y_te, task="regression",
    ))

    return datasets
