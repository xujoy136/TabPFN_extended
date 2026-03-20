from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from benchmarks.models import get_classification_models, get_regression_models


EXPECTED_CLF_NAMES = [
    "TabPFN", "XGBoost", "LightGBM", "MLP", "RF", "SVM", "DT", "ELM", "GBC",
]

EXPECTED_REG_NAMES = [
    "TabPFN", "XGBoost", "LightGBM", "MLP", "RF", "SVR", "DT", "ELM",
    "BayesianRidge", "GBR",
]


class TestClassificationModels:
    def test_returns_expected_models(self):
        models = get_classification_models()
        assert set(models.keys()) == set(EXPECTED_CLF_NAMES)

    def test_all_have_fit_predict(self):
        models = get_classification_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"

    def test_sklearn_models_fit_predict(self):
        """Quick smoke test: all non-TabPFN models fit and predict on tiny data."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 4)
        y = (X[:, 0] > 0).astype(int)
        models = get_classification_models(include_tabpfn=False)
        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)
            assert preds.shape == (30,), f"{name} wrong prediction shape"


class TestRegressionModels:
    def test_returns_expected_models(self):
        models = get_regression_models()
        assert set(models.keys()) == set(EXPECTED_REG_NAMES)

    def test_all_have_fit_predict(self):
        models = get_regression_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"

    def test_sklearn_models_fit_predict(self):
        """Quick smoke test: all non-TabPFN models fit and predict on tiny data."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 4)
        y = X[:, 0] * 2 + rng.randn(30) * 0.1
        models = get_regression_models(include_tabpfn=False)
        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)
            assert preds.shape == (30,), f"{name} wrong prediction shape"
