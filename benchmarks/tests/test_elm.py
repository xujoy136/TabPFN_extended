from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split

from benchmarks.elm import ELMClassifier, ELMRegressor


class TestELMClassifier:
    def test_fit_predict_binary(self):
        """Binary classification on linearly separable data."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf = ELMClassifier(n_hidden=50, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(preds).issubset({0, 1})
        acc = np.mean(preds == y_test)
        assert acc > 0.7

    def test_fit_predict_multiclass(self):
        """Multiclass classification on iris."""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf = ELMClassifier(n_hidden=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(preds).issubset(set(y))

    def test_predict_proba_shape(self):
        """predict_proba returns correct shape and sums to 1."""
        X, y = load_iris(return_X_y=True)
        clf = ELMClassifier(n_hidden=50, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_classes_attribute(self):
        """After fit, classes_ is set correctly."""
        X = np.random.randn(50, 3)
        y = np.array([0, 1, 2] * 16 + [0, 1])
        clf = ELMClassifier(random_state=42)
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_reproducibility(self):
        """Same random_state produces same predictions."""
        X, y = load_iris(return_X_y=True)
        clf1 = ELMClassifier(n_hidden=50, random_state=42)
        clf1.fit(X, y)
        clf2 = ELMClassifier(n_hidden=50, random_state=42)
        clf2.fit(X, y)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))


class TestELMRegressor:
    def test_fit_predict(self):
        """Regression on synthetic data."""
        X, y = make_regression(
            n_samples=100, n_features=5, noise=0.1, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        reg = ELMRegressor(n_hidden=100, random_state=42)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_output_is_float(self):
        """Regression output should be float, not int."""
        X, y = make_regression(n_samples=50, n_features=3, random_state=42)
        reg = ELMRegressor(random_state=42)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.dtype in (np.float64, np.float32)

    def test_reproducibility(self):
        """Same random_state produces same predictions."""
        X, y = make_regression(n_samples=50, n_features=3, random_state=42)
        reg1 = ELMRegressor(n_hidden=50, random_state=42)
        reg1.fit(X, y)
        reg2 = ELMRegressor(n_hidden=50, random_state=42)
        reg2.fit(X, y)
        np.testing.assert_array_equal(reg1.predict(X), reg2.predict(X))
