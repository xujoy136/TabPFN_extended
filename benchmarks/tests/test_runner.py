from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.runner import run_single, run_benchmark
from benchmarks.datasets import BenchmarkDataset


@pytest.fixture
def tiny_clf_dataset():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 3)
    y = (X[:, 0] > 0).astype(int)
    return BenchmarkDataset(
        name="tiny", X_train=X[:30], X_test=X[30:],
        y_train=y[:30], y_test=y[30:], task="classification",
    )


@pytest.fixture
def tiny_reg_dataset():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 3)
    y = X[:, 0] * 2 + rng.randn(40) * 0.1
    return BenchmarkDataset(
        name="tiny", X_train=X[:30], X_test=X[30:],
        y_train=y[:30], y_test=y[30:], task="regression",
    )


class TestRunSingle:
    def test_classification_result_keys(self, tiny_clf_dataset):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        result = run_single("DT", model, tiny_clf_dataset)
        assert result["model"] == "DT"
        assert result["dataset"] == "tiny"
        assert "accuracy" in result
        assert "f1_macro" in result
        assert "fit_time_s" in result
        assert "predict_time_s" in result

    def test_regression_result_keys(self, tiny_reg_dataset):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=42)
        result = run_single("DT", model, tiny_reg_dataset)
        assert result["model"] == "DT"
        assert "mse" in result
        assert "rmse" in result
        assert "mae" in result
        assert "r2" in result

    def test_fit_time_is_positive(self, tiny_clf_dataset):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        result = run_single("DT", model, tiny_clf_dataset)
        assert result["fit_time_s"] >= 0
        assert result["predict_time_s"] >= 0


class TestRunBenchmark:
    def test_returns_dataframe(self, tiny_clf_dataset):
        from sklearn.tree import DecisionTreeClassifier
        models = {"DT": DecisionTreeClassifier(random_state=42)}
        df = run_benchmark(models, [tiny_clf_dataset])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["model"] == "DT"
