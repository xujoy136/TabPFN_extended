from __future__ import annotations

import numpy as np

from benchmarks.datasets import (
    BenchmarkDataset,
    get_classification_datasets,
    get_regression_datasets,
)


class TestClassificationDatasets:
    def test_returns_four_datasets(self):
        datasets = get_classification_datasets()
        assert len(datasets) == 4

    def test_expected_names(self):
        datasets = get_classification_datasets()
        names = {d.name for d in datasets}
        assert names == {"breast_cancer", "iris", "wine", "digits"}

    def test_all_are_classification(self):
        for d in get_classification_datasets():
            assert d.task == "classification"

    def test_train_test_no_overlap_in_size(self):
        for d in get_classification_datasets():
            total = d.n_samples
            assert len(d.X_train) + len(d.X_test) == total
            assert len(d.X_train) > len(d.X_test)  # test_size=0.3


class TestRegressionDatasets:
    def test_returns_two_datasets(self):
        datasets = get_regression_datasets()
        assert len(datasets) == 2

    def test_expected_names(self):
        datasets = get_regression_datasets()
        names = {d.name for d in datasets}
        assert names == {"diabetes", "california_housing"}

    def test_california_subsampled(self):
        datasets = get_regression_datasets()
        cal = [d for d in datasets if d.name == "california_housing"][0]
        assert cal.n_samples == 2000  # subsampled from 20640

    def test_all_are_regression(self):
        for d in get_regression_datasets():
            assert d.task == "regression"
