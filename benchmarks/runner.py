"""Core benchmark runner logic."""

from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from benchmarks.datasets import BenchmarkDataset


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute ROC AUC, handling multiclass and edge cases."""
    try:
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro",
        ))
    except (ValueError, IndexError):
        return float("nan")


def run_single(
    name: str,
    model: object,
    dataset: BenchmarkDataset,
) -> dict:
    """Run a single (model, dataset) benchmark.

    Returns dict with model name, dataset name, metrics, and timing.
    """
    result = {"model": name, "dataset": dataset.name}

    try:
        # Fit
        t0 = time.perf_counter()
        model.fit(dataset.X_train, dataset.y_train)
        result["fit_time_s"] = time.perf_counter() - t0

        # Predict
        t0 = time.perf_counter()
        preds = model.predict(dataset.X_test)
        result["predict_time_s"] = time.perf_counter() - t0

        if dataset.task == "classification":
            result["accuracy"] = float(accuracy_score(dataset.y_test, preds))
            result["f1_macro"] = float(f1_score(
                dataset.y_test, preds, average="macro", zero_division=0,
            ))
            result["balanced_accuracy"] = float(
                balanced_accuracy_score(dataset.y_test, preds)
            )
            # ROC AUC (needs predict_proba)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(dataset.X_test)
                result["roc_auc"] = _safe_roc_auc(dataset.y_test, proba)
            else:
                result["roc_auc"] = float("nan")

        elif dataset.task == "regression":
            result["mse"] = float(mean_squared_error(dataset.y_test, preds))
            result["rmse"] = float(np.sqrt(
                mean_squared_error(dataset.y_test, preds)
            ))
            result["mae"] = float(mean_absolute_error(dataset.y_test, preds))
            result["r2"] = float(r2_score(dataset.y_test, preds))

        result["error"] = None

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


def run_benchmark(
    models: dict,
    datasets: list[BenchmarkDataset],
) -> pd.DataFrame:
    """Run all (model, dataset) combinations and return results DataFrame."""
    results = []
    total = len(models) * len(datasets)

    for i, (name, model) in enumerate(models.items()):
        for j, dataset in enumerate(datasets):
            idx = i * len(datasets) + j + 1
            print(f"[{idx}/{total}] {name} on {dataset.name}...", flush=True)
            result = run_single(name, model, dataset)

            # Print key metric
            if dataset.task == "classification" and "accuracy" in result:
                print(f"  -> accuracy={result['accuracy']:.4f}")
            elif dataset.task == "regression" and "r2" in result:
                print(f"  -> R²={result['r2']:.4f}")

            results.append(result)

    return pd.DataFrame(results)
