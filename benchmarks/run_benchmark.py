#!/usr/bin/env python
"""Run the full benchmark suite.

Usage (must run from repo root):
    # Run everything (classification + regression)
    python -m benchmarks.run_benchmark

    # Classification only
    python -m benchmarks.run_benchmark --task classification

    # Regression only
    python -m benchmarks.run_benchmark --task regression

    # Without TabPFN (e.g., no GPU or model not downloaded)
    python -m benchmarks.run_benchmark --no-tabpfn

    # Custom output directory
    python -m benchmarks.run_benchmark --output-dir my_results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `benchmarks` is importable
# when run as `python benchmarks/run_benchmark.py` instead of `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from benchmarks.datasets import get_classification_datasets, get_regression_datasets
from benchmarks.models import get_classification_models, get_regression_models
from benchmarks.runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="TabPFN Benchmark Suite")
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "both"],
        default="both",
        help="Which task to benchmark (default: both)",
    )
    parser.add_argument(
        "--no-tabpfn",
        action="store_true",
        help="Exclude TabPFN from benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to save CSV results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_tabpfn = not args.no_tabpfn

    if args.task in ("classification", "both"):
        print("=" * 60)
        print("CLASSIFICATION BENCHMARK")
        print("=" * 60)
        models = get_classification_models(
            random_state=args.random_state, include_tabpfn=include_tabpfn,
        )
        datasets = get_classification_datasets(random_state=args.random_state)
        df = run_benchmark(models, datasets)

        out_path = output_dir / "classification_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
        print("\n" + _format_summary(df, task="classification"))

    if args.task in ("regression", "both"):
        print("=" * 60)
        print("REGRESSION BENCHMARK")
        print("=" * 60)
        models = get_regression_models(
            random_state=args.random_state, include_tabpfn=include_tabpfn,
        )
        datasets = get_regression_datasets(random_state=args.random_state)
        df = run_benchmark(models, datasets)

        out_path = output_dir / "regression_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
        print("\n" + _format_summary(df, task="regression"))


def _format_summary(df: pd.DataFrame, task: str) -> str:
    """Format a summary table grouped by model, averaged across datasets."""
    if task == "classification":
        metric_cols = ["accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]
    else:
        metric_cols = ["mse", "rmse", "mae", "r2"]

    available = [c for c in metric_cols if c in df.columns]
    time_cols = [c for c in ["fit_time_s", "predict_time_s"] if c in df.columns]

    summary = df.groupby("model")[available + time_cols].mean()
    if task == "classification":
        summary = summary.sort_values("accuracy", ascending=False)
    else:
        summary = summary.sort_values("r2", ascending=False)

    return summary.to_string(float_format="%.4f")


if __name__ == "__main__":
    main()
