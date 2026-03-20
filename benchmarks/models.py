"""Model factory for benchmark algorithms.

Models that need feature scaling (SVM, MLP, ELM, BayesianRidge) are
automatically wrapped in a sklearn Pipeline with StandardScaler.
"""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from benchmarks.elm import ELMClassifier, ELMRegressor


def _scaled(model: object) -> Pipeline:
    """Wrap a model with StandardScaler in a pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def get_classification_models(
    random_state: int = 42,
    include_tabpfn: bool = True,
) -> dict:
    """Return dict of {name: estimator} for classification benchmark."""
    models = {}

    if include_tabpfn:
        from tabpfn import TabPFNClassifier
        models["TabPFN"] = TabPFNClassifier(n_estimators=8)

    models.update({
        "XGBoost": XGBClassifier(
            random_state=random_state, eval_metric="logloss", verbosity=0,
        ),
        "LightGBM": LGBMClassifier(random_state=random_state, verbose=-1),
        "MLP": _scaled(MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=500, random_state=random_state,
        )),
        "RF": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "SVM": _scaled(SVC(probability=True, random_state=random_state)),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "ELM": _scaled(ELMClassifier(n_hidden=100, random_state=random_state)),
        "GBC": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state,
        ),
    })
    return models


def get_regression_models(
    random_state: int = 42,
    include_tabpfn: bool = True,
) -> dict:
    """Return dict of {name: estimator} for regression benchmark."""
    models = {}

    if include_tabpfn:
        from tabpfn import TabPFNRegressor
        models["TabPFN"] = TabPFNRegressor(n_estimators=8)

    models.update({
        "XGBoost": XGBRegressor(random_state=random_state, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=random_state, verbose=-1),
        "MLP": _scaled(MLPRegressor(
            hidden_layer_sizes=(100,), max_iter=500, random_state=random_state,
        )),
        "RF": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "SVR": _scaled(SVR()),
        "DT": DecisionTreeRegressor(random_state=random_state),
        "ELM": _scaled(ELMRegressor(n_hidden=100, random_state=random_state)),
        "BayesianRidge": _scaled(BayesianRidge()),
        "GBR": GradientBoostingRegressor(
            n_estimators=100, random_state=random_state,
        ),
    })
    return models
