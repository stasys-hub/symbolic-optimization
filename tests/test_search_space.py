"""Tests for the shared search space."""

import optuna
import pytest
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from symbolic_optimization.search_space import (
    LogisticParams,
    RandomForestParams,
    build_estimator,
    space_description,
    suggest_params,
    validate_params,
)

LOGISTIC = {
    "estimator": "logistic",
    "use_interactions": True,
    "C": 1.0,
    "penalty": "l1",
    "class_weight": "balanced",
}
FOREST = {
    "estimator": "random_forest",
    "n_estimators": 60,
    "max_depth": 5,
    "min_samples_leaf": 2,
    "max_features": "all",
    "class_weight": "balanced_subsample",
}


def test_logistic_branch_builds_pipeline() -> None:
    pipeline = build_estimator(LOGISTIC, seed=1)
    assert isinstance(pipeline, Pipeline)
    names = [name for name, _ in pipeline.steps]
    assert names == ["scaler", "interactions", "logistic"]
    logistic = pipeline.named_steps["logistic"]
    assert logistic.l1_ratio == 1.0
    assert logistic.class_weight == "balanced"
    assert logistic.solver == "liblinear"


def test_logistic_branch_without_interactions() -> None:
    pipeline = build_estimator(LOGISTIC | {"use_interactions": False}, seed=1)
    names = [name for name, _ in pipeline.steps]
    assert names == ["scaler", "logistic"]


def test_logistic_branch_maps_subsample_to_balanced() -> None:
    pipeline = build_estimator(
        LOGISTIC | {"class_weight": "balanced_subsample"}, seed=1
    )
    assert pipeline.named_steps["logistic"].class_weight == "balanced"


def test_random_forest_branch_builds_forest() -> None:
    forest = build_estimator(FOREST, seed=1)
    assert isinstance(forest, RandomForestClassifier)
    assert forest.max_features == 1.0
    assert forest.n_jobs == 1
    assert forest.class_weight == "balanced_subsample"


def test_invalid_params_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_params(LOGISTIC | {"C": 1000.0})
    with pytest.raises(ValidationError):
        validate_params(FOREST | {"n_estimators": 10})
    with pytest.raises(ValidationError):
        validate_params({"estimator": "gradient_boosting"})


def test_space_description_lists_every_field() -> None:
    description = space_description()
    for name in LogisticParams.model_fields:
        assert name in description
    for name in RandomForestParams.model_fields:
        assert name in description


def test_suggest_params_on_fixed_trial() -> None:
    assert suggest_params(optuna.trial.FixedTrial(LOGISTIC)) == LOGISTIC
    assert suggest_params(optuna.trial.FixedTrial(FOREST)) == FOREST
