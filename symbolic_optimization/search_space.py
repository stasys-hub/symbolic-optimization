"""Shared hyperparameter search space for the Optuna and agent arms."""

from typing import Annotated, Any, Literal

from optuna.trial import Trial
from pydantic import BaseModel, Field, TypeAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class LogisticParams(BaseModel):
    """Parameters of the logistic branch.

    Attributes:
        estimator: Branch selector, always ``logistic``.
        use_interactions: Add interaction-only polynomial features.
        C: Inverse regularization strength.
        penalty: Regularization norm.
        class_weight: Class weighting mode.
    """

    estimator: Literal["logistic"]
    use_interactions: bool
    C: float = Field(gt=0.0, le=100.0)
    penalty: Literal["l1", "l2"]
    class_weight: Literal["none", "balanced", "balanced_subsample"]


class RandomForestParams(BaseModel):
    """Parameters of the random forest branch.

    Attributes:
        estimator: Branch selector, always ``random_forest``.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf.
        max_features: Feature subsampling strategy.
        class_weight: Class weighting mode.
    """

    estimator: Literal["random_forest"]
    n_estimators: int = Field(ge=50, le=400)
    max_depth: int = Field(ge=3, le=20)
    min_samples_leaf: int = Field(ge=1, le=20)
    max_features: Literal["sqrt", "log2", "all"]
    class_weight: Literal["none", "balanced", "balanced_subsample"]


SearchParams = Annotated[
    LogisticParams | RandomForestParams, Field(discriminator="estimator")
]

_PARAMS_ADAPTER: TypeAdapter[SearchParams] = TypeAdapter(SearchParams)


def validate_params(data: dict[str, Any]) -> Any:
    """Validate a parameter dictionary against the search space.

    Args:
        data: Raw parameter dictionary.

    Returns:
        The validated logistic or random forest parameter model.

    Raises:
        ValidationError: If the parameters are outside the space.
    """
    return _PARAMS_ADAPTER.validate_python(data)


def suggest_params(trial: Trial) -> dict[str, Any]:
    """Sample parameters from the shared space through an Optuna trial.

    Args:
        trial: Active Optuna trial.

    Returns:
        Sampled parameter dictionary for one branch.
    """
    estimator = trial.suggest_categorical("estimator", ["logistic", "random_forest"])
    if estimator == "logistic":
        return {
            "estimator": estimator,
            "use_interactions": trial.suggest_categorical(
                "use_interactions", [True, False]
            ),
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["none", "balanced", "balanced_subsample"]
            ),
        }
    return {
        "estimator": estimator,
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", "all"]
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["none", "balanced", "balanced_subsample"]
        ),
    }


def build_estimator(params: dict[str, Any], seed: int) -> Any:
    """Construct a scikit-learn estimator from validated parameters.

    Args:
        params: Parameter dictionary within the shared space.
        seed: Random seed passed to the estimator.

    Returns:
        A fitted-elsewhere pipeline or forest with single-threaded settings.

    Raises:
        ValidationError: If the parameters are outside the space.
    """
    valid = validate_params(params)
    if isinstance(valid, LogisticParams):
        steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
        if valid.use_interactions:
            steps.append(
                (
                    "interactions",
                    PolynomialFeatures(interaction_only=True, include_bias=False),
                )
            )
        steps.append(
            (
                "logistic",
                LogisticRegression(
                    C=valid.C,
                    penalty=valid.penalty,
                    solver="liblinear",
                    class_weight=None if valid.class_weight == "none" else "balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            )
        )
        return Pipeline(steps)
    return RandomForestClassifier(
        n_estimators=valid.n_estimators,
        max_depth=valid.max_depth,
        min_samples_leaf=valid.min_samples_leaf,
        max_features=1.0 if valid.max_features == "all" else valid.max_features,
        class_weight=None if valid.class_weight == "none" else valid.class_weight,
        random_state=seed,
        n_jobs=1,
    )


def space_description() -> str:
    """Describe the shared search space for language model prompts.

    Returns:
        Plain-text description of both branches and all parameter ranges.
    """
    return (
        "Search space. Select one of two estimator branches.\n"
        "Branch logistic (pipeline: StandardScaler, optional interaction-only"
        " PolynomialFeatures, LogisticRegression with liblinear solver):\n"
        "- use_interactions: true or false\n"
        "- C: float, log scale, 0.001 to 100\n"
        "- penalty: l1 or l2\n"
        "- class_weight: none, balanced, or balanced_subsample"
        " (balanced_subsample behaves as balanced on this branch)\n"
        "Branch random_forest (RandomForestClassifier):\n"
        "- n_estimators: int, 50 to 400\n"
        "- max_depth: int, 3 to 20\n"
        "- min_samples_leaf: int, 1 to 20\n"
        "- max_features: sqrt, log2, or all\n"
        "- class_weight: none, balanced, or balanced_subsample"
    )
