"""Reference configurations: model defaults and the ground-truth rule set."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEMP_DIFF_LIMIT_K = 8.6
LOW_RPM_LIMIT = 1380.0
POWER_LOW_W = 3500.0
POWER_HIGH_W = 9000.0
OSF_LIMIT_MIN_NM = {"L": 11000, "M": 12000, "H": 13000}
RPM_TO_RAD_S = 2.0 * np.pi / 60.0


def default_logistic() -> Pipeline:
    """Build the default logistic regression baseline.

    Returns:
        Unscaled-default scikit-learn pipeline with standardization.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000)),
        ]
    )


def default_random_forest(seed: int) -> RandomForestClassifier:
    """Build the default random forest baseline.

    Args:
        seed: Random seed for the estimator.

    Returns:
        Forest with scikit-learn defaults and a fixed seed.
    """
    return RandomForestClassifier(random_state=seed)


def process_power_w(frame: pd.DataFrame) -> pd.Series:
    """Compute process power from torque and rotational speed.

    Args:
        frame: Raw dataset.

    Returns:
        Power in watts for each row.
    """
    return frame["Torque [Nm]"] * frame["Rotational speed [rpm]"] * RPM_TO_RAD_S


def heat_dissipation_failure(frame: pd.DataFrame) -> pd.Series:
    """Apply the heat dissipation failure rule.

    Args:
        frame: Raw dataset.

    Returns:
        Boolean series, true where the temperature difference is below
        8.6 K and rotational speed is below 1380 rpm.
    """
    diff = frame["Process temperature [K]"] - frame["Air temperature [K]"]
    return (diff < TEMP_DIFF_LIMIT_K) & (
        frame["Rotational speed [rpm]"] < LOW_RPM_LIMIT
    )


def power_failure(frame: pd.DataFrame) -> pd.Series:
    """Apply the power failure rule.

    Args:
        frame: Raw dataset.

    Returns:
        Boolean series, true where process power is below 3500 W or above
        9000 W.
    """
    power = process_power_w(frame)
    return (power < POWER_LOW_W) | (power > POWER_HIGH_W)


def overstrain_failure(frame: pd.DataFrame) -> pd.Series:
    """Apply the overstrain failure rule.

    Args:
        frame: Raw dataset.

    Returns:
        Boolean series, true where tool wear times torque exceeds the
        variant-specific limit.
    """
    product = frame["Tool wear [min]"] * frame["Torque [Nm]"]
    limit = frame["Type"].map(OSF_LIMIT_MIN_NM)
    return product > limit


def ground_truth_failure(frame: pd.DataFrame) -> pd.Series:
    """Combine the three deterministic failure rules.

    Tool wear failure and random failure are stochastic and cannot be
    expressed as rules over the features, so they are not part of this
    baseline.

    Args:
        frame: Raw dataset.

    Returns:
        Integer series, one where any deterministic failure rule fires.
    """
    combined = (
        heat_dissipation_failure(frame)
        | power_failure(frame)
        | overstrain_failure(frame)
    )
    return combined.astype(int)
