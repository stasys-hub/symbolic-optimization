"""Dataset loading, feature selection, and split construction."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
LABEL_COLUMNS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET_COLUMN = "Machine failure"
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
HOLDOUT_FRACTION = 0.2
N_FOLDS = 5


def load_dataset(path: str | Path = "ai4i2020.csv") -> pd.DataFrame:
    """Load the AI4I 2020 predictive maintenance dataset.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw dataset with validated columns.

    Raises:
        ValueError: If required columns are missing.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = SENSOR_COLUMNS + LABEL_COLUMNS + ["Type"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    return frame


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix without label or identifier columns.

    Args:
        frame: Raw dataset.

    Returns:
        Float frame with five sensor columns and one-hot encoded product
        variant.
    """
    features = frame[SENSOR_COLUMNS].astype(float)
    variant = pd.get_dummies(frame["Type"], prefix="type", dtype=float)
    return pd.concat([features, variant], axis=1)


def make_target(frame: pd.DataFrame) -> pd.Series:
    """Extract the binary failure target.

    Args:
        frame: Raw dataset.

    Returns:
        Integer series with the machine failure label.
    """
    return frame[TARGET_COLUMN].astype(int)


def holdout_indices(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions into a search part and an untouched holdout.

    Args:
        frame: Raw dataset.
        seed: Random seed for reproducibility.

    Returns:
        Train and test index arrays, stratified on the failure target.
    """
    target = make_target(frame)
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)),
        test_size=HOLDOUT_FRACTION,
        stratify=target,
        random_state=seed,
    )
    return train_idx, test_idx


def cv_splitter(seed: int) -> StratifiedKFold:
    """Build the cross-validation splitter shared by all arms.

    Args:
        seed: Random seed for fold shuffling.

    Returns:
        Stratified five-fold splitter.
    """
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
