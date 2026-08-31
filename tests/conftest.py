"""Shared test fixtures."""

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from symbolic_optimization.data import load_dataset, make_target


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return load_dataset()


@pytest.fixture()
def small_frame(frame: pd.DataFrame) -> pd.DataFrame:
    small, _ = train_test_split(
        frame, train_size=600, stratify=make_target(frame), random_state=0
    )
    return small.reset_index(drop=True)
