"""Tests for shared cross-validation scoring."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from symbolic_optimization.baselines import default_random_forest
from symbolic_optimization.evaluation import cv_average_precision


def test_cv_average_precision_on_synthetic_data() -> None:
    x, y = make_classification(n_samples=300, n_features=8, random_state=0)
    scores = cv_average_precision(
        default_random_forest(seed=0),
        pd.DataFrame(x),
        pd.Series(y),
        StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    )
    assert len(scores.fold_scores) == 5
    assert 0.0 <= min(scores.fold_scores)
    assert max(scores.fold_scores) <= 1.0
    assert scores.mean == pytest.approx(float(np.mean(scores.fold_scores)))
    assert scores.std == pytest.approx(float(np.std(scores.fold_scores)))
