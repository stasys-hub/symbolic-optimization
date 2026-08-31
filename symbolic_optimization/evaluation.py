"""Shared cross-validation scoring."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import BaseCrossValidator


@dataclass(slots=True)
class CvScores:
    """Average precision scores over cross-validation folds.

    Attributes:
        fold_scores: Average precision per fold.
        mean: Mean of the fold scores.
        std: Standard deviation of the fold scores.
    """

    fold_scores: list[float]
    mean: float
    std: float


def cv_average_precision(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    splitter: BaseCrossValidator,
) -> CvScores:
    """Score an estimator with stratified cross-validation.

    Args:
        estimator: Scikit-learn classifier with ``predict_proba``; cloned
            per fold.
        features: Feature matrix.
        target: Binary target series.
        splitter: Cross-validation splitter.

    Returns:
        Fold, mean, and standard deviation of average precision.
    """
    scores: list[float] = []
    for fit_pos, val_pos in splitter.split(features, target):
        model = clone(estimator)
        model.fit(features.iloc[fit_pos], target.iloc[fit_pos])
        proba = model.predict_proba(features.iloc[val_pos])
        pos_col = int(np.flatnonzero(model.classes_ == 1)[0])
        score = average_precision_score(target.iloc[val_pos], proba[:, pos_col])
        scores.append(float(score))
    return CvScores(
        fold_scores=scores,
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
    )
