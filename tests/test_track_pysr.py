"""Tests for the PySR track."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from symbolic_optimization.track_pysr import balanced_sample_weights, run
from symbolic_optimization.trials import load_records


def test_balanced_sample_weights_equalize_classes() -> None:
    target = pd.Series([0] * 90 + [1] * 10)
    weights = balanced_sample_weights(target)
    assert weights.shape == (100,)
    negative_total = float(weights[target == 0].sum())
    positive_total = float(weights[target == 1].sum())
    assert positive_total == pytest.approx(negative_total)
    assert weights[0] == pytest.approx(100 / 180.0)
    assert weights[90] == pytest.approx(100 / 20.0)


def test_smoke_run_logs_hall_of_fame_candidates(tmp_path, small_frame) -> None:
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    path = run(
        n_iterations=2,
        max_complexity=10,
        frame=small_frame,
        out_dir=tmp_path,
        splitter=splitter,
    )
    records = load_records(path)
    assert len(records) > 0
    assert all(record.arm == "pysr" for record in records)
    assert all(0.0 <= record.mean_score <= 1.0 for record in records)
    assert all("expression" in record.params for record in records)
    assert all("complexity" in record.params for record in records)
    assert all("fold" in record.payload for record in records)
    assert all(np.isfinite(record.payload["loss"]) for record in records)
    folds = {record.payload["fold"] for record in records}
    assert folds == {0, 1}
