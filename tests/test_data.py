"""Tests for dataset handling and split construction."""

import pandas as pd
import pytest

from symbolic_optimization import data


def test_features_exclude_labels_and_identifiers(frame: pd.DataFrame) -> None:
    features = data.make_features(frame)
    banned = set(data.LABEL_COLUMNS + data.IDENTIFIER_COLUMNS)
    assert banned.isdisjoint(features.columns)
    expected = set(data.SENSOR_COLUMNS) | {"type_L", "type_M", "type_H"}
    assert set(features.columns) == expected


def test_variant_one_hot_sums_to_one(frame: pd.DataFrame) -> None:
    variant_columns = data.make_features(frame)[["type_L", "type_M", "type_H"]]
    assert variant_columns.sum(axis=1).eq(1.0).all()


def test_target_is_binary_with_known_positive_count(
    frame: pd.DataFrame,
) -> None:
    target = data.make_target(frame)
    assert set(target.unique()) == {0, 1}
    assert int(target.sum()) == 339


def test_holdout_is_disjoint_and_stratified(frame: pd.DataFrame) -> None:
    train_idx, test_idx = data.holdout_indices(frame, seed=0)
    assert len(train_idx) == 8000
    assert len(test_idx) == 2000
    assert set(train_idx).isdisjoint(test_idx)
    target = data.make_target(frame)
    train_rate = float(target.iloc[train_idx].mean())
    test_rate = float(target.iloc[test_idx].mean())
    assert test_rate == pytest.approx(train_rate, abs=0.005)


def test_cv_splitter_covers_all_rows_once(frame: pd.DataFrame) -> None:
    train_idx, _ = data.holdout_indices(frame, seed=0)
    target = data.make_target(frame).iloc[train_idx]
    splitter = data.cv_splitter(seed=0)
    seen: list[int] = []
    for fit_idx, score_idx in splitter.split(train_idx, target):
        assert len(fit_idx) == 6400
        seen.extend(score_idx.tolist())
    assert sorted(seen) == list(range(8000))
