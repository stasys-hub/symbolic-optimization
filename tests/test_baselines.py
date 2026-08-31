"""Tests for the reference configurations and the ground-truth rule set."""

import pandas as pd

from symbolic_optimization import baselines


def test_deterministic_rules_match_indicator_columns(frame: pd.DataFrame) -> None:
    hdf = baselines.heat_dissipation_failure(frame) == frame["HDF"].astype(bool)
    pwf = baselines.power_failure(frame) == frame["PWF"].astype(bool)
    osf = baselines.overstrain_failure(frame) == frame["OSF"].astype(bool)
    assert bool(hdf.all())
    assert bool(pwf.all())
    assert bool(osf.all())


def test_rule_failures_are_subset_of_machine_failure(frame: pd.DataFrame) -> None:
    rule = baselines.ground_truth_failure(frame)
    assert bool((rule <= frame["Machine failure"]).all())


def test_rule_positive_counts(frame: pd.DataFrame) -> None:
    rule = baselines.ground_truth_failure(frame)
    assert int(rule.sum()) == 287


def test_baselines_are_constructible() -> None:
    logistic = baselines.default_logistic()
    forest = baselines.default_random_forest(seed=0)
    assert logistic.steps[-1][0] == "logistic"
    assert forest.random_state == 0
