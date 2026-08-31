"""Tests for the Optuna track."""

from symbolic_optimization.track_optuna import run
from symbolic_optimization.trials import load_records


def test_smoke_run_writes_one_record_per_trial(tmp_path, small_frame) -> None:
    path = run(seed=0, n_trials=2, frame=small_frame, out_dir=tmp_path)
    records = load_records(path)
    assert [record.trial_index for record in records] == [0, 1]
    assert all(record.arm == "optuna" for record in records)
    assert all(0.0 <= record.mean_score <= 1.0 for record in records)
    assert all(
        record.params["estimator"] in ("logistic", "random_forest")
        for record in records
    )
    assert all(record.cumulative_s >= record.wall_clock_s for record in records)
