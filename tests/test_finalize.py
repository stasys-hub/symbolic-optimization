"""Tests for baseline scoring, holdout evaluation, and environment recording."""

import json

from symbolic_optimization import finalize, track_optuna
from symbolic_optimization.trials import TrialRecord, append_record, load_records

LOGISTIC_PARAMS = {
    "estimator": "logistic",
    "use_interactions": False,
    "C": 1.0,
    "penalty": "l2",
    "class_weight": "none",
}


def write_agent_record(path) -> None:
    append_record(
        path,
        TrialRecord(
            arm="agent",
            trial_index=0,
            params=LOGISTIC_PARAMS,
            fold_scores=[0.5],
            mean_score=0.5,
            std_score=0.0,
            wall_clock_s=0.1,
            cumulative_s=0.1,
            timestamp="2026-08-31T12:00:00+00:00",
        ),
    )


def test_finalize_writes_all_outputs(tmp_path, small_frame) -> None:
    track_optuna.run(seed=0, n_trials=2, frame=small_frame, out_dir=tmp_path)
    write_agent_record(tmp_path / "agent.jsonl")
    paths = finalize.run(tmp_path, frame=small_frame)
    assert [path.name for path in paths] == [
        "baselines.jsonl",
        "holdout.json",
        "environment.json",
    ]

    baselines = load_records(tmp_path / "baselines.jsonl")
    assert {record.arm for record in baselines} == {
        "baseline_default_logistic",
        "baseline_default_forest",
        "ground_truth_rules",
    }
    assert all(0.0 <= record.mean_score <= 1.0 for record in baselines)

    entries = json.loads((tmp_path / "holdout.json").read_text(encoding="utf-8"))
    arms = {entry["arm"] for entry in entries}
    assert {
        "optuna_seed0",
        "agent",
        "baseline_default_logistic",
        "baseline_default_forest",
        "ground_truth_rules",
    } == arms
    for entry in entries:
        assert 0.0 <= entry["holdout_ap"] <= 1.0

    environment = json.loads(
        (tmp_path / "environment.json").read_text(encoding="utf-8")
    )
    assert environment["threads"] == 1
    assert environment["python"].startswith("3.12")


def make_pysr_record(index: int, complexity: int, score: float, holdout: float):
    return TrialRecord(
        arm="pysr",
        trial_index=index,
        params={"complexity": complexity, "expression": f"expr_{complexity}"},
        fold_scores=[score],
        mean_score=score,
        std_score=0.0,
        wall_clock_s=0.1,
        cumulative_s=0.1,
        timestamp="2026-08-31T12:00:00+00:00",
        payload={"holdout_ap": holdout},
    )


def test_best_pysr_entry_requires_full_fold_coverage() -> None:
    records = [
        make_pysr_record(0, 5, 0.60, 0.58),
        make_pysr_record(1, 5, 0.62, 0.60),
        make_pysr_record(2, 7, 0.90, 0.88),
    ]
    entry = finalize.best_pysr_entry(records)
    assert entry["best_params"]["complexity"] == 5
    assert entry["cv_mean"] == 0.61
    assert entry["holdout_ap"] == 0.59
    assert entry["folds_present"] == 2
