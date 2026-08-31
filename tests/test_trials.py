"""Tests for the trial record schema and JSONL persistence."""

from symbolic_optimization.trials import TrialRecord, append_record, load_records


def make_record() -> TrialRecord:
    return TrialRecord(
        arm="optuna",
        trial_index=3,
        params={"estimator": "random_forest", "max_depth": 8},
        fold_scores=[0.5, 0.6, 0.7, 0.55, 0.65],
        mean_score=0.6,
        std_score=0.07,
        wall_clock_s=1.25,
        cumulative_s=12.5,
        timestamp="2026-08-31T12:00:00+00:00",
        payload={"rationale": "increase depth", "usage": {"requests": 1}},
    )


def test_record_round_trips_through_jsonl(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    append_record(path, make_record())
    append_record(path, make_record())
    records = load_records(path)
    assert len(records) == 2
    assert records[0] == make_record()
    assert records[1].trial_index == 3
    assert records[1].payload["usage"]["requests"] == 1


def test_record_defaults_to_empty_payload() -> None:
    record = TrialRecord(
        arm="pysr",
        trial_index=0,
        params={},
        fold_scores=[1.0],
        mean_score=1.0,
        std_score=0.0,
        wall_clock_s=0.0,
        cumulative_s=0.0,
        timestamp="2026-08-31T12:00:00+00:00",
    )
    assert record.payload == {}
