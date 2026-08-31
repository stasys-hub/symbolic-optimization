"""Tests for the agent track."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from symbolic_optimization import track_agent
from symbolic_optimization.data import make_features, make_target
from symbolic_optimization.search_space import LogisticParams, RandomForestParams
from symbolic_optimization.track_agent import LoopState, Proposal, build_prompt
from symbolic_optimization.trials import TrialRecord, load_records


def make_proposal(index: int) -> Proposal:
    if index % 2 == 0:
        params = LogisticParams(
            estimator="logistic",
            use_interactions=False,
            C=1.0,
            penalty="l2",
            class_weight="none",
        )
    else:
        params = RandomForestParams(
            estimator="random_forest",
            n_estimators=60,
            max_depth=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="none",
        )
    return Proposal(params=params, rationale=f"rationale {index}")


def test_loop_writes_one_record_per_evaluation(
    tmp_path, small_frame, monkeypatch
) -> None:
    queue = [make_proposal(index) for index in range(3)]

    async def fake_request(agent, prompt):
        return (
            queue.pop(0),
            SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                output_reasoning_tokens=2,
                requests=1,
            ),
            0.5,
        )

    monkeypatch.setattr(track_agent, "request_proposal", fake_request)
    path = track_agent.run(
        n_evaluations=3,
        frame=small_frame,
        out_dir=tmp_path,
        agent=object(),
        model_id="stub",
    )
    records = load_records(path)
    assert [record.trial_index for record in records] == [0, 1, 2]
    assert all(record.arm == "agent" for record in records)
    assert records[0].params["estimator"] == "logistic"
    assert records[1].params["estimator"] == "random_forest"
    assert records[0].payload["rationale"] == "rationale 0"
    assert records[0].payload["usage"]["input_tokens"] == 10
    assert records[0].payload["model"] == "stub"
    assert records[0].payload["latency_s"] == 0.5


def make_state(small_frame: pd.DataFrame) -> LoopState:
    return LoopState(
        records=[],
        features=make_features(small_frame),
        target=make_target(small_frame),
        splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        log_path=Path("unused.jsonl"),
        seed=0,
        n_evaluations=50,
        start=0.0,
    )


def test_graph_renders_node_names() -> None:
    diagram = track_agent.build_graph().render()
    for name in ("Propose", "Evaluate", "Decide"):
        assert name in diagram


def test_prompt_reports_empty_history_and_budget(small_frame) -> None:
    prompt = build_prompt(make_state(small_frame))
    assert "No evaluations yet." in prompt
    assert "Evaluations remaining: 50" in prompt
    assert "n_estimators" in prompt
    assert "C: float" in prompt


def test_prompt_includes_history(small_frame) -> None:
    state = make_state(small_frame)
    state.records.append(
        TrialRecord(
            arm="agent",
            trial_index=0,
            params={"estimator": "logistic", "C": 1.0},
            fold_scores=[0.5],
            mean_score=0.5,
            std_score=0.0,
            wall_clock_s=1.0,
            cumulative_s=1.0,
            timestamp="2026-08-31T12:00:00+00:00",
        )
    )
    prompt = build_prompt(state)
    assert "t00" in prompt
    assert "AP=0.5000" in prompt
    assert "Evaluations remaining: 49" in prompt
