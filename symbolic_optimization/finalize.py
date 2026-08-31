"""Baseline scoring, holdout evaluation, and environment recording."""

import argparse
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from symbolic_optimization.baselines import (
    default_logistic,
    default_random_forest,
    ground_truth_failure,
)
from symbolic_optimization.data import (
    cv_splitter,
    holdout_indices,
    load_dataset,
    make_features,
    make_target,
)
from symbolic_optimization.evaluation import cv_average_precision
from symbolic_optimization.search_space import build_estimator
from symbolic_optimization.trials import TrialRecord, append_record, load_records

DATA_SEED = 0
RULES_ARM = "ground_truth_rules"
TRACKED_PACKAGES = [
    "scikit-learn",
    "optuna",
    "pysr",
    "pydantic-ai",
    "pydantic-graph",
    "numpy",
    "pandas",
    "altair",
]


def rules_cv_scores(
    rules: pd.Series, features: pd.DataFrame, target: pd.Series
) -> list[float]:
    """Score the deterministic failure rules on the shared folds.

    Args:
        rules: Rule indicator aligned with the full dataset.
        features: Feature matrix of the search part.
        target: Binary target of the search part.

    Returns:
        Average precision per fold, using the rule indicator as score.
    """
    splitter = cv_splitter(seed=DATA_SEED)
    scores = []
    for _, val_pos in splitter.split(features, target):
        fold_rules = rules.iloc[val_pos].to_numpy()
        fold_target = target.iloc[val_pos].to_numpy()
        scores.append(float(average_precision_score(fold_target, fold_rules)))
    return scores


def score_baselines(frame: pd.DataFrame) -> list[TrialRecord]:
    """Evaluate the reference configurations on the shared folds.

    Args:
        frame: Raw dataset.

    Returns:
        Trial records for the default logistic and random forest
        configurations and the ground-truth rules.
    """
    train_idx, _ = holdout_indices(frame, seed=DATA_SEED)
    features = make_features(frame).iloc[train_idx]
    target = make_target(frame).iloc[train_idx]
    splitter = cv_splitter(seed=DATA_SEED)
    start = perf_counter()
    records = []
    configurations: list[tuple[str, Any]] = [
        ("baseline_default_logistic", default_logistic()),
        ("baseline_default_forest", default_random_forest(seed=DATA_SEED)),
    ]
    for arm, estimator in configurations:
        t0 = perf_counter()
        scores = cv_average_precision(estimator, features, target, splitter)
        records.append(
            TrialRecord(
                arm=arm,
                trial_index=0,
                params={"estimator": arm},
                fold_scores=scores.fold_scores,
                mean_score=scores.mean,
                std_score=scores.std,
                wall_clock_s=perf_counter() - t0,
                cumulative_s=perf_counter() - start,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )
    t0 = perf_counter()
    fold_scores = rules_cv_scores(
        ground_truth_failure(frame).iloc[train_idx], features, target
    )
    records.append(
        TrialRecord(
            arm=RULES_ARM,
            trial_index=0,
            params={"expression": "hdf | pwf | osf"},
            fold_scores=fold_scores,
            mean_score=float(np.mean(fold_scores)),
            std_score=float(np.std(fold_scores)),
            wall_clock_s=perf_counter() - t0,
            cumulative_s=perf_counter() - start,
            timestamp=datetime.now(UTC).isoformat(),
        )
    )
    return records


def holdout_ap(estimator: Any, frame: pd.DataFrame) -> float:
    """Fit an estimator on the search part and score the holdout once.

    Args:
        estimator: Scikit-learn classifier with ``predict_proba``.
        frame: Raw dataset.

    Returns:
        Average precision on the holdout.
    """
    train_idx, test_idx = holdout_indices(frame, seed=DATA_SEED)
    features = make_features(frame)
    target = make_target(frame)
    estimator.fit(features.iloc[train_idx], target.iloc[train_idx])
    proba = estimator.predict_proba(features.iloc[test_idx])
    pos_col = int(np.flatnonzero(estimator.classes_ == 1)[0])
    return float(average_precision_score(target.iloc[test_idx], proba[:, pos_col]))


def best_parametric_entry(
    arm: str, records: list[TrialRecord], frame: pd.DataFrame
) -> dict[str, Any]:
    """Refit the best record of an arm and score it on the holdout.

    Args:
        arm: Arm name for the summary row.
        records: Trial records of one Optuna or agent log.
        frame: Raw dataset.

    Returns:
        Summary dictionary with parameters and both scores.
    """
    best = max(records, key=lambda record: record.mean_score)
    estimator = build_estimator(best.params, seed=DATA_SEED)
    return {
        "arm": arm,
        "best_params": best.params,
        "cv_mean": best.mean_score,
        "holdout_ap": holdout_ap(estimator, frame),
    }


def best_pysr_entry(records: list[TrialRecord]) -> dict[str, Any]:
    """Aggregate PySR records by complexity and select the best.

    Only complexities present in every fold are eligible for selection,
    because hall-of-fame complexities appear in a varying number of
    folds and the highest raw scores tend to come from complexities
    seen in a single fold.

    Args:
        records: Trial records of the PySR log.

    Returns:
        Summary dictionary with the selected complexity and both scores.

    Raises:
        ValueError: If no records are given.
    """
    by_complexity: dict[int, list[TrialRecord]] = {}
    for record in records:
        by_complexity.setdefault(int(record.params["complexity"]), []).append(record)
    if not by_complexity:
        raise ValueError("no pysr records found")
    expected_folds = max(len(group) for group in by_complexity.values())
    best: dict[str, Any] | None = None
    for complexity, group in by_complexity.items():
        if len(group) != expected_folds:
            continue
        cv_mean = float(np.mean([record.mean_score for record in group]))
        holdout_scores = [
            record.payload.get("holdout_ap")
            for record in group
            if record.payload.get("holdout_ap") is not None
        ]
        holdout_mean = float(np.mean(holdout_scores)) if holdout_scores else None
        entry = {
            "arm": "pysr",
            "best_params": {
                "complexity": complexity,
                "expression": group[-1].params["expression"],
            },
            "cv_mean": cv_mean,
            "holdout_ap": holdout_mean,
            "folds_present": len(group),
        }
        if best is None or entry["cv_mean"] > best["cv_mean"]:
            best = entry
    if best is None:
        raise ValueError("no pysr records found")
    return best


def baseline_holdout_entries(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Score the baseline estimators and the rules on the holdout.

    Args:
        frame: Raw dataset.

    Returns:
        Summary rows for the default logistic, default forest, and rules.
    """
    _, test_idx = holdout_indices(frame, seed=DATA_SEED)
    entries = [
        {
            "arm": "baseline_default_logistic",
            "best_params": {"estimator": "default"},
            "cv_mean": None,
            "holdout_ap": holdout_ap(default_logistic(), frame),
        },
        {
            "arm": "baseline_default_forest",
            "best_params": {"estimator": "default"},
            "cv_mean": None,
            "holdout_ap": holdout_ap(default_random_forest(seed=DATA_SEED), frame),
        },
        {
            "arm": RULES_ARM,
            "best_params": {"expression": "hdf | pwf | osf"},
            "cv_mean": None,
            "holdout_ap": float(
                average_precision_score(
                    make_target(frame).iloc[test_idx],
                    ground_truth_failure(frame).iloc[test_idx],
                )
            ),
        },
    ]
    return entries


def cpu_model() -> str:
    """Read the CPU model name from procfs.

    Returns:
        CPU model string, or an empty string when unavailable.
    """
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def julia_version() -> str | None:
    """Query the system Julia version.

    Returns:
        Version string, or None when Julia is unavailable.
    """
    try:
        result = subprocess.run(
            ["julia", "--version"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info() -> dict[str, Any]:
    """Collect execution environment details.

    Returns:
        Dictionary with interpreter, package, CPU, and runtime details.
    """
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "packages": packages,
        "cpu": cpu_model(),
        "threads": 1,
        "julia": julia_version(),
        "agent_model": os.environ.get("GLM_MODEL", "glm-5.3-flash"),
    }


def run(
    out_dir: str | Path = "results",
    frame: pd.DataFrame | None = None,
) -> list[Path]:
    """Write baseline logs, holdout summary, and environment record.

    Args:
        out_dir: Results directory with the arm logs.
        frame: Dataset override, mainly for tests.

    Returns:
        Paths of the written files.
    """
    frame = load_dataset() if frame is None else frame
    out_dir = Path(out_dir)

    baselines_path = out_dir / "baselines.jsonl"
    baselines_path.parent.mkdir(parents=True, exist_ok=True)
    baselines_path.unlink(missing_ok=True)
    for record in score_baselines(frame):
        append_record(baselines_path, record)

    entries = baseline_holdout_entries(frame)
    for log_path in sorted(out_dir.glob("*.jsonl")):
        name = log_path.stem
        if name.startswith("optuna_seed"):
            entries.append(best_parametric_entry(name, load_records(log_path), frame))
        elif name == "agent":
            entries.append(
                best_parametric_entry("agent", load_records(log_path), frame)
            )
        elif name == "pysr":
            entries.append(best_pysr_entry(load_records(log_path)))
    holdout_path = out_dir / "holdout.json"
    holdout_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")

    environment_path = out_dir / "environment.json"
    environment_path.write_text(
        json.dumps(environment_info(), indent=2) + "\n", encoding="utf-8"
    )
    return [baselines_path, holdout_path, environment_path]


def main() -> None:
    """Write the baseline, holdout, and environment files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    for path in run(args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
