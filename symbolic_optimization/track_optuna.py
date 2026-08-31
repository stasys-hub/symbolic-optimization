"""Optuna track: TPE search over the shared space."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from symbolic_optimization.data import (
    cv_splitter,
    holdout_indices,
    load_dataset,
    make_features,
    make_target,
)
from symbolic_optimization.evaluation import cv_average_precision
from symbolic_optimization.search_space import build_estimator, suggest_params
from symbolic_optimization.trials import TrialRecord, append_record

DATA_SEED = 0
ARM_NAME = "optuna"


def run(
    seed: int,
    n_trials: int = 50,
    frame: pd.DataFrame | None = None,
    out_dir: str | Path = "results",
) -> Path:
    """Run the Optuna arm and write its JSONL log.

    Args:
        seed: Seed for the TPE sampler and the estimators.
        n_trials: Number of cross-validation evaluations.
        frame: Dataset override, mainly for tests.
        out_dir: Directory for the trial log.

    Returns:
        Path of the written JSONL log.
    """
    frame = load_dataset() if frame is None else frame
    train_idx, _ = holdout_indices(frame, seed=DATA_SEED)
    features = make_features(frame).iloc[train_idx]
    target = make_target(frame).iloc[train_idx]
    splitter = cv_splitter(seed=DATA_SEED)

    log_path = Path(out_dir) / f"optuna_seed{seed}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    start = perf_counter()

    def objective(trial: optuna.trial.Trial) -> float:
        params: dict[str, Any] = suggest_params(trial)
        estimator = build_estimator(params, seed=seed)
        t0 = perf_counter()
        scores = cv_average_precision(estimator, features, target, splitter)
        append_record(
            log_path,
            TrialRecord(
                arm=ARM_NAME,
                trial_index=trial.number,
                params=params,
                fold_scores=scores.fold_scores,
                mean_score=scores.mean,
                std_score=scores.std,
                wall_clock_s=perf_counter() - t0,
                cumulative_s=perf_counter() - start,
                timestamp=datetime.now(UTC).isoformat(),
                payload={"sampler": "tpe"},
            ),
        )
        return scores.mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return log_path


def main() -> None:
    """Run the Optuna arm from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    print(run(args.seed, args.trials))


if __name__ == "__main__":
    main()
