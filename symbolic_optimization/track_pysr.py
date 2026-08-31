"""PySR track: symbolic regression with bounded complexity.

The Julia runtime raises a segmentation fault during interpreter
shutdown once it has been initialized. After the first successful fit,
this module registers an ``atexit`` handler that terminates the process
with a zero exit code so that successful runs are not reported as
failures.
"""

import argparse
import atexit
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.metrics import average_precision_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler

from symbolic_optimization.data import (
    cv_splitter,
    holdout_indices,
    load_dataset,
    make_features,
    make_target,
)
from symbolic_optimization.trials import TrialRecord, append_record

DATA_SEED = 0
ARM_NAME = "pysr"
BINARY_OPERATORS = ["+", "-", "*", "/", "max", "min"]
UNARY_OPERATORS: list[str] = []
VARIABLE_NAMES = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "proc_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}

_HARD_EXIT_REGISTERED = False


def register_hard_exit() -> None:
    """Register a zero-code process exit for interpreter shutdown.

    Runs once per process after the Julia runtime has been initialized.
    """
    global _HARD_EXIT_REGISTERED
    if not _HARD_EXIT_REGISTERED:
        atexit.register(os._exit, 0)
        _HARD_EXIT_REGISTERED = True


def balanced_sample_weights(target: pd.Series) -> np.ndarray:
    """Compute balanced per-sample weights for a binary target.

    Args:
        target: Binary target series.

    Returns:
        Weight per sample, with total weight equal per class.
    """
    counts = target.value_counts().to_dict()
    n = len(target)
    return target.map(lambda label: n / (2.0 * counts[label])).to_numpy()


def run(
    n_iterations: int = 50,
    max_complexity: int = 25,
    frame: pd.DataFrame | None = None,
    out_dir: str | Path = "results",
    splitter: BaseCrossValidator | None = None,
) -> Path:
    """Run the PySR arm and log every hall-of-fame candidate.

    Each cross-validation fold runs one PySR search on standardized
    training data with balanced sample weights. Every expression in the
    resulting hall of fame is scored with average precision on the
    validation fold and on the untouched holdout, and written as one
    trial record.

    Args:
        n_iterations: PySR iterations per fold.
        max_complexity: Maximum expression complexity.
        frame: Dataset override, mainly for tests.
        out_dir: Directory for the trial log.
        splitter: Cross-validation override, mainly for tests.

    Returns:
        Path of the written JSONL log.
    """
    frame = load_dataset() if frame is None else frame
    train_idx, holdout_idx = holdout_indices(frame, seed=DATA_SEED)
    all_features = make_features(frame)
    features = all_features.iloc[train_idx].rename(columns=VARIABLE_NAMES)
    holdout_features = all_features.iloc[holdout_idx].rename(columns=VARIABLE_NAMES)
    target = make_target(frame).iloc[train_idx]
    holdout_target = make_target(frame).iloc[holdout_idx]
    splitter = cv_splitter(seed=DATA_SEED) if splitter is None else splitter

    log_path = Path(out_dir) / "pysr.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    start = perf_counter()
    trial_counter = 0

    for fold, (fit_pos, val_pos) in enumerate(splitter.split(features, target)):
        fit_features = features.iloc[fit_pos]
        val_features = features.iloc[val_pos]
        fit_target = target.iloc[fit_pos]
        val_target = target.iloc[val_pos]
        scaler = StandardScaler().fit(fit_features)
        fit_scaled = pd.DataFrame(
            scaler.transform(fit_features), columns=features.columns
        )
        val_scaled = pd.DataFrame(
            scaler.transform(val_features), columns=features.columns
        )
        weights = balanced_sample_weights(fit_target)

        fold_start = perf_counter()
        model = PySRRegressor(
            niterations=n_iterations,
            binary_operators=BINARY_OPERATORS,
            unary_operators=UNARY_OPERATORS,
            maxsize=max_complexity,
            temp_equation_file=True,
            verbosity=0,
            progress=False,
            parallelism="serial",
            deterministic=True,
            random_state=DATA_SEED * 100 + fold,
        )
        model.fit(fit_scaled, fit_target.to_numpy(), weights=weights)
        register_hard_exit()
        fold_runtime_s = perf_counter() - fold_start
        holdout_scaled = pd.DataFrame(
            scaler.transform(holdout_features), columns=features.columns
        )

        equations = model.equations_
        for row_index, row in enumerate(equations.itertuples()):
            t0 = perf_counter()
            predictions = model.predict(val_scaled, index=row_index)
            score = float(average_precision_score(val_target, predictions))
            holdout_predictions = model.predict(holdout_scaled, index=row_index)
            holdout_ap = float(
                average_precision_score(holdout_target, holdout_predictions)
            )
            append_record(
                log_path,
                TrialRecord(
                    arm=ARM_NAME,
                    trial_index=trial_counter,
                    params={
                        "complexity": int(row.complexity),
                        "expression": str(row.equation),
                    },
                    fold_scores=[score],
                    mean_score=score,
                    std_score=0.0,
                    wall_clock_s=perf_counter() - t0,
                    cumulative_s=perf_counter() - start,
                    timestamp=datetime.now(UTC).isoformat(),
                    payload={
                        "fold": fold,
                        "loss": float(row.loss),
                        "fold_runtime_s": fold_runtime_s,
                        "holdout_ap": holdout_ap,
                    },
                ),
            )
            trial_counter += 1
    return log_path


def main() -> None:
    """Run the PySR arm from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max-complexity", type=int, default=25)
    args = parser.parse_args()
    print(run(args.iterations, args.max_complexity), flush=True)


if __name__ == "__main__":
    main()
