"""Trial records and JSONL persistence."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrialRecord:
    """One evaluated configuration of a search arm.

    Attributes:
        arm: Name of the search arm.
        trial_index: Zero-based evaluation counter within the arm.
        params: Configuration evaluated on this trial.
        fold_scores: Per-fold scores, one per cross-validation fold.
        mean_score: Mean of the fold scores.
        std_score: Standard deviation of the fold scores.
        wall_clock_s: Elapsed time of this evaluation.
        cumulative_s: Elapsed time since the start of the arm run.
        timestamp: ISO-format UTC time at the end of the evaluation.
        payload: Arm-specific fields, for example optuna distributions or
            agent rationales.
    """

    arm: str
    trial_index: int
    params: dict[str, Any]
    fold_scores: list[float]
    mean_score: float
    std_score: float
    wall_clock_s: float
    cumulative_s: float
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)


def append_record(path: str | Path, record: TrialRecord) -> None:
    """Append one record to a JSONL file.

    Args:
        path: Target JSONL file, created if absent.
        record: Record to persist.
    """
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def load_records(path: str | Path) -> list[TrialRecord]:
    """Read all records from a JSONL file.

    Args:
        path: Source JSONL file.

    Returns:
        Records in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    records: list[TrialRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            records.append(TrialRecord(**json.loads(line)))
    return records
