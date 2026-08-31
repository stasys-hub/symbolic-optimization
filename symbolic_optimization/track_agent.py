"""Agent track: a pydantic-graph loop that tunes scikit-learn parameters.

The model receives the shared search space description, the dataset
context, and the trial history, and returns one configuration plus a
rationale as structured output. The host executes all cross-validation;
the model performs no tool calls. The loop stops after a fixed
evaluation budget. Proposals that fail validation are retried by
pydantic-ai; a run aborts if the retries are exhausted.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import pandas as pd
from pydantic import BaseModel, BeforeValidator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphBuilder,
    GraphRunContext,
    StepContext,
)
from sklearn.model_selection import BaseCrossValidator

from symbolic_optimization.data import (
    cv_splitter,
    holdout_indices,
    load_dataset,
    make_features,
    make_target,
)
from symbolic_optimization.evaluation import cv_average_precision
from symbolic_optimization.search_space import (
    SearchParams,
    build_estimator,
    space_description,
)
from symbolic_optimization.trials import TrialRecord, append_record

DATA_SEED = 0
ARM_NAME = "agent"
DEFAULT_MODEL = "glm-5.3-flash"
DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


def decode_string_params(value: object) -> object:
    """Decode a JSON string into a dictionary.

    Some models return the nested parameter object of a tool call as a
    JSON string instead of a structured object.

    Args:
        value: Raw field value.

    Returns:
        The decoded dictionary, or the value unchanged.
    """
    if isinstance(value, str):
        return json.loads(value)
    return value


CoercedParams = Annotated[SearchParams, BeforeValidator(decode_string_params)]


class Proposal(BaseModel):
    """One configuration proposal returned by the model.

    Attributes:
        params: Parameters within the shared search space.
        rationale: One-sentence justification for the configuration.
    """

    params: CoercedParams
    rationale: str


@dataclass
class LoopState:
    """Mutable state of the agent loop.

    Attributes:
        records: Evaluated trial records in order.
        features: Feature matrix of the search part.
        target: Binary target of the search part.
        splitter: Cross-validation splitter.
        log_path: JSONL log file.
        seed: Seed for the estimators.
        n_evaluations: Evaluation budget.
        start: Start time of the run.
    """

    records: list[TrialRecord]
    features: pd.DataFrame
    target: pd.Series
    splitter: BaseCrossValidator
    log_path: Path
    seed: int
    n_evaluations: int
    start: float


@dataclass
class Deps:
    """Dependencies passed to every graph node.

    Attributes:
        agent: Configured pydantic-ai agent.
        model_id: Model identifier for logging.
    """

    agent: Agent[None, Proposal]
    model_id: str


def make_agent() -> tuple[Agent[None, Proposal], str]:
    """Build the agent from environment configuration.

    Returns:
        Configured agent and its model identifier.

    Raises:
        RuntimeError: If ``GLM_API_KEY`` is not set.
    """
    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        raise RuntimeError("GLM_API_KEY is not set")
    model_id = os.environ.get("GLM_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("GLM_BASE_URL", DEFAULT_BASE_URL)
    model = OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )
    agent: Agent[None, Proposal] = Agent(
        model,
        output_type=Proposal,
        retries=3,
        model_settings=ModelSettings(temperature=0.0),
    )
    return agent, model_id


async def request_proposal(
    agent: Agent[None, Proposal], prompt: str
) -> tuple[Proposal, Any, float]:
    """Request the next configuration from the model.

    Args:
        agent: Configured pydantic-ai agent.
        prompt: Prompt with history and budget.

    Returns:
        Proposal, token usage, and request latency in seconds.
    """
    started = perf_counter()
    result = await agent.run(prompt)
    return result.output, result.usage, perf_counter() - started


def usage_to_dict(usage: Any) -> dict[str, Any]:
    """Convert a usage object to a JSON-serializable dictionary.

    Args:
        usage: Usage object or dictionary.

    Returns:
        Dictionary of usage counters.
    """
    if isinstance(usage, dict):
        return usage
    if is_dataclass(usage) and not isinstance(usage, type):
        return asdict(usage)
    return dict(vars(usage))


def format_history(records: list[TrialRecord]) -> str:
    """Format the trial history for the prompt.

    Args:
        records: Evaluated trial records in order.

    Returns:
        One line per record, or a placeholder when empty.
    """
    if not records:
        return "No evaluations yet."
    lines = []
    for record in records:
        params = json.dumps(record.params, separators=(",", ":"))
        lines.append(
            f"t{record.trial_index:02d} | {params} | AP={record.mean_score:.4f}"
            f" (+-{record.std_score:.4f}) | {record.wall_clock_s:.1f}s"
        )
    return "\n".join(lines)


def build_prompt(state: LoopState) -> str:
    """Build the proposal prompt from the loop state.

    Args:
        state: Current loop state.

    Returns:
        Prompt string with task, space, history, and budget.
    """
    best = max((record.mean_score for record in state.records), default=None)
    remaining = state.n_evaluations - len(state.records)
    return (
        "You optimize hyperparameters for a binary failure classifier.\n"
        f"Dataset: AI4I 2020 predictive maintenance, {len(state.target)}"
        f" training rows, {float(state.target.mean()):.3%} positive labels.\n"
        "Metric: mean average precision over a stratified 5-fold"
        " cross-validation; the standard deviation over folds is"
        " reported.\n"
        f"{space_description()}\n\n"
        "Trial history (oldest first):\n"
        f"{format_history(state.records)}\n\n"
        f"Best mean AP so far: {best if best is None else f'{best:.4f}'}\n"
        f"Evaluations remaining: {remaining}\n\n"
        "Return one configuration for the next evaluation and a"
        " one-sentence rationale."
    )


@dataclass
class Propose(BaseNode[LoopState, Deps, None]):
    """Node that requests the next configuration from the model."""

    async def run(
        self, ctx: GraphRunContext[LoopState, Deps]
    ) -> "Evaluate | End[None]":
        prompt = build_prompt(ctx.state)
        proposal, usage, latency = await request_proposal(ctx.deps.agent, prompt)
        return Evaluate(proposal=proposal, usage=usage, latency=latency)


@dataclass
class Evaluate(BaseNode[LoopState, Deps, None]):
    """Node that evaluates a proposal and logs the result."""

    proposal: Proposal
    usage: Any
    latency: float

    async def run(self, ctx: GraphRunContext[LoopState, Deps]) -> "Decide | End[None]":
        params = self.proposal.params.model_dump()
        estimator = build_estimator(params, seed=ctx.state.seed)
        started = perf_counter()
        scores = cv_average_precision(
            estimator, ctx.state.features, ctx.state.target, ctx.state.splitter
        )
        record = TrialRecord(
            arm=ARM_NAME,
            trial_index=len(ctx.state.records),
            params=params,
            fold_scores=scores.fold_scores,
            mean_score=scores.mean,
            std_score=scores.std,
            wall_clock_s=perf_counter() - started,
            cumulative_s=perf_counter() - ctx.state.start,
            timestamp=datetime.now(UTC).isoformat(),
            payload={
                "rationale": self.proposal.rationale,
                "usage": usage_to_dict(self.usage),
                "latency_s": self.latency,
                "model": ctx.deps.model_id,
            },
        )
        append_record(ctx.state.log_path, record)
        ctx.state.records.append(record)
        return Decide()


@dataclass
class Decide(BaseNode[LoopState, Deps, None]):
    """Node that continues or stops the loop."""

    async def run(self, ctx: GraphRunContext[LoopState, Deps]) -> "Propose | End[None]":
        if len(ctx.state.records) >= ctx.state.n_evaluations:
            return End(None)
        return Propose()


def build_graph() -> Graph[LoopState, Deps, None, None]:
    """Assemble the propose, evaluate, decide graph.

    Returns:
        Executable graph without attached state or dependencies.
    """
    builder = GraphBuilder(
        state_type=LoopState,
        deps_type=Deps,
        output_type=type(None),
    )

    @builder.step
    async def start(ctx: StepContext[LoopState, Deps, None]) -> Propose:
        return Propose()

    builder.add(
        builder.node(Propose),
        builder.node(Evaluate),
        builder.node(Decide),
        builder.edge_from(builder.start_node).to(start),
    )
    return builder.build()


def run(
    n_evaluations: int = 50,
    frame: pd.DataFrame | None = None,
    out_dir: str | Path = "results",
    agent: Agent[None, Proposal] | None = None,
    model_id: str | None = None,
) -> Path:
    """Run the agent arm and write its JSONL log.

    Args:
        n_evaluations: Evaluation budget.
        frame: Dataset override, mainly for tests.
        out_dir: Directory for the trial log.
        agent: Agent override, mainly for tests.
        model_id: Model identifier override, mainly for tests.

    Returns:
        Path of the written JSONL log.
    """
    if agent is None or model_id is None:
        built_agent, built_model_id = make_agent()
        agent = built_agent if agent is None else agent
        model_id = built_model_id if model_id is None else model_id
    frame = load_dataset() if frame is None else frame
    train_idx, _ = holdout_indices(frame, seed=DATA_SEED)
    features = make_features(frame).iloc[train_idx]
    target = make_target(frame).iloc[train_idx]

    log_path = Path(out_dir) / "agent.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    state = LoopState(
        records=[],
        features=features,
        target=target,
        splitter=cv_splitter(seed=DATA_SEED),
        log_path=log_path,
        seed=DATA_SEED,
        n_evaluations=n_evaluations,
        start=perf_counter(),
    )
    graph = build_graph()
    graph.run_sync(state=state, deps=Deps(agent=agent, model_id=model_id))
    return log_path


def main() -> None:
    """Run the agent arm from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=int, default=50)
    args = parser.parse_args()
    print(run(args.evaluations))


if __name__ == "__main__":
    main()
