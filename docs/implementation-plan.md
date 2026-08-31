# Implementation plan

## Goal

Compare three search strategies on the AI4I 2020 predictive maintenance
dataset and record enough detail to analyze runtime behavior and decisions:
Optuna over scikit-learn pipelines, symbolic regression with PySR, and a
pydantic-graph agent loop that tunes scikit-learn hyperparameters. The
analysis produces plots and tables for an accompanying blog post series.

## Settled decisions

| Decision | Value |
| --- | --- |
| Target | Binary `Machine failure` |
| Excluded columns | `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` |
| Features | Five sensor columns plus `Type`, one-hot encoded |
| Metric | Average precision (PR-AUC) |
| Validation | Stratified 5-fold CV, one fixed seed shared by all arms |
| Budget | 50 CV evaluations per arm |
| Timing | Wall-clock recorded per evaluation and cumulatively |
| Arms 1 and 3 search space | Estimator choice (logistic branch, random forest branch) plus per-branch hyperparameters, identical for both arms |
| Agent model | GLM 5.3 flash via the OpenAI-compatible pydantic-ai interface |
| Agent endpoint | `https://api.z.ai/api/coding/paas/v4`, default of `GLM_BASE_URL` |
| Agent configuration | Key from `GLM_API_KEY`, model id from `GLM_MODEL` (default `glm-5.3-flash`), temperature 0 |
| Canonical data file | `ai4i2020.csv` at repository root, tracked |

### Search space for arms 1 and 3

- `estimator`: `logistic` or `random_forest`.
- Logistic branch: pipeline of `StandardScaler`, optional
  `PolynomialFeatures(interaction_only=True, include_bias=False)`,
  `LogisticRegression`. Parameters: `use_interactions` (boolean), `C`
  (log scale, 1e-3 to 1e2), `penalty` (l1 or l2), `class_weight` (none or
  balanced), fixed `liblinear` solver.
- Random forest branch: `n_estimators` (50 to 400), `max_depth` (3 to 20),
  `min_samples_leaf` (1 to 20), `max_features` (sqrt, log2, or all),
  `class_weight` (none or balanced_subsample).

## Trial schema

One JSONL file per arm under `results/`. Common fields:

    arm               str
    trial_index       int
    params            dict
    fold_scores       list[float]
    mean_score        float
    std_score         float
    wall_clock_s      float
    cumulative_s      float
    timestamp         str

Arm-specific payload fields:

- Optuna: distribution names and the sampler step count.
- PySR: candidate expression string, complexity, training loss. The budget is
  expressed as PySR iterations, with every hall-of-fame candidate logged as a
  trial record so that the evaluation count stays comparable.
- Agent: rationale text as returned by the model, token counts, and model id.

## Phases

### Phase 0: setup

- Add `pydantic-graph` with uv; optuna is already present.
- Remove the `main.py` bootstrap file.
- Verify the PySR import and the Julia installation with a minimal regression
  on synthetic data.
- Verify the GLM endpoint with a one-token pydantic-ai request, confirming
  the model id string.

Acceptance: `uv run pytest` collects zero tests without error, PySR fits a
toy expression, the endpoint call returns a completion.

### Phase 1: package skeleton, data module, trial schema

- Create `symbolic_optimization/__init__.py`, `data.py`, `trials.py`.
- `data.py`: dataset loading, feature and target selection, the shared CV
  splitter factory, one-hot encoding of the product variant.
- `trials.py`: the trial record dataclass, JSONL append and load functions.
- Tests: no label or identifier column appears in the feature matrix, the
  splitter is stratified and seeded, records round-trip through JSONL.

Acceptance: `uv run pytest` passes, `uv run ruff check` is clean.

### Phase 2: Optuna track

- `track_optuna.py`: TPE study over the shared space, objective built on the
  shared evaluation function, per-trial JSONL logging, fixed seed.
- Tests: the study returns a best configuration, the log contains one record
  per trial.

Acceptance: a 10-trial smoke run writes a valid log.

### Phase 3: PySR track

- `track_pysr.py`: weighted `PySRRegressor` on the binary target, bounded
  complexity, logging of hall-of-fame candidates mapped to trial records.
- Ground-truth comparison helper: evaluates the known rule set as a
  classifier on the same folds for reference.
- Tests: the mapping from PySR output to trial records is correct.

Acceptance: a short run produces expressions with logged complexity and loss.

### Phase 4: agent track

- `track_agent.py`: pydantic-graph loop with nodes `propose`, `evaluate`,
  `decide`. The agent receives the search space description, the trial
  history as a table, and the remaining budget. It returns the next
  configuration and a rationale as structured output. The host executes all
  cross-validation; the model performs no tool calls.
- Stop after 50 evaluations. Log rationale, latency, and token counts per
  step.
- Tests: the output schema rejects invalid configurations; the loop
  terminates at the budget.

Acceptance: a 5-evaluation smoke run against the live endpoint writes a
valid log.

### Phase 5: execution

- Run all three arms at full budget, commit the resulting JSONL logs.
- Record environment details (Python, package versions, CPU) in
  `results/environment.json`.

### Phase 6: analysis notebook

- `notebooks/analysis.ipynb`: best-so-far curves versus evaluation index and
  versus wall-clock time, parameter trajectories per arm, agent decision
  timeline with rationales, PySR expressions against the ground-truth rules.
- Altair for all plots, with matplotlib as the documented fallback.

Acceptance: the notebook runs top to bottom on the committed logs.

### Phase 7: blog support

- Export publication figures from the notebook.
- Produce the summary table (best score, evaluations to reach it, total time)
  as a printed or written artifact for the post.

## Risks

- PySR depends on a Julia runtime installed through juliacall. First
  installation needs network access and several minutes. Phase 0 verifies
  this before any dependent work starts.
- The model id string for the GLM endpoint is unverified. Phase 0 confirms it.
- The coding-plan endpoint may enforce rate limits. The agent loop records
  latency and retries, and the log keeps every request-response pair.
- LLM output is not reproducible in general. Temperature 0 and full logging
  of requests bound the problem; the blog post reports the recorded run.
- Class imbalance near 3.4% makes accuracy uninformative. Average precision
  and class-weight options in the search space address this.

## Out of scope

Multiclass failure-type prediction, deep learning baselines, deployment of
any trained model, and the blog prose itself.
