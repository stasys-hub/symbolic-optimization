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
| Agent configuration | Key from `GLM_API_KEY`, model id from `GLM_MODEL` (default `glm-5.3-flash`), temperature 0 |
| Holdout | Stratified 80/20 split; search uses the train part only; each arm's final configuration is fit once and scored once on the test part |
| Execution policy | Single-threaded everywhere: no parallel CV in Optuna, fixed PySR thread count; thread counts, CPU, and Julia version recorded |
| PySR preprocessing | Standardization fit inside the CV folds only |
| Baselines | Default logistic regression and default random forest; the ground-truth rule classifier, scored on the same splits |
| Repetitions | Arm 1 repeated with 5 seeds for a variance band; arms 2 and 3 single-run |
| Agent model | GLM 5.3 flash via the OpenAI-compatible pydantic-ai interface |
| Agent endpoint | `https://api.z.ai/api/coding/paas/v4`, default of `GLM_BASE_URL` |
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
- Verify PySR against the installed Julia 1.12.6 (system package) with a
  minimal regression on synthetic data, confirming juliacall uses it.
- Verify pydantic-ai structured output against the GLM endpoint with a small
  typed response, confirming the model id and the structured-output mechanism.
  If strict structured output is unsupported, record the fallback of parsing
  the content field as JSON.

Acceptance: PySR fits a toy expression, the endpoint call returns a typed
response or a documented fallback.

### Phase 1: package skeleton, data module, trial schema

- Create `symbolic_optimization/__init__.py`, `data.py`, `trials.py`,
  `baselines.py`.
- `data.py`: dataset loading, feature and target selection, the stratified
  80/20 holdout split, the shared CV splitter factory for the train part,
  one-hot encoding of the product variant.
- `trials.py`: the trial record dataclass, JSONL append and load functions.
- `baselines.py`: default logistic regression and random forest
  configurations; the ground-truth rule classifier derived from the dataset
  documentation.
- Tests: no label or identifier column appears in the feature matrix, the
  holdout is disjoint from the training part and stratified, records
  round-trip through JSONL, the deterministic rules reproduce the `HDF`,
  `PWF`, and `OSF` indicator columns exactly.

Acceptance: `uv run pytest` passes, `uv run ruff check` is clean.

### Phase 2: Optuna track

- `track_optuna.py`: TPE study over the shared space, objective built on the
  shared evaluation function, per-trial JSONL logging, fixed seed.
- The arm runs with 5 different seeds, one log file per seed.
- Tests: the study returns a best configuration, the log contains one record
  per trial.

Acceptance: a 10-trial smoke run writes a valid log.

### Phase 3: PySR track

- `track_pysr.py`: weighted `PySRRegressor` on the binary target,
  standardization fit inside each fold, bounded complexity, logging of
  hall-of-fame candidates mapped to trial records.
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

- Run all three arms at full budget, commit the resulting JSONL logs
  (arm 1 once per seed).
- Run the baselines on the same folds.
- Score every final configuration once on the holdout and record it.
- Record environment details (Python, package versions, CPU, thread counts,
  Julia version) in `results/environment.json`.

### Phase 6: analysis notebook

- `notebooks/analysis.ipynb`: best-so-far curves versus evaluation index and
  versus wall-clock time, with the arm 1 seed band, baseline reference lines,
  and the ground-truth rule line; parameter trajectories per arm; agent
  decision timeline with rationales; PySR expressions against the
  ground-truth rules; holdout scores as a summary table.
- Altair for all plots, with matplotlib as the documented fallback.
- Every figure carries axis labels, units, and a title; the notebook states
  the single-run limitation for arms 2 and 3.

Acceptance: the notebook runs top to bottom on the committed logs.

### Phase 7: blog support

- Export publication figures from the notebook.
- Produce the summary table (best score, evaluations to reach it, total time)
  as a printed or written artifact for the post.

## Risks

- Julia 1.12.6 is installed system-wide. juliacall may still fetch a private
  runtime; phase 0 verifies which one is used and records it.
- The endpoint answers plain completions (verified with curl). Strict
  `json_schema` structured output through pydantic-ai is unverified; the
  fallback is manual JSON parsing of the content field.
- glm-5.3-flash produces internal reasoning tokens before the answer. The
  loop needs a generous `max_tokens` budget and separate logging of
  reasoning tokens.
- The coding-plan endpoint may enforce rate limits. The agent loop records
  latency and retries, and the log keeps every request-response pair.
- LLM output is not reproducible in general. Temperature 0 and full logging
  of requests bound the problem; the blog post reports the recorded run.
- Class imbalance near 3.4% makes accuracy uninformative. Average precision
  and class-weight options in the search space address this.

## Out of scope

Multiclass failure-type prediction, deep learning baselines, deployment of
any trained model, and the blog prose itself.
