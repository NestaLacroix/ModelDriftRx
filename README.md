# ModelDriftRx
Detect that your model is dying, it diagnoses why, fixes itself, and proves the fix worked, all autonomously. Think of it as an immune system for ML models. Will feature a dashboard to monitor model health and more, development in progress

## Implementation Phases

### Phase 0 - Project Skeleton

**Goal:** Set up the repository, tooling, and CI pipeline so that every future phase starts with
a working development environment.

**What was built:**
- Repository structure with all directories and __init__.py files
- pyproject.toml with all dependencies and tool configuration (ruff, mypy, pytest)
- Makefile with commands: make install, make lint, make format, make typecheck, make test
- GitHub Actions CI workflow (ci.yml) that runs lint, type-check, and tests on every push
- .env.example with all configurable environment variables
- .gitignore for data, models, reports, and environment files

**Files:**
- `pyproject.toml` - project metadata, dependencies, tool settings
- `Makefile` - developer commands
- `.github/workflows/ci.yml` - CI pipeline
- `.github/workflows/scheduled-drift-check.yml` - scheduled drift check (placeholder)
- `.env.example` - environment variable template
- `.gitignore` - ignored files and directories

---

### Phase 1 - Contracts, Interfaces, and Test Foundation

**Goal:** Define every data structure and interface the system will use. Build the test
infrastructure. No monitoring logic yet, just the shapes of data that will flow between
components.

**What was built:**

- `src/utils/config.py` - Central frozen dataclass configuration with env var overrides
- `src/contracts.py` - Dataclasses (ModelMetrics, FeatureDrift, DriftReport, DiagnosisResult, HealingOutcome, IncidentReport) and enums (DriftSeverity, HealAction)
- `src/protocols.py` - MonitorableModel Protocol for model-agnostic integration
- `tests/mocks.py` - FakeModel for testing without real ML frameworks
- `tests/conftest.py` - Shared pytest fixtures
- `tests/unit/test_contracts.py` - Tests for all contracts and edge cases

---

### Phase 2 - Drift Detection

**Goal:** Build the component that compares incoming data against a baseline and determines
if distribution shift has occurred.

**What was built:**

`src/detector.py` - The DriftDetector class. Takes baseline data and incoming data as numpy
arrays. For each feature column, it computes:

- PSI: Measures how much the distribution has shifted. Splits both distributions into bins 
  and compares the proportions. Low PSI means stable, high PSI means the data has changed 
  significantly.

- KS Test: A statistical hypothesis test that determines whether two samples come from the 
  same distribution. Returns a p-value. Low p-value means the distributions are different.

The detector combines both metrics to assign a DriftSeverity to each feature, then produces
a DriftReport containing all the per-feature results.

Controlled by config.py

`tests/unit/test_detector.py` - Tests using the sample_baseline_data and sample_drifted_data
fixtures from conftest. Verifies that feature_0 (big shift) is flagged as severe, feature_1
(small shift) is flagged as low, and features 2-9 (no shift) are flagged as none.

---

### Phase 3 - Drift Diagnosis

**Goal:** Once drift is detected, figure out why. Which features matter most to the model,
and which of those have drifted the hardest.

**What was built:**

`src/diagnoser.py` - The DriftDiagnoser class. Takes a DriftReport and the current model.
Uses SHAP to compute feature importance scores. Then cross-references importance with drift severity.

`tests/unit/test_diagnoser.py` - Tests using FakeModel and pre-built DriftReports. Verifies
that high-importance drifted features rank above low-importance ones.

---

### Phase 4 - Self Healing

**Goal:** Automatically retrain a challenger model on recent data and decide whether to
promote it or keep the current champion.

**What will be built:**

`src/healer.py` - The Healer class. Takes a DiagnosisResult, the current champion model,
and training data. Calls retrain() on the model to produce a challenger. Evaluates both
champion and challenger on the same holdout set, comparing the two. The promotion decision uses a configurable threshold from config.py. The challenger must beat the champion by at least that margin (default 2% accuracy) to be promoted. Produces a HealingOutcome with both sets of metrics, the action taken (PROMOTE, ROLLBACK, or NO_ACTION), and a human-readable reason for the decision.

`tests/unit/test_healer.py` - Tests three scenarios: challenger wins (promote), challenger
loses (rollback), challenger wins by less than the threshold (no action). Uses FakeModel
with different accuracy configurations for each scenario.

---

### Phase 5 - Reporting and Visualization

**Goal:** Generate human-readable incident reports and charts summarizing what happened.

**What was built:**

`src/reporter.py` - The Reporter class. Takes a HealingOutcome and produces an IncidentReport.
Generates a text summary describing the full incident (what drifted, by how much, what the
model did about it, what the result was). Generates matplotlib/seaborn charts:

- Drift bar chart: PSI scores per feature, color-coded by severity
- Distribution comparison: baseline vs current distribution for drifted features
- Champion vs challenger: side-by-side metric comparison

Saves charts to the reports/ directory and records their file paths in the IncidentReport
charts dict.

`tests/unit/test_reporter.py` - Tests summary text generation (does it mention the right
features, the right numbers). Tests that chart file paths are populated. Tests edge cases
like no drift detected (should produce a clean report saying everything is fine).

---