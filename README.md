# ModelDriftRx
Detect that your model is dying, it diagnoses why, fixes itself, and proves the fix worked, all autonomously. Think of it as an immune system for ML models. Will feature a dashboard to monitor model health and more, development in progress...

## Development - current state & how to run

The core detection/diagnosis/healing pipeline is implemented and exposed via a FastAPI backend; a Streamlit dashboard provides a visual front-end that consumes the API. To run the current stack locally:

- Install dependencies (use the environment of your choice and install dev extras from `pyproject.toml`).
- Start the API service:

  `make run-api` runs FastAPI on http://localhost:8000

- Start the dashboard frontend:

  `make run-dashboard` or `streamlit run dashboard/app.py` runs Dashboard on http://localhost:8501


Notes: the dashboard falls back to deterministic synthetic data when the API is unreachable so it is always runnable for demos.

**Brief description of the AI "immune system" underneath**

- **Detect:** the detector compares incoming data to a baseline and computes per-feature metrics (PSI and KS p-value) to flag distribution shifts.
- **Diagnose:** the diagnoser ranks features by model importance (SHAP) and cross-references importance with drift severity to prioritise likely root causes.
- **Fix (Self-heal):** the healer retrains a challenger model on recent data, evaluates on a holdout set, and decides to `PROMOTE`, `ROLLBACK`, or take `NO_ACTION` based on a configurable improvement threshold.
- **Prove & Report:** the reporter generates human-readable incident summaries and charts (PSI bars, distribution comparisons, champion vs challenger) which are stored with the incident for auditing and review. All viewable on the dashboard.
- **Track:** optional MLflow integration records healing metrics, decisions, incident tags, and chart artifacts. Promoted challengers are registered in the MLflow Model Registry when `MLFLOW_ENABLED=true`.

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

### Phase 6 - FastAPI Service

**Goal:** Expose the monitoring system as a lightweight REST API for programmatic
integration, automated testing, and a backend for the dashboard.

**What was built:**

- `api/main.py` - FastAPI application with lifespan startup logic and CORS middleware.
- `api/schemas.py` - Pydantic v2 request/response models that drive OpenAPI docs.
- `api/state.py` - Small in-memory `AppState` holder and dependency for runtime state.
- `api/routers/*.py` - Router modules exposing the core endpoints:
  - `POST /predict` - run batch inference against the loaded champion model
  - `GET /health` - service status (model/baseline loaded, last check, incident count)
  - `POST /check-drift` - run detector against uploaded feature batches
  - `GET /incidents` and `GET /incidents/{id}` - incident summaries and details
- `tests/e2e/test_api.py` - end-to-end API tests (FastAPI TestClient + `FakeModel`) that
  exercise the endpoints and verify response contracts.

Notes: the API is developer-focused (UI at `/docs`) and returns realistic
results when `AppState` is populated by the demo or a loader script.

---

### Phase 7 - Streamlit Dashboard

**Goal:** Provide a lightweight, interactive UI to monitor model health, inspect recent
drift checks, and review self-healing outcomes.

**What was built:**

- `dashboard/app.py` — Streamlit app with four main pages:
  - **Health Overview:** service/model status, KPI cards, latest drift snapshot (PSI bar chart) and an action-distribution donut.
  - **Drift Timeline:** multi-feature PSI timeline (area chart) and the most-recent check's feature table.
  - **Incidents:** full incident history with expandable summaries and (when the API is online) detailed healing outcomes.
  - **Champion vs Challenger:** grouped bar chart comparing metrics, decision card, and a metric-level delta table.

- Sidebar controls: editable `API URL` (overrides `DRIFTRX_API_URL`), manual `Refresh` button, and an **Auto-refresh** toggle (fixed 30s interval; enabled by default).

- Dashboard fetches data from the FastAPI backend when available, and falls back to deterministic synthetic demo data when the API is unreachable so the UI is always runnable.

- Components: reusable Plotly chart generators (`psi_bar_chart`, `drift_timeline_chart`, `champion_challenger_chart`, `action_donut`), CSS theme and KPI/badge helpers in `dashboard/components`.

Notes: the dashboard is intentionally decoupled from the monitoring internals - it consumes the API and presents human-friendly visualisations.

---

### Phase 8 - MLflow Integration

**Goal:** Track retraining decisions, model metrics, promoted model versions, and incident
artifacts in MLflow while keeping MLflow optional for local development and deployments.

**What was built:**

- `src/utils/config.py` - MLflow settings loaded from environment variables:
  `MLFLOW_ENABLED`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, and
  `MLFLOW_REGISTRY_NAME`. Tracking is disabled by default.
- `src/utils/mlflow.py` - Lazy `MLflowTracker` that records healing and incident runs. The
  monitoring pipeline continues working when MLflow is disabled, not installed, or unavailable.
- `src/healer.py` - Logs champion and challenger accuracy, F1, precision, recall, loss, custom
  metrics, improvement, action, and decision reason. Promoted challengers are logged through an
  MLflow `pyfunc` adapter and registered in the Model Registry.
- `src/reporter.py` - Logs incident ID, action, severity, and generated chart artifacts. The
  resulting MLflow run ID is retained on the incident report.
- `src/contracts.py`, `api/schemas.py`, and `api/routers/history.py` - Preserve and expose
  `mlflow_run_id` through incident serialization and API responses.
- `tests/unit/test_mlflow.py` - Tests disabled tracking, metric logging, custom metrics,
  promotion registration, enum actions, incident tags, and chart artifacts with a fake MLflow
  client.

**Configuration:**

Copy the MLflow values from `.env.example` into `.env` when tracking is wanted:

```text
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=DriftRx
MLFLOW_REGISTRY_NAME=DriftRxModel
```

The default `MLFLOW_ENABLED=false` setting means existing tests and application workflows do not
require an MLflow server. MLflow errors are non-fatal and never prevent detection, healing, or
report generation.

**How to test it:**

Run the automated Phase 8 tests:

```powershell
pytest -q tests/unit/test_mlflow.py
```

Run the complete regression suite:

```powershell
pytest -q
```

To verify real local MLflow tracking, install the project dependencies, enable tracking, and
start the MLflow UI in a separate terminal:

```powershell
pip install -e ".[dev]"
$env:MLFLOW_ENABLED="true"
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
$env:MLFLOW_EXPERIMENT_NAME="DriftRx"
$env:MLFLOW_REGISTRY_NAME="DriftRxModel"
mlflow ui --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` to inspect the `DriftRx` experiment. A healing comparison creates a
run with champion/challenger metrics; a promoted challenger also creates a registered model.
Generated incident charts appear as run artifacts. The same environment variables must be set
in the terminal that starts the API or runs the healing workflow.

---

### Phase 9 - Example Model (Fraud Detection)

**Goal:** Provide a working PyTorch example that satisfies the model protocol and can be used
with DriftRx's monitoring and self-healing components.

Phase 9 supplies the concrete model adapter that earlier phases intentionally did not include.
The detector remains model-agnostic, while this example shows how a PyTorch model can plug into
the existing system without changing `src/` monitoring logic.

**What was built:**

- `example_model/model.py` - Small PyTorch binary-classification network for fraud detection.
- `example_model/data/download.py` - Deterministic synthetic transaction data with six named
  features and configurable drift for monitoring experiments.
- `example_model/wrapper.py` - `MonitorableModel` adapter implementing `predict`,
  `predict_proba`, `evaluate`, and non-mutating `retrain`. Evaluation returns the shared
  `ModelMetrics` contract, and checkpoints can be saved and loaded.
- `example_model/train.py` - Command-line trainer that saves a model checkpoint.
- `api/state.py` and `api/main.py` - Load `models/fraud_model.pt` and its six-feature baseline
  at API startup when the checkpoint is available. The API remains usable without the optional
  checkpoint.
- `dashboard/pages/health.py` - Displays the loaded example model name from the live API.
- `tests/unit/test_example_model.py` - Tests data generation, protocol compliance, predictions,
  evaluation, retraining, persistence, and checkpoint training.
- `tests/integration/test_example_model_pipeline.py` - Verifies the example wrapper can be passed
  into the existing `Healer` and produce a valid healing outcome.

**How it integrates:**

```text
generate_fraud_data()
  -> FraudModelWrapper
  -> MonitorableModel.predict/evaluate/retrain
  -> Healer champion/challenger comparison
  -> Reporter incident charts and summary
  -> optional MLflow metrics and model registration
```

The wrapper converts NumPy arrays into PyTorch tensors, returns binary predictions, calculates
the shared accuracy/F1/precision/recall/loss metrics, and creates a new challenger during
retraining. The champion is never mutated by `retrain`. Checkpoints preserve the network weights,
input size, and training configuration so the model can be loaded later.

**Train the example model:**

Activate the environment selected for this project first. In a new PowerShell terminal, replace
`modeldrift` with the exact environment name or path selected in VS Code:

```powershell
conda activate modeldrift
python -c "import sys; print(sys.executable)"
```

The printed path must belong to the `modeldrift` environment, not the base environment. Install
the project dependencies there so PyTorch is available:

```powershell
pip install -e ".[dev]"
```

Then train and save a checkpoint:

```powershell
python -m example_model.train --output models/fraud_model.pt
```

The generated data uses these features:

```text
transaction_amount, account_age_days, transaction_count,
credit_score, merchant_risk, distance_from_home
```

Run the Phase 9 tests with:

```powershell
pytest -q tests/unit/test_example_model.py
pytest -q tests/integration/test_example_model_pipeline.py
```

Both commands should run the PyTorch-specific tests instead of reporting them as skipped. If
`import torch` fails with `WinError 1114` or a `c10.dll` error, the terminal is using a broken
PyTorch installation. Reinstall the CPU build in the active environment:

```powershell
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__); print(torch.rand(1))"
```

Only run the training command after that import check succeeds:

```powershell
python -m example_model.train --output models/fraud_model.pt --samples 2000 --epochs 100
```

The model is an example adapter only. The framework-agnostic monitoring code in `src/` still
depends on the three-method `MonitorableModel` protocol rather than importing PyTorch directly.

---

### Phase 10 - Simulation and End-to-End Monitoring

**Goal:** Exercise and connect the lifecycle from generated drift through diagnosis, supervised
healing, incident reporting, API storage, dashboard views, and optional MLflow tracking.

**What was built:**

- `simulation/generate_data.py` - Produces reproducible baseline and shifted fraud datasets with
  the same six feature columns and aligned labels.
- `src/pipeline.py` - Runs diagnosis, splits labeled incoming data into training/holdout sets,
  calls the existing healer, adapts the result to shared contracts, and creates a Reporter
  incident. A promoted challenger becomes the returned champion.
- `simulation/run_simulation.py` - Trains a fresh champion on baseline data, injects drift,
  detects it, runs the monitoring cycle, writes chart files and `simulation_result.json`, and
  appends incident JSON. Optional arguments control sample count, epochs, drift size, paths, and
  checkpoint saving.
- `POST /check-drift` - Detection-only requests still work. If drift crosses the threshold and
  binary labels are supplied for every incoming row, the API runs the monitoring cycle, stores
  the incident, and updates its in-memory champion. Responses report `healing_started`,
  `healing_status`, `action`, `incident_id`, and `mlflow_run_id`. Without labels,
  `healing_status` is `labels_required` and no retraining occurs.
- `GET /drift-history` - Returns recorded checks for the live dashboard timeline.
- Dashboard Health, Drift Timeline, and Champion vs Challenger pages use real API status,
  history, and incident outcomes. A live API with no records shows empty states; synthetic
  samples are only used when the API is offline.
- Tests cover reproducible data, drift/no-drift simulation, the shared cycle, labeled API
  healing, history responses, and dashboard data mapping.

**Run the standalone simulation:**

In the `modeldrift` environment:

```powershell
python -m simulation.run_simulation --samples 600 --epochs 25 --drift-amount 1.0
```

The console prints drift severity, healing decision, incident ID, and MLflow run ID when
available. Output is written under `reports/`; the default incident log is
`reports/incidents.json`. `make demo` runs the simulation with default settings.

**Run the full API-to-dashboard cycle:**

Start MLflow, API, and dashboard as described in the local run guide. Then submit features and
matching binary labels to `POST /check-drift`. When drift is severe, the response indicates that
healing completed, and the new incident appears in the dashboard's incident count, action chart,
timeline, and champion/challenger page. Without labels, only detection runs.