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
