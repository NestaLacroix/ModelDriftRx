"""
DriftRx API - Application state.

A single AppState dataclass holds every piece of mutable runtime data the
API needs:  the current champion model, the baseline dataset, incident log,
and timing metadata.

The module-level singleton (_app_state) is what the real server uses.
Tests override the FastAPI dependency (get_state) to inject a fresh
AppState per test, so real and test state never mix.

Usage in routers:
    from fastapi import Depends
    from api.state import AppState, get_state

    @router.get("/foo")
    def foo(state: AppState = Depends(get_state)):
        ...

Usage in tests:
    from api.main import app
    from api.state import AppState, get_state

    def override():
        s = AppState()
        s.model = FakeModel()
        return s

    app.dependency_overrides[get_state] = override
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from pathlib import Path

import numpy as np

from src.utils.config import CONFIG


@dataclass
class AppState:
    """All mutable runtime state for the API.  One instance per server process."""

    # The current champion model.  Any object with .predict() / .evaluate() / .retrain().
    model: Any = None
    model_name: str | None = None
    model_path: str | None = None

    # Baseline distribution used by DriftDetector.  Shape: (n_samples, n_features).
    baseline: np.ndarray | None = None

    # Feature column names - shared by the detector and returned in drift responses.
    feature_names: list[str] | None = None

    # In-memory incident log.  Each entry is the dict produced by IncidentReport.to_dict().
    incidents: list[dict[str, Any]] = field(default_factory=list)

    # PSI history consumed by the dashboard timeline.
    drift_history: list[dict[str, Any]] = field(default_factory=list)

    # Timestamp of the most recent /check-drift call (or None if never called).
    last_drift_check: datetime | None = None


# ---------------------------------------------------------------------------
# Module-level singleton used by the production server.
# ---------------------------------------------------------------------------

_app_state: AppState = AppState()


def get_state() -> AppState:
    """FastAPI dependency - returns the production AppState singleton."""
    return _app_state


def load_example_model(state: AppState) -> None:
    """Load the Phase 9 model and baseline when a checkpoint is available."""
    checkpoint_path = Path(CONFIG.model_dir) / "fraud_model.pt"
    if not checkpoint_path.exists():
        return

    if importlib.util.find_spec("torch") is None:
        return
    torch_check = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        check=False,
    )
    if torch_check.returncode != 0:
        return

    try:
        from example_model.data.download import FEATURE_NAMES, generate_fraud_data
        from example_model.wrapper import FraudModelWrapper

        state.model = FraudModelWrapper.load(str(checkpoint_path))
        state.model_name = "Fraud Detection Example"
        state.model_path = str(checkpoint_path)
        state.feature_names = FEATURE_NAMES.copy()
        state.baseline, _ = generate_fraud_data(n_samples=2000, seed=42)
    except Exception:
        # A missing or incompatible optional example model must not stop the API.
        state.model = None
        state.model_name = None
        state.model_path = None
        state.baseline = None
        state.feature_names = None


def reset_state() -> AppState:
    """
    Replace the singleton with a brand-new AppState and return it.
    Intended for test setup; never call this in production code.
    """
    global _app_state
    _app_state = AppState()
    return _app_state
