"""
DriftWatch — Central Configuration

All thresholds, paths, and settings live here. Nothing is hardcoded
anywhere else in the project. Values are loaded from environment
variables (via .env file).

Usage:
    from src.utils.config import CONFIG
    if psi > CONFIG.psi_severe:
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


def _env_float(key: str, default: float) -> float:
    """Read a float from environment variables, falling back to default."""
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int) -> int:
    """Read an int from environment variables, falling back to default."""
    return int(os.getenv(key, str(default)))


def _env_str(key: str, default: str) -> str:
    """Read a string from environment variables, falling back to default."""
    return os.getenv(key, default)


@dataclass(frozen=True)
class DriftWatchConfig:
    """
    Immutable configuration for the entire DriftWatch system.
    Frozen so nothing can accidentally mutate thresholds at runtime.
    All values can be overridden via environment variables.
    """

    # Drift Detection Thresholds (PSI):
    #
    # Population Stability Index ranges:
    #   < 0.1          -> no meaningful drift
    #   0.1  to 0.2    -> low drift, monitor closely
    #   0.2  to 0.3    -> moderate drift, consider retraining
    #   > 0.3          -> severe drift, retrain immediately
    
    psi_none: float = field(default_factory=lambda: _env_float("PSI_NONE", 0.1))
    psi_low: float = field(default_factory=lambda: _env_float("PSI_LOW", 0.2))
    psi_moderate: float = field(default_factory=lambda: _env_float("PSI_MODERATE", 0.3))
    # Anything above psi_moderate is considered SEVERE

    # Drift Detection Thresholds (KS Test):
    #
    # Kolmogorov-Smirnov test p-value:
    #   p > 0.05  -> distributions are similar (no drift)
    #   p < 0.05  -> distributions differ significantly (drift)
    
    ks_significance_level: float = field(
        default_factory=lambda: _env_float("KS_SIGNIFICANCE_LEVEL", 0.05)
    )

    # Drift Detection — Healing Trigger:
    #
    # Minimum severity level that triggers the healing pipeline.
    # Options: "moderate" or "severe"
    # "moderate" = more aggressive (heals sooner)
    # "severe"   = more conservative (only heals on major drift)
    
    healing_trigger_severity: str = field(
        default_factory=lambda: _env_str("HEALING_TRIGGER_SEVERITY", "moderate")
    )

    # Performance Monitoring:
    #
    # If accuracy drops by more than this fraction compared to baseline,
    # flag performance.
    # This catches concept drift (data looks the same but labels changed).
    #
    # 0.05 = flag if accuracy drops by more than 5%
    
    accuracy_drop_threshold: float = field(
        default_factory=lambda: _env_float("ACCURACY_DROP_THRESHOLD", 0.05)
    )

    # Champion vs. Challenger:
    #
    # Makes sure new model is more accurate:
    # Minimum accuracy improvement required to promote a challenger.
    # Prevents promoting models that are only marginally better (noise)
    #
    # 0.02 = challenger must be at least 2% better than champion
    #
    min_improvement: float = field(
        default_factory=lambda: _env_float("MIN_IMPROVEMENT", 0.02)
    )

    # SHAP Explainability:
    #
    # Number of samples to use for SHAP explanations.
    # SHAP used for diagnosis - is slow so we can't use whole data set
    # Sampling keeps diagnosis fast while still being representative
    
    shap_sample_size: int = field(
        default_factory=lambda: _env_int("SHAP_SAMPLE_SIZE", 100)
    )

    # Retraining:
    #
    # How many recent data points to use when retraining a challenger.
    # Too small = challenger undertrained
    # Too large = includes stale data from before the drift
    
    retrain_window_size: int = field(
        default_factory=lambda: _env_int("RETRAIN_WINDOW_SIZE", 5000)
    )

    # PSI Binning:
    #
    # Number of bins to use when computing PSI.
    # More bins = more sensitive to small distribution changes
    # Fewer bins = more robust but less granular
    
    psi_num_bins: int = field(
        default_factory=lambda: _env_int("PSI_NUM_BINS", 10)
    )

    # Paths:

    model_dir: str = field(
        default_factory=lambda: _env_str("MODEL_DIR", "models")
    )
    model_archive_dir: str = field(
        default_factory=lambda: _env_str("MODEL_ARCHIVE_DIR", "models/archive")
    )
    reports_dir: str = field(
        default_factory=lambda: _env_str("REPORTS_DIR", "reports")
    )
    incident_store_path: str = field(
        default_factory=lambda: _env_str("INCIDENT_STORE_PATH", "reports/incidents.json")
    )

    # MLflow:

    mlflow_tracking_uri: str = field(
        default_factory=lambda: _env_str("MLFLOW_TRACKING_URI", "file:./mlruns")
    )
    mlflow_experiment_name: str = field(
        default_factory=lambda: _env_str("MLFLOW_EXPERIMENT_NAME", "driftwatch")
    )

    # Utility Methods:

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [
            self.model_dir,
            self.model_archive_dir,
            self.reports_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Singleton:
# everywhere else: from src.utils.config import CONFIG
CONFIG = DriftWatchConfig()