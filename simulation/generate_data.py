"""Generate reproducible baseline and drifted fraud data for simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from example_model.data.download import FEATURE_NAMES, generate_fraud_data


@dataclass(frozen=True)
class SimulationData:
    """Paired baseline and incoming transaction data for one demo run."""

    baseline_features: np.ndarray
    baseline_labels: np.ndarray
    incoming_features: np.ndarray
    incoming_labels: np.ndarray
    feature_names: list[str]


def generate_simulation_data(
    n_samples: int = 2000,
    drift_amount: float = 1.0,
    seed: int = 42,
) -> SimulationData:
    """Return matching clean and drifted datasets with stable feature names."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")
    if drift_amount <= 0:
        raise ValueError("drift_amount must be positive")

    baseline_features, baseline_labels = generate_fraud_data(
        n_samples=n_samples,
        seed=seed,
    )
    incoming_features, incoming_labels = generate_fraud_data(
        n_samples=n_samples,
        seed=seed,
        drift_amount=drift_amount,
    )
    return SimulationData(
        baseline_features=baseline_features,
        baseline_labels=baseline_labels,
        incoming_features=incoming_features,
        incoming_labels=incoming_labels,
        feature_names=FEATURE_NAMES.copy(),
    )
