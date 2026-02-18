"""
DriftRx - Model Interface Protocol

Defines the contract any model must satisfy to be monitored.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from src.contracts import ModelMetrics


@runtime_checkable
class MonitorableModel(Protocol):
    """Any model that wants monitoring needs these 3 methods."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Takes features (n_samples, n_features), returns predictions (n_samples)."""
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Takes features + labels, returns ModelMetrics."""
        pass

    def retrain(self, x: np.ndarray, y: np.ndarray) -> MonitorableModel:
        """Returns a NEW trained model. Does not mutate self."""
        pass
