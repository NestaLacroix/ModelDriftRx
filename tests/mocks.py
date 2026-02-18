# DriftRx - Mock model for testing.

from __future__ import annotations

import numpy as np

from src.contracts import ModelMetrics


class FakeModel:
    def __init__(
        self,
        accuracy: float = 0.95,
        f1: float = 0.90,
        precision: float = 0.88,
        recall: float = 0.92,
        loss: float = 0.1,
        retrain_accuracy_boost: float = 0.05,
    ) -> None:
        self.accuracy = accuracy
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.loss = loss
        self._retrain_accuracy_boost = retrain_accuracy_boost

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Returns array of ones to imitate model predictions."""
        return np.ones(len(x), dtype=int)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Returns the configured metrics"""
        return ModelMetrics(
            accuracy=self.accuracy,
            f1=self.f1,
            precision=self.precision,
            recall=self.recall,
            loss=self.loss,
        )

    def retrain(self, x: np.ndarray, y: np.ndarray) -> FakeModel:
        """Returns a new FakeModel with slightly better accuracy."""
        return FakeModel(
            accuracy=min(self.accuracy + self._retrain_accuracy_boost, 1.0),
            f1=self.f1,
            precision=self.precision,
            recall=self.recall,
            loss=max(self.loss - 0.02, 0.0),
            retrain_accuracy_boost=self._retrain_accuracy_boost,
        )
