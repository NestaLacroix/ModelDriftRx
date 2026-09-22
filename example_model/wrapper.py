"""Protocol adapter for the example PyTorch fraud model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from example_model.model import FraudDetectionNetwork
from src.contracts import ModelMetrics


@dataclass(frozen=True)
class FraudModelConfig:
	"""Training settings used to recreate a challenger model."""

	hidden_size: int = 32
	learning_rate: float = 0.01
	epochs: int = 100
	seed: int = 42


class FraudModelWrapper:
	"""Make the PyTorch fraud network satisfy ``MonitorableModel``."""

	def __init__(
		self,
		input_size: int,
		config: FraudModelConfig | None = None,
		network: FraudDetectionNetwork | None = None,
	) -> None:
		self.input_size = input_size
		self.config = config or FraudModelConfig()
		self.network = network or FraudDetectionNetwork(
			input_size=input_size,
			hidden_size=self.config.hidden_size,
		)

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Return binary fraud predictions for a 2-D feature array."""
		features = self._features(x)
		self.network.eval()
		with torch.no_grad():
			probabilities = torch.sigmoid(self.network(features))
		return (probabilities >= 0.5).cpu().numpy().astype(int)

	def predict_proba(self, x: np.ndarray) -> np.ndarray:
		"""Return fraud probabilities for diagnostics and inspection."""
		features = self._features(x)
		self.network.eval()
		with torch.no_grad():
			probabilities = torch.sigmoid(self.network(features))
		return probabilities.cpu().numpy()

	def evaluate(self, x: np.ndarray, y: np.ndarray) -> ModelMetrics:
		"""Evaluate predictions and return the shared monitoring metrics."""
		labels = self._labels(y)
		predictions = self.predict(x)
		probabilities = self.predict_proba(x)

		true_positive = float(np.sum((predictions == 1) & (labels == 1)))
		true_negative = float(np.sum((predictions == 0) & (labels == 0)))
		false_positive = float(np.sum((predictions == 1) & (labels == 0)))
		false_negative = float(np.sum((predictions == 0) & (labels == 1)))

		accuracy = (true_positive + true_negative) / len(labels)
		precision = self._safe_divide(true_positive, true_positive + false_positive)
		recall = self._safe_divide(true_positive, true_positive + false_negative)
		f1 = self._safe_divide(2 * precision * recall, precision + recall)

		loss_fn = nn.BCEWithLogitsLoss()
		self.network.eval()
		with torch.no_grad():
			logits = self.network(self._features(x))
			loss = float(loss_fn(logits, torch.from_numpy(labels)).item())

		return ModelMetrics(
			accuracy=accuracy,
			f1=f1,
			precision=precision,
			recall=recall,
			loss=loss,
			extra={"positive_rate": float(np.mean(probabilities >= 0.5))},
		)

	def retrain(self, x: np.ndarray, y: np.ndarray) -> FraudModelWrapper:
		"""Train and return a new wrapper without mutating the champion."""
		challenger = FraudModelWrapper(
			input_size=self.input_size,
			config=self.config,
		)
		challenger._fit(x, y)
		return challenger

	def save(self, path: str) -> None:
		"""Save network weights and the settings needed to reload them."""
		torch.save(
			{
				"input_size": self.input_size,
				"config": self.config.__dict__,
				"state_dict": self.network.state_dict(),
			},
			path,
		)

	@classmethod
	def load(cls, path: str) -> FraudModelWrapper:
		"""Load a wrapper saved with :meth:`save`."""
		checkpoint = torch.load(path, map_location="cpu", weights_only=False)
		config = FraudModelConfig(**checkpoint["config"])
		network = FraudDetectionNetwork(
			input_size=checkpoint["input_size"],
			hidden_size=config.hidden_size,
		)
		network.load_state_dict(checkpoint["state_dict"])
		return cls(
			input_size=checkpoint["input_size"],
			config=config,
			network=network,
		)

	def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
		torch.manual_seed(self.config.seed)
		features = self._features(x)
		labels = self._labels(y)
		optimizer = torch.optim.Adam(
			self.network.parameters(),
			lr=self.config.learning_rate,
		)
		loss_fn = nn.BCEWithLogitsLoss()

		self.network.train()
		for _ in range(self.config.epochs):
			optimizer.zero_grad()
			loss = loss_fn(self.network(features), torch.from_numpy(labels))
			loss.backward()
			optimizer.step()

	def _features(self, x: np.ndarray) -> torch.Tensor:
		values = np.asarray(x, dtype=np.float32)
		if values.ndim != 2:
			raise ValueError("features must be a 2-D numpy array")
		if values.shape[1] != self.input_size:
			raise ValueError(
				f"features have {values.shape[1]} columns, expected {self.input_size}"
			)
		return torch.from_numpy(values)

	@staticmethod
	def _labels(y: np.ndarray) -> np.ndarray:
		labels = np.asarray(y, dtype=np.float32).reshape(-1)
		if labels.size == 0:
			raise ValueError("labels must not be empty")
		if not np.all(np.isin(labels, [0.0, 1.0])):
			raise ValueError("labels must contain only 0 and 1")
		return labels

	@staticmethod
	def _safe_divide(numerator: float, denominator: float) -> float:
		return numerator / denominator if denominator else 0.0

