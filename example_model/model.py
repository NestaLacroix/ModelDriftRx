"""PyTorch network used by the Phase 9 fraud-detection example."""

from __future__ import annotations

import torch
from torch import nn


class FraudDetectionNetwork(nn.Module):
	"""Small binary-classification network for synthetic transaction data."""

	def __init__(
		self,
		input_size: int,
		hidden_size: int = 32,
	) -> None:
		super().__init__()
		if input_size < 1:
			raise ValueError("input_size must be positive")
		if hidden_size < 1:
			raise ValueError("hidden_size must be positive")

		self.network = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1),
		)

	def forward(self, features: torch.Tensor) -> torch.Tensor:
		"""Return one fraud logit per input row."""
		return self.network(features).squeeze(-1)

