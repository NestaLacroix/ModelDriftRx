"""Generate deterministic synthetic credit-card transaction data."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
	"transaction_amount",
	"account_age_days",
	"transaction_count",
	"credit_score",
	"merchant_risk",
	"distance_from_home",
]


def generate_fraud_data(
	n_samples: int = 1000,
	seed: int = 42,
	drift_amount: float = 0.0,
	fraud_rate: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return synthetic transaction features and binary fraud labels.

	``drift_amount`` shifts transaction amount and merchant risk so the same
	generator can produce baseline and incoming monitoring datasets.
	"""
	if n_samples < 1:
		raise ValueError("n_samples must be positive")
	if not 0 < fraud_rate < 1:
		raise ValueError("fraud_rate must be between 0 and 1")

	rng = np.random.default_rng(seed)
	features = np.column_stack(
		[
			rng.lognormal(mean=4.5 + drift_amount, sigma=0.7, size=n_samples),
			rng.normal(900, 320, n_samples),
			rng.poisson(8, n_samples).astype(float),
			rng.normal(680, 45, n_samples),
			np.clip(rng.beta(2, 5, n_samples) + drift_amount * 0.08, 0, 1),
			rng.exponential(18, n_samples),
		]
	).astype(np.float32)

	score = (
		0.9 * np.log1p(features[:, 0])
		- 0.0015 * features[:, 1]
		+ 0.08 * features[:, 2]
		- 0.004 * features[:, 3]
		+ 2.2 * features[:, 4]
		+ 0.015 * features[:, 5]
	)
	threshold = np.quantile(score, 1 - fraud_rate)
	labels = (score >= threshold).astype(np.float32)
	return features, labels

