"""Train and save the Phase 9 fraud-detection example model."""

from __future__ import annotations

import argparse
from pathlib import Path

from example_model.data.download import FEATURE_NAMES, generate_fraud_data
from example_model.wrapper import FraudModelConfig, FraudModelWrapper


def train_example_model(
	output_path: str = "models/fraud_model.pt",
	n_samples: int = 2000,
	epochs: int = 100,
	seed: int = 42,
) -> tuple[FraudModelWrapper, list[str]]:
	"""Train the baseline fraud model and save its checkpoint."""
	features, labels = generate_fraud_data(n_samples=n_samples, seed=seed)
	config = FraudModelConfig(epochs=epochs, seed=seed)
	model = FraudModelWrapper(input_size=features.shape[1], config=config)
	model = model.retrain(features, labels)
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	model.save(output_path)
	return model, FEATURE_NAMES.copy()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", default="models/fraud_model.pt")
	parser.add_argument("--samples", type=int, default=2000)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	model, _ = train_example_model(
		output_path=args.output,
		n_samples=args.samples,
		epochs=args.epochs,
		seed=args.seed,
	)
	features, labels = generate_fraud_data(n_samples=args.samples, seed=args.seed)
	print(f"Saved fraud model to {args.output}")
	print(model.evaluate(features, labels).to_dict())


if __name__ == "__main__":
	main()

