"""Run a complete fraud-model drift, healing, and reporting simulation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from example_model.wrapper import FraudModelConfig, FraudModelWrapper
from simulation.generate_data import SimulationData, generate_simulation_data
from src.contracts import DriftReport
from src.detector import DriftDetector
from src.pipeline import MonitoringCycleResult, run_monitoring_cycle
from src.utils.config import CONFIG


@dataclass
class SimulationResult:
	"""Outputs from one complete or detection-only simulation run."""

	drift_report: DriftReport
	champion_model: FraudModelWrapper
	cycle: MonitoringCycleResult | None
	data: SimulationData


def run_simulation(
	n_samples: int = 2000,
	drift_amount: float = 1.0,
	epochs: int = 100,
	seed: int = 42,
	reports_dir: str | None = None,
	incident_store_path: str | None = None,
	model_output_path: str | None = None,
) -> SimulationResult:
	"""Train on a clean baseline, inject drift, heal, and generate a report."""
	data = generate_simulation_data(
		n_samples=n_samples,
		drift_amount=drift_amount,
		seed=seed,
	)
	model_config = FraudModelConfig(epochs=epochs, seed=seed)
	untrained_model = FraudModelWrapper(
		input_size=data.baseline_features.shape[1],
		config=model_config,
	)
	champion_model = untrained_model.retrain(
		data.baseline_features,
		data.baseline_labels,
	)
	if model_output_path:
		Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
		champion_model.save(model_output_path)

	drift_report = DriftDetector(
		data.baseline_features,
		feature_names=data.feature_names,
	).check(data.incoming_features)

	cycle = None
	if drift_report.triggered_healing:
		cycle = run_monitoring_cycle(
			drift_report=drift_report,
			champion_model=champion_model,
			baseline=data.baseline_features,
			incoming=data.incoming_features,
			labels=data.incoming_labels,
			feature_names=data.feature_names,
		)
		champion_model = cycle.champion_model
		_save_incident(cycle.incident.to_dict(), incident_store_path)
		if model_output_path and cycle.healing_outcome.action.value == "promote":
			champion_model.save(model_output_path)

	result = SimulationResult(
		drift_report=drift_report,
		champion_model=champion_model,
		cycle=cycle,
		data=data,
	)
	_save_run_summary(result, reports_dir)
	return result


def _save_incident(incident: dict[str, Any], store_path: str | None) -> None:
	path = Path(store_path or CONFIG.incident_store_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	incidents: list[dict[str, Any]] = []
	if path.exists():
		try:
			existing = json.loads(path.read_text(encoding="utf-8"))
			if isinstance(existing, list):
				incidents = existing
		except (OSError, json.JSONDecodeError):
			incidents = []
	incidents.append(incident)
	path.write_text(json.dumps(incidents, indent=2), encoding="utf-8")


def _save_run_summary(result: SimulationResult, reports_dir: str | None) -> Path:
	output_dir = Path(reports_dir or CONFIG.reports_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	summary: dict[str, Any] = {
		"drift_report": result.drift_report.to_dict(),
		"healing_triggered": result.drift_report.triggered_healing,
		"healing_started": result.cycle is not None,
	}
	if result.cycle is not None:
		summary["incident"] = result.cycle.incident.to_dict()
	path = output_dir / "simulation_result.json"
	path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	return path


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--samples", type=int, default=2000)
	parser.add_argument("--drift-amount", type=float, default=1.0)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--reports-dir", default=None)
	parser.add_argument("--incident-store", default=None)
	parser.add_argument("--save-model", default=None)
	args = parser.parse_args()

	result = run_simulation(
		n_samples=args.samples,
		drift_amount=args.drift_amount,
		epochs=args.epochs,
		seed=args.seed,
		reports_dir=args.reports_dir,
		incident_store_path=args.incident_store,
		model_output_path=args.save_model,
	)
	print(f"Overall drift severity: {result.drift_report.overall_severity.value}")
	print(f"Drifted features: {len(result.drift_report.drifted_features)}")
	print(f"Healing threshold crossed: {result.drift_report.triggered_healing}")
	if result.cycle is not None:
		incident = result.cycle.incident
		print(f"Healing action: {result.cycle.healing_outcome.action.value}")
		print(f"Incident ID: {incident.id}")
		print(f"MLflow run ID: {incident.mlflow_run_id or 'disabled or unavailable'}")
		print(f"Incident summary: {incident.summary}")
		print(f"Incident store: {args.incident_store or CONFIG.incident_store_path}")
	else:
		print("Healing was not run because the drift threshold was not crossed.")
	print(f"Simulation report: {Path(args.reports_dir or CONFIG.reports_dir) / 'simulation_result.json'}")


if __name__ == "__main__":
	main()

