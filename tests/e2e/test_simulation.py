from __future__ import annotations

import json

import pytest

from simulation.run_simulation import run_simulation


def test_simulation_completes_drift_healing_and_reporting(tmp_path):
	result = run_simulation(
		n_samples=120,
		drift_amount=1.0,
		epochs=2,
		seed=14,
		reports_dir=str(tmp_path / "reports"),
		incident_store_path=str(tmp_path / "incidents.json"),
		model_output_path=str(tmp_path / "models" / "fraud.pt"),
	)

	assert result.drift_report.triggered_healing is True
	assert result.cycle is not None
	assert result.cycle.incident.summary
	assert result.cycle.incident.charts
	assert (tmp_path / "models" / "fraud.pt").exists()
	assert (tmp_path / "reports" / "simulation_result.json").exists()
	incidents = json.loads((tmp_path / "incidents.json").read_text(encoding="utf-8"))
	assert len(incidents) == 1
	assert incidents[0]["id"] == result.cycle.incident.id


def test_simulation_skips_healing_when_drift_threshold_not_crossed(tmp_path):
	result = run_simulation(
		n_samples=120,
		drift_amount=0.01,
		epochs=2,
		seed=14,
		reports_dir=str(tmp_path / "reports"),
		incident_store_path=str(tmp_path / "incidents.json"),
	)

	assert result.drift_report.triggered_healing is False
	assert result.cycle is None
	assert not (tmp_path / "incidents.json").exists()
	summary = json.loads(
		(tmp_path / "reports" / "simulation_result.json").read_text(encoding="utf-8")
	)
	assert summary["healing_started"] is False

