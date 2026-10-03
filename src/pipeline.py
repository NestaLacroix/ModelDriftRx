"""Coordinate one complete drift diagnosis, healing, and reporting cycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contracts import (
    DiagnosisResult,
    DriftReport,
    HealAction,
    HealingOutcome as ContractHealingOutcome,
    IncidentReport,
    ModelMetrics,
)
from src.diagnoser import Diagnoser
from src.healer import Healer
from src.reporter import Reporter
from src.utils.mlflow import MLflowTracker


@dataclass
class MonitoringCycleResult:
    """Contracts produced by one completed monitoring cycle."""

    diagnosis: DiagnosisResult
    healing_outcome: ContractHealingOutcome
    incident: IncidentReport
    champion_model: Any


def run_monitoring_cycle(
    drift_report: DriftReport,
    champion_model: Any,
    baseline: np.ndarray,
    incoming: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    reporter: Reporter | None = None,
    tracker: MLflowTracker | None = None,
) -> MonitoringCycleResult:
    """Run diagnosis, retraining, evaluation, and incident reporting.

    Labeled incoming rows are split in order into a challenger training set and
    holdout set. A new champion is returned only when the healer promotes it.
    """
    labels = np.asarray(labels).reshape(-1)
    if incoming.ndim != 2 or baseline.ndim != 2:
        raise ValueError("baseline and incoming data must be 2-D arrays")
    if incoming.shape[0] != labels.size:
        raise ValueError("incoming row count must match labels")
    if incoming.shape[0] < 2:
        raise ValueError("at least two labeled incoming rows are required")
    if incoming.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match data columns")

    diagnosis_report = Diagnoser(baseline, feature_names).diagnose(
        incoming,
        top_k=min(3, len(feature_names)),
    )
    drift_by_name = {item.feature_name: item for item in drift_report.feature_drifts}
    contributors = [
        drift_by_name[item.feature_name]
        for item in diagnosis_report.feature_diagnoses
        if item.feature_name in drift_by_name
    ]
    contributor_names = [item.feature_name for item in contributors]
    root_cause = (
        "Most shifted features: " + ", ".join(contributor_names)
        if contributor_names
        else "No dominant drifted features were identified."
    )
    diagnosis = DiagnosisResult(
        drift_report=drift_report,
        top_contributors=contributors,
        estimated_accuracy_drop=0.0,
        root_cause_summary=root_cause,
    )

    split_index = max(1, min(incoming.shape[0] - 1, int(incoming.shape[0] * 0.7)))
    healing = Healer(champion_model, tracker=tracker).heal(
        incoming[:split_index],
        labels[:split_index],
        incoming[split_index:],
        labels[split_index:],
    )
    contract_outcome = ContractHealingOutcome(
        diagnosis=diagnosis,
        champion_metrics=_contract_metrics(healing.champion_metrics.raw),
        challenger_metrics=_contract_metrics(healing.challenger_metrics.raw),
        action=HealAction(healing.action.lower()),
        reason=healing.reason,
    )
    report = (reporter or Reporter(tracker=tracker)).generate(
        contract_outcome,
        baseline=baseline,
        current=incoming,
        feature_names=feature_names,
    )
    promoted_model = (
        healing.challenger if healing.action == HealAction.PROMOTE.name else champion_model
    )
    return MonitoringCycleResult(
        diagnosis=diagnosis,
        healing_outcome=contract_outcome,
        incident=report,
        champion_model=promoted_model,
    )


def _contract_metrics(metrics: Any) -> ModelMetrics:
    """Normalize model evaluation output to the shared metrics contract."""
    if isinstance(metrics, ModelMetrics):
        return metrics
    if isinstance(metrics, dict):
        extra = metrics.get("extra", {})
        return ModelMetrics(
            accuracy=float(metrics.get("accuracy", 0.0)),
            f1=float(metrics.get("f1", 0.0)),
            precision=float(metrics.get("precision", 0.0)),
            recall=float(metrics.get("recall", 0.0)),
            loss=float(metrics.get("loss", 0.0)),
            extra=extra if isinstance(extra, dict) else {},
        )
    raise TypeError("model.evaluate() must return ModelMetrics or a metric dictionary")
