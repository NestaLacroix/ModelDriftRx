"""
DriftWatch - Core Data Contracts

Every component in the system communicates through these dataclasses:

Hierarchy:
    FeatureDrift        -> drift info for ONE feature
    DriftReport         -> drift info for ALL features (list of FeatureDrift)
    ModelMetrics        -> performance snapshot of a model
    DiagnosisResult     -> root cause analysis (wraps DriftReport)
    HealingOutcome      -> retrain result + promote/rollback decision
    IncidentReport      -> complete record of a healing event
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Enums

class DriftSeverity(Enum):
    # How severe the detected drift is for a feature or overall.

    NONE = "none"            # PSI < 0.1  — no meaningful change
    LOW = "low"              # PSI 0.1-0.2 — minor shift, monitor
    MODERATE = "moderate"    # PSI 0.2-0.3 — significant, consider action
    SEVERE = "severe"        # PSI > 0.3  — major shift, act immediately

    def __ge__(self, other: DriftSeverity) -> bool:
        # Allow severity comparisons: SEVERE >= MODERATE → True.
        order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MODERATE,
            DriftSeverity.SEVERE,
        ]
        return order.index(self) >= order.index(other)

    def __gt__(self, other: DriftSeverity) -> bool:
        order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MODERATE,
            DriftSeverity.SEVERE,
        ]
        return order.index(self) > order.index(other)

    def __le__(self, other: DriftSeverity) -> bool:
        return not self.__gt__(other)

    def __lt__(self, other: DriftSeverity) -> bool:
        return not self.__ge__(other)


class HealAction(Enum):
    # What the system decided to do after evaluating a challenger model.

    PROMOTE = "promote"        # Challenger is better -> swap it in
    ROLLBACK = "rollback"      # Challenger is worse -> keep champion
    NO_ACTION = "no_action"    # Drift wasn't severe enough to trigger healing


# ModelMetrics

@dataclass
class ModelMetrics:
    """
    Performance snapshot of a model at a point in time.
    Used for both champion and challenger evaluation. The `extra` field
    allows users to add custom metrics without modifying this class.
    """

    accuracy: float
    f1: float
    precision: float
    recall: float
    loss: float
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        base = {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "loss": self.loss,
        }
        # Merge extra metrics at top level for clean API responses
        base.update(self.extra)
        return base


# FeatureDrift

@dataclass
class FeatureDrift:
    """
    Drift information for a single feature/column.

    Produced by the Detector for each feature it scans. Contains both
    the statistical test results and the human-readable context.
    """

    feature_name: str
    psi_score: float
    ks_p_value: float
    severity: DriftSeverity
    baseline_mean: float
    current_mean: float

    @property
    def shift_percentage(self) -> float:
    # Percent change from baseline to current mean.
        if self.baseline_mean == 0:
            return 0.0
        return ((self.current_mean - self.baseline_mean) / abs(self.baseline_mean)) * 100

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "psi_score": self.psi_score,
            "ks_p_value": self.ks_p_value,
            "severity": self.severity.value,
            "baseline_mean": self.baseline_mean,
            "current_mean": self.current_mean,
            "shift_percentage": self.shift_percentage,
        }


# DriftReport

@dataclass
class DriftReport:
    # Complete drift analysis across all features.
    # Produced by the Detector, -> consumed by the Diagnoser.

    timestamp: datetime
    overall_severity: DriftSeverity
    feature_drifts: list[FeatureDrift]
    triggered_healing: bool

    @property
    def drifted_features(self) -> list[FeatureDrift]:
        # Return only features where drift was detected (severity > NONE).
        return [fd for fd in self.feature_drifts if fd.severity > DriftSeverity.NONE]

    def top_drifted_features(self, n: int = 3) -> list[FeatureDrift]:
        """
        Return the top N features sorted by PSI score (highest first).
        Used by the Diagnoser to focus root cause analysis on the
        features that drifted the most.
        """
        sorted_features = sorted(
            self.feature_drifts, key=lambda fd: fd.psi_score, reverse=True
        )
        return sorted_features[:n]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_severity": self.overall_severity.value,
            "feature_drifts": [fd.to_dict() for fd in self.feature_drifts],
            "triggered_healing": self.triggered_healing,
            "num_drifted_features": len(self.drifted_features),
        }


# DiagnosisResult

@dataclass
class DiagnosisResult:
    """
    Root cause analysis of a drift event.
    Extends the DriftReport with SHAP-based feature attribution to
    explain not just WHAT drifted, but WHICH drifted features actually
    caused the model's performance to degrade.

    Output of the Diagnoser -> Consumed by the Healer.
    """

    drift_report: DriftReport
    top_contributors: list[FeatureDrift]  # ranked by impact (SHAP x drift)
    estimated_accuracy_drop: float        # e.g., 0.15 means 15% drop
    root_cause_summary: str               # human-readable explanation

    def to_dict(self) -> dict:
        return {
            "drift_report": self.drift_report.to_dict(),
            "top_contributors": [fc.to_dict() for fc in self.top_contributors],
            "estimated_accuracy_drop": self.estimated_accuracy_drop,
            "root_cause_summary": self.root_cause_summary,
        }


# HealingOutcome

@dataclass
class HealingOutcome:
    """
    Result of the auto-retraining and champion vs. challenger evaluation.

    Contains everything needed to understand what happened:
    - What triggered the healing (diagnosis)
    - How both models performed (metrics)
    - What the system decided (action + reason)

    Output of the Healer, -> Consumed by the Reporter.
    """

    diagnosis: DiagnosisResult
    champion_metrics: ModelMetrics
    challenger_metrics: ModelMetrics
    action: HealAction
    reason: str

    @property
    def improvement(self) -> float:
        """
        Accuracy delta between challenger and champion.
        Positive = challenger is better.
        Negative = challenger is worse.
        """
        return self.challenger_metrics.accuracy - self.champion_metrics.accuracy

    def to_dict(self) -> dict:
        return {
            "diagnosis": self.diagnosis.to_dict(),
            "champion_metrics": self.champion_metrics.to_dict(),
            "challenger_metrics": self.challenger_metrics.to_dict(),
            "action": self.action.value,
            "reason": self.reason,
            "improvement": self.improvement,
        }


# IncidentReport

@dataclass
class IncidentReport:
    # Complete record of a healing event.

    healing_outcome: HealingOutcome
    timestamp: datetime
    summary: str                                          # plain English paragraph
    charts: dict[str, str] = field(default_factory=dict)  # Later will have driftplot.png
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "healing_outcome": self.healing_outcome.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "charts": self.charts,
        }
