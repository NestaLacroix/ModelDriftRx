"""
Healer - retrain challenger models and decide promote/rollback/no-action

This implementation is conservative:
- Calls 'retrain' on the champion to produce a challenger instance
- Calls 'evaluate' on both champion and challenger using the provided holdout set
- Uses CONFIG.min_improvement to decide promotion
- Returns a HealingOutcome dataclass
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.utils.config import CONFIG


@dataclass
class ModelMetricsSummary:
    accuracy: float
    raw: Any


@dataclass
class HealingOutcome:
    timestamp_utc: float
    champion_metrics: ModelMetricsSummary
    challenger_metrics: ModelMetricsSummary
    action: str  # "PROMOTE" | "ROLLBACK" | "NO_ACTION"
    reason: str
    champion: Any
    challenger: Any


class Healer:
    def __init__(self, champion_model: Any) -> None:
        self.champion = champion_model

    def _extract_accuracy(self, metrics: Any) -> float:
        """
        Accept either a dict-like metrics {'accuracy': ...} or an object with .accuracy attribute.
        Fall back to 0.0 if not found to keep failures explicit and testable.
        """
        if metrics is None:
            return 0.0
        if isinstance(metrics, dict):
            return float(metrics.get("accuracy", 0.0))
        # duck-type: object with attribute
        if hasattr(metrics, "accuracy"):
            try:
                return float(metrics.accuracy)
            except Exception:
                return 0.0
        # last resort: try attribute-like lookup
        try:
            return float(metrics["accuracy"])  # type: ignore[index]
        except Exception:
            return 0.0

    def _evaluate_model(self, model: Any, x_holdout: Any, y_holdout: Any) -> ModelMetricsSummary:
        if not hasattr(model, "evaluate"):
            raise AttributeError("Model does not implement evaluate(x, y)")

        raw_metrics = model.evaluate(x_holdout, y_holdout)
        acc = self._extract_accuracy(raw_metrics)
        return ModelMetricsSummary(accuracy=acc, raw=raw_metrics)

    def _retrain_challenger(self, model: Any, x_train: Any, y_train: Any) -> Any:
        if not hasattr(model, "retrain"):
            raise AttributeError("Model does not implement retrain(x, y)")

        return model.retrain(x_train, y_train)

    def heal(
        self,
        x_train: Any,
        y_train: Any,
        x_holdout: Any,
        y_holdout: Any,
    ) -> HealingOutcome:
        """
        Perform healing:
        1. Retrain a challenger from the champion using x_train/y_train
        2. Evaluate both champion and challenger on the same holdout set
        3. Compare accuracy improvement to CONFIG.min_improvement
        4. Return HealingOutcome with decision and reason
        """
        # Retrain -> challenger
        challenger = self._retrain_challenger(self.champion, x_train, y_train)

        # Evaluate both
        champion_metrics = self._evaluate_model(self.champion, x_holdout, y_holdout)
        challenger_metrics = self._evaluate_model(challenger, x_holdout, y_holdout)

        # Decision logic
        acc_champion = champion_metrics.accuracy
        acc_challenger = challenger_metrics.accuracy
        improvement = acc_challenger - acc_champion

        min_req = float(CONFIG.min_improvement)

        if improvement >= min_req:
            action = "PROMOTE"
            reason = (
                f"Challenger accuracy {acc_challenger:.4f} >= champion {acc_champion:.4f} "
                f"+ min_improvement ({min_req:.4f}); promote."
            )
        elif improvement < 0:
            action = "ROLLBACK"
            reason = (
                f"Challenger accuracy {acc_challenger:.4f} is worse than champion "
                f"{acc_champion:.4f}; rollback."
            )
        else:
            action = "NO_ACTION"
            reason = (
                f"Challenger improved by {improvement:.4f} which is less than "
                f"min_improvement ({min_req:.4f}); no action taken."
            )

        outcome = HealingOutcome(
            timestamp_utc=datetime.now(UTC).timestamp(),
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            action=action,
            reason=reason,
            champion=self.champion,
            challenger=challenger,
        )

        return outcome
