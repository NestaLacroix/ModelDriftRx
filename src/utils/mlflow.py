"""
DriftRx - Optional MLflow integration.

MLflow is loaded only when tracking is enabled so the monitoring pipeline can
run without an MLflow installation or tracking server.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.config import CONFIG


class MLflowTracker:
    """Log monitoring events to MLflow without making it a runtime requirement."""

    def __init__(self, config: Any = CONFIG) -> None:
        self._config = config

    def _load_mlflow(self) -> Any | None:
        if not self._config.mlflow_enabled:
            return None
        try:
            import mlflow
        except ImportError:
            return None
        return mlflow

    @staticmethod
    def _metric_value(metrics: Any, name: str) -> float:
        if isinstance(metrics, dict):
            return float(metrics.get(name, 0.0))
        return float(getattr(metrics, name, 0.0))

    @classmethod
    def _metric_values(cls, metrics: Any, prefix: str) -> dict[str, float]:
        raw_metrics = getattr(metrics, "raw", metrics)
        values = {
            f"{prefix}_accuracy": cls._metric_value(metrics, "accuracy"),
            f"{prefix}_f1": cls._metric_value(raw_metrics, "f1"),
            f"{prefix}_precision": cls._metric_value(raw_metrics, "precision"),
            f"{prefix}_recall": cls._metric_value(raw_metrics, "recall"),
            f"{prefix}_loss": cls._metric_value(raw_metrics, "loss"),
        }
        extra = (
            raw_metrics.get("extra", {})
            if isinstance(raw_metrics, dict)
            else getattr(raw_metrics, "extra", {})
        )
        if isinstance(extra, dict):
            values.update({f"{prefix}_{name}": float(value) for name, value in extra.items()})
        return values

    @staticmethod
    def _action_value(action: Any) -> str:
        return str(getattr(action, "value", action)).upper()

    def log_healing(self, outcome: Any) -> str | None:
        """Log model comparison metrics and return the MLflow run ID."""
        mlflow = self._load_mlflow()
        if mlflow is None:
            return None

        try:
            mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
            mlflow.set_experiment(self._config.mlflow_experiment_name)
            with mlflow.start_run() as run:
                champion = outcome.champion_metrics
                challenger = outcome.challenger_metrics
                metrics = {}
                metrics.update(self._metric_values(champion, "champion"))
                metrics.update(self._metric_values(challenger, "challenger"))
                metrics["improvement"] = challenger.accuracy - champion.accuracy
                mlflow.log_metrics(metrics)
                mlflow.set_tags(
                    {
                        "event_type": "healing",
                        "action": self._action_value(outcome.action),
                        "reason": outcome.reason,
                    }
                )
                if self._action_value(outcome.action) == "PROMOTE":
                    try:
                        python_model = _ModelWrapper(outcome.challenger)
                        mlflow.pyfunc.log_model(
                            artifact_path="model",
                            python_model=python_model,
                            input_example=getattr(
                                outcome,
                                "input_example",
                                np.zeros((1, 1), dtype=float),
                            ),
                        )
                        mlflow.register_model(
                            f"runs:/{run.info.run_id}/model",
                            self._config.mlflow_registry_name,
                        )
                    except Exception:
                        pass
                return run.info.run_id
        except Exception:
            return None

    def log_incident(self, report: Any) -> str | None:
        """Log incident metadata and generated chart artifacts."""
        mlflow = self._load_mlflow()
        if mlflow is None:
            return None

        try:
            mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
            mlflow.set_experiment(self._config.mlflow_experiment_name)
            with mlflow.start_run() as run:
                outcome = report.healing_outcome
                mlflow.set_tags(
                    {
                        "event_type": "incident",
                        "incident_id": report.id,
                        "action": self._action_value(outcome.action),
                        "overall_severity": (
                            outcome.diagnosis.drift_report.overall_severity.value
                        ),
                    }
                )
                for chart_path in report.charts.values():
                    mlflow.log_artifact(chart_path)
                return run.info.run_id
        except Exception:
            return None


class _ModelWrapper:
    """Adapt a MonitorableModel-like object to MLflow's pyfunc interface."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, context: Any, model_input: Any) -> np.ndarray:
        del context
        return np.asarray(self._model.predict(np.asarray(model_input)))

    def __call__(self, model_input: Any) -> np.ndarray:
        """Support MLflow versions that accept a callable pyfunc model."""
        return np.asarray(self._model.predict(np.asarray(model_input)))
