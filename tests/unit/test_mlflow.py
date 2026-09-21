from __future__ import annotations

from types import SimpleNamespace

from src.utils.config import DriftRxConfig
from src.utils.mlflow import MLflowTracker


class FakeRun:
    def __init__(self):
        self.info = SimpleNamespace(run_id="run-123")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeMLflow:
    def __init__(self):
        self.metrics = {}
        self.tags = {}
        self.artifacts = []
        self.registered = None
        self.pyfunc = SimpleNamespace(log_model=self.log_model)

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment_name = name

    def start_run(self):
        return FakeRun()

    def log_metrics(self, metrics):
        self.metrics = metrics

    def set_tags(self, tags):
        self.tags = tags

    def log_model(self, artifact_path, python_model, input_example=None):
        self.model_artifact = (artifact_path, python_model, input_example)

    def register_model(self, model_uri, name):
        self.registered = (model_uri, name)

    def log_artifact(self, path):
        self.artifacts.append(path)


def _outcome(action="PROMOTE"):
    return SimpleNamespace(
        champion_metrics=SimpleNamespace(
            accuracy=0.80,
            raw={"f1": 0.75},
        ),
        challenger_metrics=SimpleNamespace(
            accuracy=0.86,
            raw={"f1": 0.84},
        ),
        action=action,
        reason="test decision",
        challenger=SimpleNamespace(predict=lambda values: values),
    )


def test_disabled_tracking_is_a_no_op():
    tracker = MLflowTracker(DriftRxConfig(mlflow_enabled=False))

    assert tracker.log_healing(_outcome()) is None


def test_promoted_model_is_logged_and_registered(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)
    config = DriftRxConfig(
        mlflow_enabled=True,
        mlflow_tracking_uri="file:./test-mlruns",
        mlflow_experiment_name="TestExperiment",
        mlflow_registry_name="TestModel",
    )

    run_id = MLflowTracker(config).log_healing(_outcome())

    assert run_id == "run-123"
    assert fake_mlflow.metrics["champion_accuracy"] == 0.80
    assert fake_mlflow.metrics["challenger_f1"] == 0.84
    assert fake_mlflow.metrics["champion_precision"] == 0.0
    assert fake_mlflow.metrics["challenger_loss"] == 0.0
    assert fake_mlflow.tags["action"] == "PROMOTE"
    assert fake_mlflow.registered == ("runs:/run-123/model", "TestModel")


def test_custom_metrics_are_logged(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)
    outcome = _outcome()
    outcome.challenger_metrics.raw = {
        "accuracy": 0.86,
        "f1": 0.84,
        "precision": 0.83,
        "recall": 0.85,
        "loss": 0.14,
        "extra": {"auc": 0.91},
    }

    MLflowTracker(DriftRxConfig(mlflow_enabled=True)).log_healing(outcome)

    assert fake_mlflow.metrics["challenger_auc"] == 0.91


def test_non_promoted_model_is_not_registered(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)
    config = DriftRxConfig(mlflow_enabled=True)

    run_id = MLflowTracker(config).log_healing(_outcome(action="NO_ACTION"))

    assert run_id == "run-123"
    assert fake_mlflow.registered is None


def test_incident_metadata_and_artifacts_are_logged(monkeypatch, tmp_path):
    fake_mlflow = FakeMLflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)
    chart = tmp_path / "drift.png"
    chart.write_bytes(b"chart")
    report = SimpleNamespace(
        id="incident-123",
        charts={"drift_bar": str(chart)},
        healing_outcome=SimpleNamespace(
            action="PROMOTE",
            diagnosis=SimpleNamespace(
                drift_report=SimpleNamespace(
                    overall_severity=SimpleNamespace(value="severe")
                )
            ),
        ),
    )

    run_id = MLflowTracker(DriftRxConfig(mlflow_enabled=True)).log_incident(report)

    assert run_id == "run-123"
    assert fake_mlflow.tags["incident_id"] == "incident-123"
    assert fake_mlflow.tags["overall_severity"] == "severe"
    assert fake_mlflow.artifacts == [str(chart)]
