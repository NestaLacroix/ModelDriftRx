from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from example_model.data.download import FEATURE_NAMES, generate_fraud_data


def test_generate_fraud_data_is_deterministic():
    first_x, first_y = generate_fraud_data(100, seed=9)
    second_x, second_y = generate_fraud_data(100, seed=9)

    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)
    assert first_x.shape == (100, len(FEATURE_NAMES))
    assert set(np.unique(first_y)).issubset({0.0, 1.0})


def test_generate_fraud_data_drift_changes_distribution():
    baseline, _ = generate_fraud_data(500, seed=9)
    drifted, _ = generate_fraud_data(500, seed=9, drift_amount=0.4)

    assert drifted[:, 0].mean() > baseline[:, 0].mean()
    assert drifted[:, 4].mean() > baseline[:, 4].mean()


def test_generate_fraud_data_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n_samples"):
        generate_fraud_data(0)
    with pytest.raises(ValueError, match="fraud_rate"):
        generate_fraud_data(10, fraud_rate=1.0)


def _torch_is_usable() -> bool:
    """Check the native PyTorch import in a child process."""
    if importlib.util.find_spec("torch") is None:
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


_TORCH_AVAILABLE = _torch_is_usable()

if _TORCH_AVAILABLE:
    from example_model.train import train_example_model
    from example_model.wrapper import FraudModelConfig, FraudModelWrapper

if not _TORCH_AVAILABLE:
    FraudModelConfig = None
    FraudModelWrapper = None
    train_example_model = None

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is not installed or usable")
def test_wrapper_satisfies_model_protocol():
    from src.protocols import MonitorableModel

    model = FraudModelWrapper(6, FraudModelConfig(epochs=2))
    assert isinstance(model, MonitorableModel)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is not installed or usable")
def test_wrapper_predict_and_evaluate():
    features, labels = generate_fraud_data(80, seed=3)
    model = FraudModelWrapper(6, FraudModelConfig(epochs=3))
    predictions = model.predict(features)
    metrics = model.evaluate(features, labels)

    assert predictions.shape == (80,)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert "positive_rate" in metrics.extra


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is not installed or usable")
def test_retrain_returns_new_model(tmp_path):
    features, labels = generate_fraud_data(100, seed=4)
    model = FraudModelWrapper(6, FraudModelConfig(epochs=3))
    challenger = model.retrain(features, labels)
    path = tmp_path / "fraud_model.pt"
    challenger.save(str(path))
    loaded = FraudModelWrapper.load(str(path))

    assert challenger is not model
    assert path.exists()
    assert loaded.predict(features).shape == (100,)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is not installed or usable")
def test_train_example_model_saves_checkpoint(tmp_path):
    path = tmp_path / "trained.pt"
    model, names = train_example_model(
        output_path=str(path),
        n_samples=100,
        epochs=2,
    )

    assert path.exists()
    assert names == FEATURE_NAMES
    assert model.input_size == len(FEATURE_NAMES)
