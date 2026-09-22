from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from example_model.data.download import generate_fraud_data


def _torch_is_usable() -> bool:
    """Check the native PyTorch import without crashing test collection."""
    if importlib.util.find_spec("torch") is None:
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.skipif(not _torch_is_usable(), reason="PyTorch is not usable")
def test_example_model_runs_through_healer():
    from example_model.wrapper import FraudModelConfig, FraudModelWrapper
    from src.healer import Healer

    features, labels = generate_fraud_data(120, seed=12)
    model = FraudModelWrapper(6, FraudModelConfig(epochs=3, seed=12))
    outcome = Healer(model).heal(features, labels, features, labels)

    assert outcome.action in {"PROMOTE", "ROLLBACK", "NO_ACTION"}
    assert 0.0 <= outcome.champion_metrics.accuracy <= 1.0
    assert 0.0 <= outcome.challenger_metrics.accuracy <= 1.0
    assert outcome.challenger is not model
