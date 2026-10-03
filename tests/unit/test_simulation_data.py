from __future__ import annotations

import numpy as np
import pytest

from simulation.generate_data import generate_simulation_data


def test_simulation_data_has_matching_shapes_and_names():
    data = generate_simulation_data(n_samples=100, seed=7)

    assert data.baseline_features.shape == data.incoming_features.shape == (100, 6)
    assert data.baseline_labels.shape == data.incoming_labels.shape == (100,)
    assert len(data.feature_names) == 6


def test_simulation_data_is_reproducible():
    first = generate_simulation_data(n_samples=100, seed=12)
    second = generate_simulation_data(n_samples=100, seed=12)

    assert np.array_equal(first.baseline_features, second.baseline_features)
    assert np.array_equal(first.incoming_features, second.incoming_features)
    assert np.array_equal(first.incoming_labels, second.incoming_labels)


def test_simulation_data_incoming_distribution_is_shifted():
    data = generate_simulation_data(n_samples=500, drift_amount=1.0, seed=21)

    assert data.incoming_features[:, 0].mean() > data.baseline_features[:, 0].mean()
    assert data.incoming_features[:, 4].mean() > data.baseline_features[:, 4].mean()


def test_simulation_data_validates_arguments():
    with pytest.raises(ValueError, match="n_samples"):
        generate_simulation_data(n_samples=10)
    with pytest.raises(ValueError, match="drift_amount"):
        generate_simulation_data(drift_amount=0)
