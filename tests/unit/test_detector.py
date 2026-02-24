"""
DriftRx - Tests for drift detection.

Uses the sample baseline and drifted data from conftest to verify
that the detector correctly identifies shifted features.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import DriftSeverity
from src.detector import DriftDetector


class TestDriftDetectorInit:
    """Verify detector setup and input validation."""

    def test_creates_with_baseline(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        assert detector.n_features == 10

    def test_auto_generates_feature_names(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        assert detector.feature_names[0] == "feature_0"
        assert detector.feature_names[9] == "feature_9"

    def test_accepts_custom_feature_names(
        self,
        sample_baseline_data: np.ndarray,
        sample_feature_names: list[str],
    ) -> None:
        detector = DriftDetector(sample_baseline_data, sample_feature_names)
        assert detector.feature_names == sample_feature_names

    def test_rejects_wrong_name_count(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="feature_names length"):
            DriftDetector(sample_baseline_data, ["a", "b"])

    def test_rejects_1d_baseline(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            DriftDetector(np.ones(10))

    def test_rejects_3d_baseline(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            DriftDetector(np.ones((10, 5, 3)))


class TestDriftDetectorCheck:
    """Verify drift detection results."""

    def test_rejects_wrong_feature_count(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        wrong_shape = np.ones((100, 5))
        with pytest.raises(ValueError, match="features"):
            detector.check(wrong_shape)

    def test_rejects_1d_incoming(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        with pytest.raises(ValueError, match="2D"):
            detector.check(np.ones(10))

    def test_no_drift_on_same_data(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        report = detector.check(sample_baseline_data)
        assert report.overall_severity == DriftSeverity.NONE
        assert report.triggered_healing is False
        assert len(report.drifted_features) == 0

    def test_detects_severe_drift(
        self,
        sample_baseline_data: np.ndarray,
        sample_drifted_data: np.ndarray,
        sample_feature_names: list[str],
    ) -> None:
        detector = DriftDetector(sample_baseline_data, sample_feature_names)
        report = detector.check(sample_drifted_data)

        assert report.overall_severity >= DriftSeverity.MODERATE

        # feature_0 had a big shift (50 -> 70), should be caught
        drifted_names = [fd.feature_name for fd in report.drifted_features]
        assert "feature_0" in drifted_names

    def test_feature_0_is_worst(
        self,
        sample_baseline_data: np.ndarray,
        sample_drifted_data: np.ndarray,
        sample_feature_names: list[str],
    ) -> None:
        detector = DriftDetector(sample_baseline_data, sample_feature_names)
        report = detector.check(sample_drifted_data)

        top = report.top_drifted_features(1)
        assert len(top) == 1
        assert top[0].feature_name == "feature_0"

    def test_undrifted_features_are_none(
        self,
        sample_baseline_data: np.ndarray,
        sample_drifted_data: np.ndarray,
        sample_feature_names: list[str],
    ) -> None:
        detector = DriftDetector(sample_baseline_data, sample_feature_names)
        report = detector.check(sample_drifted_data)

        # features 2-9 should have no drift
        for fd in report.feature_drifts:
            if fd.feature_name not in ("feature_0", "feature_1"):
                assert fd.severity <= DriftSeverity.LOW, (
                    f"{fd.feature_name} should be NONE but got {fd.severity}"
                )

    def test_triggers_healing_on_severe(
        self,
        sample_baseline_data: np.ndarray,
        sample_drifted_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        report = detector.check(sample_drifted_data)
        assert report.triggered_healing is True

    def test_report_has_correct_feature_count(
        self,
        sample_baseline_data: np.ndarray,
        sample_drifted_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        report = detector.check(sample_drifted_data)
        assert len(report.feature_drifts) == 10

    def test_report_has_timestamp(
        self,
        sample_baseline_data: np.ndarray,
        sample_drifted_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        report = detector.check(sample_drifted_data)
        assert report.timestamp is not None

    def test_to_dict_works(
        self,
        sample_baseline_data: np.ndarray,
        sample_drifted_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        report = detector.check(sample_drifted_data)
        d = report.to_dict()
        assert "overall_severity" in d
        assert "feature_drifts" in d
        assert len(d["feature_drifts"]) == 10


class TestPSI:
    """Verify PSI calculation directly."""

    def test_identical_distributions_low_psi(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        col = sample_baseline_data[:, 0]
        psi = detector._compute_psi(col, col)
        assert psi < 0.1

    def test_shifted_distribution_high_psi(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        baseline_col = sample_baseline_data[:, 0]
        shifted_col = baseline_col + 20  # shift mean by 20
        psi = detector._compute_psi(baseline_col, shifted_col)
        assert psi > 0.3

    def test_psi_is_non_negative(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        rng = np.random.default_rng(99)
        col_a = rng.normal(50, 10, size=500)
        col_b = rng.normal(50, 10, size=500)
        psi = detector._compute_psi(col_a, col_b)
        assert psi >= 0.0


class TestSeverityClassification:
    """Verify severity assignment logic."""

    def test_none_severity(self, sample_baseline_data: np.ndarray) -> None:
        detector = DriftDetector(sample_baseline_data)
        severity = detector._classify_severity(psi=0.05, ks_p=0.5)
        assert severity == DriftSeverity.NONE

    def test_low_severity(self, sample_baseline_data: np.ndarray) -> None:
        detector = DriftDetector(sample_baseline_data)
        severity = detector._classify_severity(psi=0.15, ks_p=0.5)
        assert severity == DriftSeverity.LOW

    def test_moderate_severity(self, sample_baseline_data: np.ndarray) -> None:
        detector = DriftDetector(sample_baseline_data)
        severity = detector._classify_severity(psi=0.25, ks_p=0.5)
        assert severity == DriftSeverity.MODERATE

    def test_severe_severity(self, sample_baseline_data: np.ndarray) -> None:
        detector = DriftDetector(sample_baseline_data)
        severity = detector._classify_severity(psi=0.35, ks_p=0.5)
        assert severity == DriftSeverity.SEVERE

    def test_ks_bumps_none_to_low(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        severity = detector._classify_severity(psi=0.05, ks_p=0.01)
        assert severity == DriftSeverity.LOW

    def test_ks_bumps_moderate_to_severe(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        severity = detector._classify_severity(psi=0.25, ks_p=0.01)
        assert severity == DriftSeverity.SEVERE

    def test_ks_does_not_bump_severe(
        self, sample_baseline_data: np.ndarray,
    ) -> None:
        detector = DriftDetector(sample_baseline_data)
        severity = detector._classify_severity(psi=0.35, ks_p=0.01)
        assert severity == DriftSeverity.SEVERE
