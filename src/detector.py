"""
DriftRx - Drift Detection.

Compares incoming data against a baseline using two statistical tests
per feature to see if there is distribution drift.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from scipy import stats

from src.contracts import DriftReport, DriftSeverity, FeatureDrift
from src.utils.config import CONFIG


class DriftDetector:
    # PSI and KS tests
    def __init__(
        self,
        baseline: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        if baseline.ndim != 2:
            msg = f"baseline must be 2D, got {baseline.ndim}D"
            raise ValueError(msg)

        self.baseline = baseline
        self.n_features = baseline.shape[1]

        if feature_names is not None:
            if len(feature_names) != self.n_features:
                msg = (
                    f"feature_names length ({len(feature_names)}) "
                    f"does not match number of features ({self.n_features})"
                )
                raise ValueError(msg)
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

    def check(self, incoming: np.ndarray) -> DriftReport:
        # Compare incoming data against the baseline. (PSI and KS tests per feature)
        if incoming.ndim != 2:
            msg = f"incoming must be 2D, got {incoming.ndim}D"
            raise ValueError(msg)
        if incoming.shape[1] != self.n_features:
            msg = (
                f"incoming has {incoming.shape[1]} features, "
                f"expected {self.n_features}"
            )
            raise ValueError(msg)

        feature_drifts: list[FeatureDrift] = []
        worst_severity = DriftSeverity.NONE

        for i in range(self.n_features):
            baseline_col = self.baseline[:, i]
            incoming_col = incoming[:, i]

            psi = self._compute_psi(baseline_col, incoming_col)
            ks_stat, ks_p = stats.ks_2samp(baseline_col, incoming_col)
            severity = self._classify_severity(psi, ks_p)

            if severity > worst_severity:
                worst_severity = severity

            fd = FeatureDrift(
                feature_name=self.feature_names[i],
                psi_score=round(psi, 6),
                ks_p_value=round(ks_p, 6),
                severity=severity,
                baseline_mean=round(float(np.mean(baseline_col)), 4),
                current_mean=round(float(np.mean(incoming_col)), 4),
            )
            feature_drifts.append(fd)

        triggered = worst_severity >= CONFIG.healing_trigger_severity

        return DriftReport(
            timestamp=datetime.now(tz=timezone.utc),
            overall_severity=worst_severity,
            feature_drifts=feature_drifts,
            triggered_healing=triggered,
        )

    def _compute_psi(
        self,
        baseline_col: np.ndarray,
        incoming_col: np.ndarray,
    ) -> float:

        n_bins = CONFIG.psi_bins

        # Bin edges from the baseline distribution
        min_val = float(np.min(baseline_col))
        max_val = float(np.max(baseline_col))
        edges = np.linspace(min_val, max_val, n_bins + 1)

        # Widen edges slightly so all values fall inside
        edges[0] = edges[0] - 1e-6
        edges[-1] = edges[-1] + 1e-6

        baseline_counts = np.histogram(baseline_col, bins=edges)[0]
        incoming_counts = np.histogram(incoming_col, bins=edges)[0]

        # Convert to proportions with a small constant to avoid zero
        eps = 1e-8
        baseline_pct = (baseline_counts / len(baseline_col)) + eps
        incoming_pct = (incoming_counts / len(incoming_col)) + eps

        psi = float(np.sum(
            (incoming_pct - baseline_pct) * np.log(incoming_pct / baseline_pct),
        ))

        return max(psi, 0.0)

    def _classify_severity(self, psi: float, ks_p: float) -> DriftSeverity:
        # Assinging severity based on PSI thresholds, then bumping up if KS is significant - same nums from config
        low, mod, sev = CONFIG.psi_thresholds

        if psi >= sev:
            severity = DriftSeverity.SEVERE
        elif psi >= mod:
            severity = DriftSeverity.MODERATE
        elif psi >= low:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        # KS can bump severity up one level
        if ks_p < CONFIG.ks_significance:
            if severity == DriftSeverity.NONE:
                severity = DriftSeverity.LOW
            elif severity == DriftSeverity.LOW:
                severity = DriftSeverity.MODERATE
            elif severity == DriftSeverity.MODERATE:
                severity = DriftSeverity.SEVERE

        return severity