"""
Diagnoser - analyze drifted features and produce a diagnosis report.

Simple, testable implementation:
- Computes per-feature PSI and KS p-value (mirrors detector logic)
- Ranks features by PSI, returns top_k diagnostics
- If a model is provided and shap is installed, compute SHAP mean absolute impacts
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from scipy import stats

from src.utils.config import CONFIG


@dataclass
class FeatureDiagnosis:
    feature_name: str
    psi_score: float
    ks_p_value: float
    baseline_mean: float
    current_mean: float
    shap_mean_abs: float | None = None
    rank: int | None = None


@dataclass
class DiagnosisReport:
    timestamp_utc: float
    feature_diagnoses: list[FeatureDiagnosis]


class Diagnoser:
    def __init__(self, baseline: np.ndarray, feature_names: list[str]) -> None:
        if baseline.ndim != 2:
            raise ValueError("baseline must be a 2D numpy array (n_samples, n_features)")
        if baseline.shape[1] != len(feature_names):
            raise ValueError("feature_names length must match baseline columns")
        self.baseline = baseline
        self.feature_names = list(feature_names)

    def diagnose(
        self,
        current: np.ndarray,
        model=None,
        top_k: int = 3,
    ) -> DiagnosisReport:
        if current.ndim != 2:
            raise ValueError("current must be a 2D numpy array (n_samples, n_features)")
        if current.shape[1] != len(self.feature_names):
            raise ValueError("current columns must match feature_names length")

        diagnoses: list[FeatureDiagnosis] = []
        for i, name in enumerate(self.feature_names):
            baseline_col = self.baseline[:, i]
            current_col = current[:, i]

            psi = self._compute_psi(baseline_col, current_col)
            _, ks_p = stats.ks_2samp(baseline_col, current_col)

            fd = FeatureDiagnosis(
                feature_name=name,
                psi_score=float(psi),
                ks_p_value=float(ks_p),
                baseline_mean=float(np.nanmean(baseline_col)),
                current_mean=float(np.nanmean(current_col)),
            )
            diagnoses.append(fd)

        # Optional SHAP explanation (skip if shap not installed or fails)
        if model is not None:
            try:
                import shap  # type: ignore

                n = min(CONFIG.shap_sample_size, current.shape[0])
                sample = current[-n:, :]
                explainer = shap.Explainer(model.predict, sample)
                shap_values = explainer(sample)
                arr = np.asarray(shap_values.values)

                # Handle multiclass shapes: (n, features, classes)
                if arr.ndim == 3:
                    arr = np.mean(arr, axis=-1)

                mean_abs = np.mean(np.abs(arr), axis=0)
                for idx, fd in enumerate(diagnoses):
                    fd.shap_mean_abs = float(mean_abs[idx])
            except Exception:
                # SHAP is optional — any failure here must not break diagnosis
                pass

        # Rank by PSI descending and pick top_k
        diagnoses.sort(key=lambda x: x.psi_score, reverse=True)
        for idx, fd in enumerate(diagnoses, start=1):
            fd.rank = idx

        top_k = max(1, min(top_k, len(diagnoses)))
        top = diagnoses[:top_k]

        timestamp_utc = datetime.now(UTC).timestamp()
        return DiagnosisReport(timestamp_utc=timestamp_utc, feature_diagnoses=top)

    def _compute_psi(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if a.size == 0 or b.size == 0:
            return 0.0

        n_bins = CONFIG.psi_bins
        combined = np.concatenate([a, b])
        try:
            bins = np.histogram_bin_edges(combined, bins=n_bins)
        except Exception:
            bins = np.linspace(np.min(combined), np.max(combined), n_bins + 1)

        a_counts, _ = np.histogram(a, bins=bins)
        b_counts, _ = np.histogram(b, bins=bins)

        a_perc = np.where(a_counts == 0, 1e-8, a_counts / a_counts.sum())
        b_perc = np.where(b_counts == 0, 1e-8, b_counts / b_counts.sum())

        psi = np.sum((a_perc - b_perc) * np.log(a_perc / b_perc))
        return float(max(0.0, psi))
