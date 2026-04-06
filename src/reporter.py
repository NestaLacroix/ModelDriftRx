"""
Reporter - generate incident reports and visualization charts.

Takes a HealingOutcome (from contracts.py) and produces an IncidentReport.
Charts are saved as PNG files to the configured reports directory.

Charts produced:
  drift_bar            - PSI score per feature, color-coded by severity
  dist_compare         - baseline vs. current histogram for top drifted features
                         (only when raw arrays are supplied)
  champion_vs_challenger - side-by-side metric bar chart for both models
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

# Non-interactive backend - safe for servers and test runners (no display needed)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.contracts import (
    DriftSeverity,
    HealAction,
    HealingOutcome,
    IncidentReport,
)
from src.utils.config import CONFIG

# Severity → bar color mapping
_SEVERITY_COLORS: dict[DriftSeverity, str] = {
    DriftSeverity.NONE: "#4caf50",      # green
    DriftSeverity.LOW: "#ff9800",       # amber
    DriftSeverity.MODERATE: "#f44336",  # red
    DriftSeverity.SEVERE: "#9c27b0",    # purple
}


class Reporter:
    """
    Generates an IncidentReport from a HealingOutcome.

    Usage:
        reporter = Reporter()                              # uses CONFIG.reports_dir
        reporter = Reporter(reports_dir="custom/path")    # override path

        report = reporter.generate(outcome)
        report = reporter.generate(outcome, baseline=X_base, current=X_curr,
                                   feature_names=names)
    """

    def __init__(self, reports_dir: str | None = None) -> None:
        self._reports_dir = Path(reports_dir if reports_dir is not None else CONFIG.reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- public

    def generate(
        self,
        outcome: HealingOutcome,
        baseline: np.ndarray | None = None,
        current: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> IncidentReport:
        """
        Build a complete IncidentReport.

        Parameters
        ----------
        outcome:        The HealingOutcome produced by the Healer.
        baseline:       Optional 2-D array (n_samples, n_features).
                        Required to generate the distribution-comparison chart.
        current:        Optional 2-D array (n_samples, n_features).
                        Required to generate the distribution-comparison chart.
        feature_names:  Column names for baseline/current.  When omitted,
                        feature_0, feature_1, … are assumed.
        """
        summary = self._build_summary(outcome)
        charts = self._generate_charts(outcome, baseline, current, feature_names)
        return IncidentReport(
            healing_outcome=outcome,
            timestamp=datetime.now(tz=UTC),
            summary=summary,
            charts=charts,
        )

    # -------------------------------------------------------------------- private

    def _build_summary(self, outcome: HealingOutcome) -> str:
        """Compose a plain-English paragraph describing the incident."""
        diagnosis = outcome.diagnosis
        drift_report = diagnosis.drift_report
        drifted = drift_report.drifted_features
        lines: list[str] = []

        # 1. Drift overview
        if not drifted:
            lines.append(
                "No meaningful drift was detected. Model performance is stable."
            )
        else:
            worst = max(drifted, key=lambda fd: fd.psi_score)
            lines.append(
                f"Drift detected: overall severity is {drift_report.overall_severity.value}. "
                f"{len(drifted)} feature(s) shifted. "
                f"Worst offender: {worst.feature_name} "
                f"(PSI={worst.psi_score:.4f}, shift={worst.shift_percentage:+.1f}%)."
            )

        # 2. Root cause from diagnosis
        if diagnosis.root_cause_summary:
            lines.append(f"Root cause: {diagnosis.root_cause_summary}.")

        if diagnosis.estimated_accuracy_drop > 0:
            lines.append(
                f"Estimated accuracy drop: "
                f"{diagnosis.estimated_accuracy_drop * 100:.1f}%."
            )

        # 3. Healing decision
        action_label = (
            outcome.action.value
            if isinstance(outcome.action, HealAction)
            else str(outcome.action)
        )
        lines.append(
            f"Healing decision: {action_label.upper()}. "
            f"Champion accuracy {outcome.champion_metrics.accuracy:.4f}, "
            f"challenger accuracy {outcome.challenger_metrics.accuracy:.4f}. "
            f"Reason: {outcome.reason}"
        )

        return " ".join(lines)

    # ---- chart generation ----------------------------------------------------

    def _chart_path(self, prefix: str, incident_id: str) -> Path:
        return self._reports_dir / f"{prefix}_{incident_id}.png"

    def _generate_charts(
        self,
        outcome: HealingOutcome,
        baseline: np.ndarray | None,
        current: np.ndarray | None,
        feature_names: list[str] | None,
    ) -> dict[str, str]:
        """Generate all charts and return a {name: path_str} dict."""
        # Unique suffix so parallel runs don't overwrite each other
        incident_id = str(uuid.uuid4())[:8]
        charts: dict[str, str] = {}

        path = self._drift_bar_chart(outcome, incident_id)
        if path:
            charts["drift_bar"] = str(path)

        if baseline is not None and current is not None:
            path = self._dist_compare_chart(
                outcome, baseline, current, feature_names, incident_id
            )
            if path:
                charts["dist_compare"] = str(path)

        path = self._champ_vs_chall_chart(outcome, incident_id)
        if path:
            charts["champion_vs_challenger"] = str(path)

        plt.close("all")
        return charts

    def _drift_bar_chart(
        self,
        outcome: HealingOutcome,
        incident_id: str,
    ) -> Path | None:
        """Bar chart of PSI scores per feature, colored by drift severity."""
        feature_drifts = outcome.diagnosis.drift_report.feature_drifts
        if not feature_drifts:
            return None

        names = [fd.feature_name for fd in feature_drifts]
        psi_scores = [fd.psi_score for fd in feature_drifts]
        colors = [_SEVERITY_COLORS.get(fd.severity, "#9e9e9e") for fd in feature_drifts]

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 5))
        ax.bar(names, psi_scores, color=colors)
        ax.set_title("PSI Drift Score per Feature")
        ax.set_xlabel("Feature")
        ax.set_ylabel("PSI Score")
        ax.tick_params(axis="x", rotation=45)

        # Threshold reference lines
        ax.axhline(
            CONFIG.psi_none, color="grey", linestyle="--", linewidth=0.8,
            alpha=0.7, label=f"Low threshold ({CONFIG.psi_none})"
        )
        ax.axhline(
            CONFIG.psi_low, color="orange", linestyle="--", linewidth=0.8,
            alpha=0.7, label=f"Moderate threshold ({CONFIG.psi_low})"
        )
        ax.axhline(
            CONFIG.psi_moderate, color="red", linestyle="--", linewidth=0.8,
            alpha=0.7, label=f"Severe threshold ({CONFIG.psi_moderate})"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()

        path = self._chart_path("drift_bar", incident_id)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def _dist_compare_chart(
        self,
        outcome: HealingOutcome,
        baseline: np.ndarray,
        current: np.ndarray,
        feature_names: list[str] | None,
        incident_id: str,
    ) -> Path | None:
        """Histogram of baseline vs. current for the top drifted features."""
        drifted = outcome.diagnosis.top_contributors
        if not drifted:
            return None

        n_features = baseline.shape[1]
        fnames = feature_names or [f"feature_{i}" for i in range(n_features)]
        name_to_idx = {name: i for i, name in enumerate(fnames)}

        # Only plot features we can find in the arrays
        plottable = [fd for fd in drifted if fd.feature_name in name_to_idx]
        if not plottable:
            return None

        n = len(plottable)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

        for col, fd in enumerate(plottable):
            ax = axes[0][col]
            idx = name_to_idx[fd.feature_name]
            ax.hist(baseline[:, idx], bins=30, alpha=0.55, label="Baseline", color="#2196f3")
            ax.hist(current[:, idx], bins=30, alpha=0.55, label="Current", color="#f44336")
            ax.set_title(f"{fd.feature_name}\nPSI={fd.psi_score:.3f}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

        fig.suptitle("Baseline vs. Current Distribution")
        fig.tight_layout()

        path = self._chart_path("dist_compare", incident_id)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def _champ_vs_chall_chart(
        self,
        outcome: HealingOutcome,
        incident_id: str,
    ) -> Path | None:
        """Side-by-side bar chart comparing champion vs. challenger metrics."""
        champ = outcome.champion_metrics
        chall = outcome.challenger_metrics

        metric_labels = ["accuracy", "f1", "precision", "recall"]
        champ_vals = [champ.accuracy, champ.f1, champ.precision, champ.recall]
        chall_vals = [chall.accuracy, chall.f1, chall.precision, chall.recall]

        x = list(range(len(metric_labels)))
        width = 0.35

        action_label = (
            outcome.action.value
            if isinstance(outcome.action, HealAction)
            else str(outcome.action)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            [i - width / 2 for i in x], champ_vals, width,
            label="Champion", color="#2196f3"
        )
        ax.bar(
            [i + width / 2 for i in x], chall_vals, width,
            label="Challenger", color="#ff9800"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(f"Champion vs. Challenger - Decision: {action_label.upper()}")
        ax.legend()
        fig.tight_layout()

        path = self._chart_path("champ_vs_chall", incident_id)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path
