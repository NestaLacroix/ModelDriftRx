"""
DriftRx Dashboard - Reusable Plotly chart components.

Every function returns a ``plotly.graph_objects.Figure`` ready for
``st.plotly_chart(..., use_container_width=True)``.
Colours mirror the CSS tokens declared in ``gauges.py``.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Colour palette (mirrors CSS tokens)
# ---------------------------------------------------------------------------

PURPLE = "#8b5cf6"
BLUE   = "#60a5fa"
PINK   = "#f472b6"
GREEN  = "#34d399"
ORANGE = "#fb923c"
RED    = "#f87171"

_SEVERITY_COLORS = {
    "none":     GREEN,
    "low":      BLUE,
    "moderate": ORANGE,
    "severe":   RED,
}

_ACTION_COLORS = {
    "promote":   PURPLE,
    "rollback":  RED,
    "no_action": "#94a3b8",
}

# Semi-transparent fill versions for area charts
_FILL_ALPHA: dict[str, str] = {
    PURPLE: "rgba(139, 92,246,0.08)",
    BLUE:   "rgba( 96,165,250,0.08)",
    PINK:   "rgba(244,114,182,0.08)",
    GREEN:  "rgba( 52,211,153,0.08)",
    ORANGE: "rgba(251,146, 60,0.08)",
    RED:    "rgba(248,113,113,0.08)",
}

_PALETTE = [PURPLE, BLUE, PINK, GREEN, ORANGE, RED]

# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------

_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(139,92,246,0.22)",
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1a1a30",
        bordercolor="rgba(139,92,246,0.5)",
        font=dict(color="#f1f5f9", size=12),
    ),
)


def _base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#64748b"), x=0),
        **_BASE,
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        showgrid=True,
        zeroline=False,
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        showgrid=True,
        zeroline=False,
        tickfont=dict(size=11),
    )
    return fig


# ---------------------------------------------------------------------------
# PSI bar chart
# ---------------------------------------------------------------------------


def psi_bar_chart(
    feature_names: list[str],
    psi_scores: list[float],
    severities: list[str],
) -> go.Figure:
    """Horizontal bar chart of PSI scores, colour-coded by severity."""
    if not feature_names:
        return go.Figure()

    colors = [_SEVERITY_COLORS.get(s, PURPLE) for s in severities]
    paired = sorted(zip(psi_scores, feature_names, colors), reverse=True)
    scores_s, names_s, colors_s = zip(*paired)

    fig = go.Figure(
        go.Bar(
            x=list(scores_s),
            y=list(names_s),
            orientation="h",
            marker=dict(
                color=list(colors_s),
                opacity=0.82,
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" for v in scores_s],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
            hovertemplate="<b>%{y}</b><br>PSI: %{x:.4f}<extra></extra>",
        )
    )
    _base(fig, "PSI Score per Feature")
    fig.add_vline(
        x=0.10,
        line_dash="dot",
        line_color=ORANGE,
        opacity=0.55,
        annotation_text="Low (0.10)",
        annotation_font_color=ORANGE,
        annotation_font_size=10,
        annotation_position="top right",
    )
    fig.add_vline(
        x=0.25,
        line_dash="dot",
        line_color=RED,
        opacity=0.55,
        annotation_text="High (0.25)",
        annotation_font_color=RED,
        annotation_font_size=10,
        annotation_position="top right",
    )
    fig.update_layout(height=max(220, len(feature_names) * 46 + 60))
    return fig


# ---------------------------------------------------------------------------
# Drift timeline
# ---------------------------------------------------------------------------


def drift_timeline_chart(
    df: pd.DataFrame,
    title: str = "PSI Over Time",
) -> go.Figure:
    """
    Multi-line area chart.  ``df`` must have a ``timestamp`` column and one
    numeric column per feature.
    """
    feature_cols = [c for c in df.columns if c != "timestamp"]
    fig = go.Figure()
    for i, col in enumerate(feature_cols):
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[col],
                name=col,
                mode="lines+markers",
                line=dict(color=color, width=2.2),
                marker=dict(size=5, color=color, symbol="circle"),
                fill="tozeroy",
                fillcolor=_FILL_ALPHA.get(color, "rgba(139,92,246,0.08)"),
                hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.4f}}<extra></extra>",
            )
        )
    _base(fig, title)
    fig.update_layout(height=320, hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# Champion vs challenger
# ---------------------------------------------------------------------------


def champion_challenger_chart(
    metric_names: list[str],
    champion_values: list[float],
    challenger_values: list[float],
) -> go.Figure:
    """Grouped bar chart comparing champion and challenger metrics."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Champion",
            x=metric_names,
            y=champion_values,
            marker=dict(
                color=BLUE,
                opacity=0.82,
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" for v in champion_values],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
            hovertemplate="Champion %{x}: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Challenger",
            x=metric_names,
            y=challenger_values,
            marker=dict(
                color=PURPLE,
                opacity=0.82,
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" for v in challenger_values],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
            hovertemplate="Challenger %{x}: %{y:.4f}<extra></extra>",
        )
    )
    _base(fig, "Champion vs Challenger")
    fig.update_layout(barmode="group", height=320)
    return fig


# ---------------------------------------------------------------------------
# Action distribution donut
# ---------------------------------------------------------------------------


def action_donut(counts: dict[str, int]) -> go.Figure:
    """Donut chart showing the distribution of healing actions."""
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [_ACTION_COLORS.get(l, BLUE) for l in labels]

    fig = go.Figure(
        go.Pie(
            labels=[l.replace("_", " ").upper() for l in labels],
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="#09091a", width=2)),
            textfont=dict(size=11, color="#f1f5f9"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )
    _base(fig, "Healing Action Distribution")
    fig.update_layout(height=260, showlegend=True)
    return fig

