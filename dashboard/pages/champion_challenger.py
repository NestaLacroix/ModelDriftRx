"""
Champion vs Challenger page.

Side-by-side metric comparison for the most recent retraining outcome.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import champion_challenger_chart
from dashboard.components.gauges import metric_card, page_header, severity_badge
from dashboard.data import build_synthetic_champ_vs_chall


def render(api_base: str) -> None:
    page_header(
        "Champion vs Challenger",
        "Latest retraining outcome and metric comparison.",
    )

    data = build_synthetic_champ_vs_chall()

    # --- Outcome summary ---
    c1, c2 = st.columns(2, gap="small")
    with c1:
        badge = severity_badge(data["action"])
        action_desc = {
            "promote": "Challenger beat champion by the required margin - it has been promoted to production.",
            "rollback": "Challenger did not improve enough - keeping the current champion.",
            "no_action": "Drift was detected but no retraining was triggered (below severity threshold).",
        }.get(data["action"], "")
        metric_card(
            "Decision",
            data["action"].replace("_", " ").upper(),
            desc=action_desc,
        )
    with c2:
        metric_card(
            "Reason",
            data["reason"],
            desc="The exact rule that determined which model wins. Configured via `min_improvement` in config.py.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Grouped bar chart ---
    st.caption("Blue bars = current production model (champion). Purple bars = newly retrained candidate (challenger).")
    st.plotly_chart(
        champion_challenger_chart(
            data["metric_names"],
            data["champion"],
            data["challenger"],
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Metric table ---
    st.markdown("**Metric Detail**")
    st.caption("Delta = Challenger minus Champion. Positive delta means the challenger improved on that metric.")
    rows = []
    for name, champ, chall in zip(
        data["metric_names"], data["champion"], data["challenger"]
    ):
        delta = chall - champ
        rows.append(
            {
                "Metric":     name,
                "Champion":   f"{champ:.4f}",
                "Challenger": f"{chall:.4f}",
                "Delta":      f"{delta:+.4f}",
                "Winner":     "Challenger" if delta > 0 else "Champion",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

