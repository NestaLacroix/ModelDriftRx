"""
Drift Timeline page.

Historical PSI trend chart per feature plus a detail breakdown of the most
recent drift check.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import drift_timeline_chart, psi_bar_chart
from dashboard.components.gauges import metric_card, page_header
from dashboard.data import build_drift_timeline, fetch_latest_drift_check


def render(api_base: str) -> None:
    page_header("Drift Timeline", "Historical PSI trend per feature across recent checks.")

    # --- Timeline area chart ---
    st.caption("Each line shows the PSI drift score for one feature over time. A rising line means that feature's distribution is shifting away from baseline.")
    df = build_drift_timeline(api_base)
    if len(df) > 0:
        st.plotly_chart(
            drift_timeline_chart(df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("No drift checks have been recorded yet.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Latest check KPIs ---
    drift = fetch_latest_drift_check(api_base)
    if drift is None:
        st.info("Run POST /check-drift to populate this view.")
        return
    fd = drift["feature_drifts"]

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        metric_card(
            "Overall Severity",
            drift["overall_severity"].upper(),
            desc="Worst severity level seen across all features in the last check.",
        )
    with c2:
        metric_card(
            "Features Checked",
            str(len(fd)),
            desc="Number of input features analysed for distribution shift.",
        )
    with c3:
        metric_card(
            "Healing Status",
            drift.get("healing_status", "triggered" if drift["triggered_healing"] else "not_triggered").replace("_", " ").title(),
            desc="Whether the drift threshold was crossed and whether a labeled healing cycle ran.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PSI bar + feature table ---
    st.markdown("**Feature PSI Scores - Last Check**")
    st.caption("Bars show each feature's PSI score. The dashed lines mark the Low (0.10) and High (0.25) thresholds.")
    names  = [f["feature_name"] for f in fd]
    scores = [f["psi_score"]    for f in fd]
    sevs   = [f["severity"]     for f in fd]
    st.plotly_chart(
        psi_bar_chart(names, scores, sevs),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    rows = [
        {
            "Feature":    f["feature_name"],
            "PSI":        f"{f['psi_score']:.4f}",
            "KS p-value": f"{f['ks_p_value']:.4f}",
            "Severity":   f["severity"].upper(),
            "Shift %":    f"{f['shift_percentage']:+.1f}%",
        }
        for f in fd
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("What do these columns mean?"):
        st.markdown(
            """
| Column | What it means |
|--------|---------------|
| **Feature** | The input variable being analysed for distribution shift. |
| **PSI** | Population Stability Index - measures how much the feature's distribution has shifted. Above 0.10 = low drift, above 0.25 = severe. |
| **KS p-value** | Kolmogorov-Smirnov test p-value. A low value (< 0.05) means the current and baseline distributions are significantly different. |
| **Severity** | Derived from PSI: NONE (< 0.10), LOW (0.10-0.25), MODERATE, or SEVERE (> 0.25). |
| **Shift %** | Percentage change in the feature's mean value from baseline. Positive = increased, negative = decreased. |
"""
        )

