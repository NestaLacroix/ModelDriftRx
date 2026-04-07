"""
Health Overview page.

Shows live service status KPIs, the latest drift snapshot as a PSI bar
chart, and a donut of healing action distribution.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import action_donut, psi_bar_chart
from dashboard.components.gauges import metric_card, page_header, status_dot
from dashboard.data import build_synthetic_drift_check, fetch_health, fetch_incidents


def render(api_base: str) -> None:
    page_header("Health Overview", "Live service status, KPIs, and latest drift snapshot.")

    health = fetch_health(api_base)
    incidents, _synthetic = fetch_incidents(api_base)

    if health.get("_synthetic"):
        st.info(
            "API not reachable - showing synthetic demo data. "
            "Start the API with `make run-api` and refresh."
        )

    # --- KPI row ---
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        model_ok = bool(health.get("model_loaded"))
        dot = status_dot(model_ok)
        label = "Loaded" if model_ok else "Not loaded"
        metric_card(
            "Model Status",
            f"{dot}{label}",
            desc="Champion model currently serving predictions.",
        )
    with c2:
        bl = bool(health.get("baseline_loaded"))
        bl_dot = status_dot(bl)
        metric_card(
            "Baseline",
            f"{bl_dot}{'Loaded' if bl else 'Not loaded'}",
            desc="Reference data used for distribution comparison.",
        )
    with c3:
        raw_ts = health.get("last_drift_check") or "Never"
        if raw_ts != "Never":
            raw_ts = raw_ts[:16].replace("T", " ")
        metric_card(
            "Last Drift Check",
            raw_ts,
            desc="Timestamp of the most recent drift scan.",
        )
    with c4:
        metric_card(
            "Total Incidents",
            str(health.get("incident_count", 0)),
            desc="Total drift events detected and auto-healed.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Drift snapshot + action donut ---
    col_left, col_right = st.columns([3, 2], gap="large")

    drift = build_synthetic_drift_check()
    fd = drift["feature_drifts"]
    names  = [f["feature_name"] for f in fd]
    scores = [f["psi_score"]    for f in fd]
    sevs   = [f["severity"]     for f in fd]

    with col_left:
        st.markdown("**Latest Drift Snapshot**")
        st.caption("PSI score per feature from the most recent drift check. Higher = more shifted from baseline.")
        st.plotly_chart(
            psi_bar_chart(names, scores, sevs),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_right:
        st.markdown("**Healing Action Distribution**")
        st.caption("How the system responded to past drift events: promoted a new model, rolled back, or took no action.")
        counts: dict[str, int] = {}
        for inc in incidents:
            action = inc.get("action", "no_action")
            counts[action] = counts.get(action, 0) + 1
        if not counts:
            counts = {"promote": 2, "rollback": 1, "no_action": 2}
        st.plotly_chart(
            action_donut(counts),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Feature detail table ---
    st.markdown("**Feature Drift Detail**")
    rows = [
        {
            "Feature": f["feature_name"],
            "PSI Score":      f"{f['psi_score']:.4f}",
            "KS p-value":     f"{f['ks_p_value']:.4f}",
            "Severity":       f["severity"].upper(),
            "Baseline Mean":  f"{f['baseline_mean']:.2f}",
            "Current Mean":   f"{f['current_mean']:.2f}",
            "Shift %":        f"{f['shift_percentage']:+.1f}%",
        }
        for f in fd
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("What do these columns mean?"):
        st.markdown(
            """
| Column | What it means |
|---|---|
| **Feature** | The input variable being monitored (e.g. `transaction_amount`). |
| **PSI Score** | Population Stability Index. Measures how much the distribution has shifted from baseline. 0-0.1 = stable, 0.1-0.25 = moderate shift, >0.25 = severe drift. |
| **KS p-value** | Kolmogorov-Smirnov test p-value. A low value (< 0.05) means the current data is statistically different from the baseline - strong evidence of drift. |
| **Severity** | Combined verdict: NONE, LOW, MODERATE, or SEVERE - based on both PSI and KS. |
| **Baseline Mean** | The average value of this feature in the original training data. |
| **Current Mean** | The average value of this feature in the incoming data being checked. |
| **Shift %** | Percentage change from baseline mean to current mean. Positive = higher values incoming, negative = lower. |
            """
        )

