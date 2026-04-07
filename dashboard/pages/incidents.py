"""
Incidents page.

Full history of detected drift events and healing decisions.  Each entry is
expandable to show the full summary; if the API is online, the full
``healing_outcome`` dict is also shown.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.gauges import metric_card, page_header, severity_badge
from dashboard.data import fetch_incident_detail, fetch_incidents


def render(api_base: str) -> None:
    page_header("Incidents", "History of detected drift events and self-healing decisions.")

    incidents, is_synthetic = fetch_incidents(api_base)

    if is_synthetic:
        st.info(
            "API not reachable - showing synthetic demo data. "
            "Start the API with `make run-api` and refresh."
        )

    if not incidents:
        st.markdown(
            '<div class="card"><p style="color:var(--text-secondary)">'
            "No incidents recorded yet.</p></div>",
            unsafe_allow_html=True,
        )
        return

    # --- Summary KPIs ---
    actions = [inc.get("action", "no_action") for inc in incidents]
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        metric_card("Total Incidents", str(len(incidents)), desc="Number of drift events detected in total.")
    with c2:
        metric_card("Promotions", str(actions.count("promote")), desc="Events where the challenger model was promoted to production.")
    with c3:
        metric_card("Rollbacks", str(actions.count("rollback")), desc="Events where the system reverted to a previous stable model.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Incident list ---
    for inc in incidents:
        action = inc.get("action", "no_action")
        badge  = severity_badge(action)
        raw_ts = (inc.get("timestamp", "") or "")[:16].replace("T", " ")
        inc_id = inc.get("id", "")

        label = f"{raw_ts}  |  {action.replace('_', ' ').upper()}  |  {inc_id}"
        with st.expander(label):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.markdown("**Action**")
                st.markdown(badge, unsafe_allow_html=True)
            with col_b:
                st.markdown("**Summary**")
                st.markdown(inc.get("summary", "No summary available."))

            # Full detail from API when online
            if not is_synthetic:
                detail = fetch_incident_detail(api_base, inc_id)
                if detail and detail.get("healing_outcome"):
                    st.markdown("**Healing Outcome**")
                    st.json(detail["healing_outcome"])

