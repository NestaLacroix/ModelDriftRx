"""
DriftRx - Streamlit Dashboard entry point.

Run:
    streamlit run dashboard/app.py
    or..
    make run-dashboard

The dashboard talks to the FastAPI service (default: http://localhost:8000).
If the API is offline every page falls back to synthetic demo data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is discoverable when running as:
# streamlit run dashboard/app.py
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from dashboard.components.gauges import inject_theme
from dashboard.pages import champion_challenger, drift_timeline, health, incidents

# Page config - must be the first Streamlit call
st.set_page_config(
    page_title="DriftRx",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject dark theme CSS tokens
inject_theme()

# Default API URL (can be overridden in the sidebar)
_API_DEFAULT = os.environ.get("DRIFTRX_API_URL", "http://localhost:8000")

# Sidebar
with st.sidebar:
    st.markdown('<p class="logo-text">DriftRx</p>', unsafe_allow_html=True)
    st.caption("ML Model Dashboard - Visual Monitor")
    st.divider()

    api_base: str = st.text_input(
        "API URL",
        value=_API_DEFAULT,
        help="URL of the running DriftRx FastAPI service",
    )

    st.divider()
    if st.button("Refresh"):
        components.html(
            """
            <script>window.parent.location.reload();</script>
            """,
            height=0,
        )
    auto_refresh = st.checkbox(
        "Auto-refresh",
        value=True,
        key="auto_refresh_checkbox",
        help="When enabled the dashboard reloads every 30 seconds.",
    )

    if auto_refresh:
        components.html(
            """
            <script>setTimeout(function(){ window.parent.location.reload(); }, 30000);</script>
            """,
            height=0,
        )

    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "Health Overview",
            "Drift Timeline",
            "Incidents",
            "Champion vs Challenger",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

# Page routing
if page == "Health Overview":
    health.render(api_base)
elif page == "Drift Timeline":
    drift_timeline.render(api_base)
elif page == "Incidents":
    incidents.render(api_base)
elif page == "Champion vs Challenger":
    champion_challenger.render(api_base)

