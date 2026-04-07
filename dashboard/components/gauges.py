"""
DriftRx Dashboard - CSS theme injection and KPI card components.

Call ``inject_theme()`` once at the top of ``app.py`` before any other
Streamlit calls.  All styling is driven by CSS custom properties defined in
``_THEME_CSS``; change colours there and every component updates at once.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens - edit these to retheme the entire dashboard
# ---------------------------------------------------------------------------

_THEME_CSS = """
<style>
/* ===================== TOKEN DEFINITIONS ===================== */
:root {
  --bg:             #09091a;
  --bg-card:        rgba(255, 255, 255, 0.04);
  --bg-card-hover:  rgba(255, 255, 255, 0.07);
  --border:         rgba(139, 92, 246, 0.20);
  --border-hover:   rgba(139, 92, 246, 0.55);

  --purple:  #8b5cf6;
  --blue:    #60a5fa;
  --pink:    #f472b6;
  --green:   #34d399;
  --orange:  #fb923c;
  --red:     #f87171;

  --text-primary:   #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted:     #475569;

  --radius: 14px;
  --shadow: 0 4px 24px rgba(139, 92, 246, 0.10);
  --glow-purple: 0 0 20px rgba(139, 92, 246, 0.32);
  --glow-blue:   0 0 20px rgba(96,  165, 250, 0.28);
}

/* ===================== STREAMLIT BASE ===================== */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

.main .block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* Hide default Streamlit chrome */
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }

/* Hide the auto-generated multipage nav Streamlit adds from the pages/ folder */
[data-testid="stSidebarNav"] { display: none !important; }

/* ===================== SIDEBAR ===================== */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0e0e24 0%, var(--bg) 100%) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--text-secondary) !important;
}

.logo-text {
    font-size: 1.45rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--purple), var(--blue), var(--pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.2rem 0;
    display: block;
}

/* Sidebar radio - highlight selected */
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.06) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}

/* ===================== CARDS ===================== */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.25rem;
    box-shadow: var(--shadow);
    transition: border-color 0.18s, box-shadow 0.18s;
}

.card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--glow-purple);
}

.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin-bottom: 0.25rem;
    transition: border-color 0.18s, box-shadow 0.18s;
}

.kpi-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--glow-purple);
}

.kpi-left {}

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin: 0 0 0.5rem 0;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.15;
    word-break: break-word;
}

.kpi-delta-up   { font-size: 0.78rem; color: var(--green); margin-top: 0.35rem; }
.kpi-delta-down { font-size: 0.78rem; color: var(--red);   margin-top: 0.35rem; }

.kpi-desc {
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 0.45rem;
    line-height: 1.45;
}

/* ===================== BADGES ===================== */
.badge {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.badge-none      { background: rgba( 52,211,153,0.14); color: #34d399; border: 1px solid rgba( 52,211,153,0.30); }
.badge-low       { background: rgba( 96,165,250,0.14); color: #60a5fa; border: 1px solid rgba( 96,165,250,0.30); }
.badge-moderate  { background: rgba(251,146, 60,0.14); color: #fb923c; border: 1px solid rgba(251,146, 60,0.30); }
.badge-severe    { background: rgba(248,113,113,0.14); color: #f87171; border: 1px solid rgba(248,113,113,0.30); }
.badge-ok        { background: rgba( 52,211,153,0.14); color: #34d399; border: 1px solid rgba( 52,211,153,0.30); }
.badge-promote   { background: rgba(139, 92,246,0.14); color: #8b5cf6; border: 1px solid rgba(139, 92,246,0.30); }
.badge-rollback  { background: rgba(248,113,113,0.14); color: #f87171; border: 1px solid rgba(248,113,113,0.30); }
.badge-no_action { background: rgba(148,163,184,0.14); color: #94a3b8; border: 1px solid rgba(148,163,184,0.30); }

/* ===================== PAGE HEADER ===================== */
.page-header {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.15rem;
    background: linear-gradient(90deg, var(--purple) 0%, var(--blue) 60%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
}

.page-sub {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-top: 0;
    margin-bottom: 1.4rem;
}

/* ===================== STATUS DOTS ===================== */
.dot-live { display:inline-block; width:9px; height:9px; border-radius:50%;
    background:var(--green); box-shadow:0 0 7px var(--green); margin-right:7px; }
.dot-dead { display:inline-block; width:9px; height:9px; border-radius:50%;
    background:var(--red);   box-shadow:0 0 7px var(--red);   margin-right:7px; }
.dot-warn { display:inline-block; width:9px; height:9px; border-radius:50%;
    background:var(--orange); box-shadow:0 0 7px var(--orange); margin-right:7px; }

/* ===================== DATAFRAME ===================== */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ===================== PLOTLY CHART CONTAINER ===================== */
[data-testid="stPlotlyChart"] > div {
    border-radius: var(--radius);
    overflow: hidden;
}

/* ===================== EXPANDER ===================== */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
}
</style>
"""


def inject_theme() -> None:
    """Inject the global CSS theme.  Must be called once before any page renders."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    desc: str = "",
    delta: str | None = None,
    delta_positive: bool = True,
) -> None:
    """Render a KPI metric card with optional description and trend delta."""
    delta_html = ""
    if delta:
        cls = "kpi-delta-up" if delta_positive else "kpi-delta-down"
        arrow = "&#x2191;" if delta_positive else "&#x2193;"
        delta_html = f'<p class="{cls}">{arrow} {delta}</p>'

    desc_html = f'<p class="kpi-desc">{desc}</p>' if desc else ""

    # Flat HTML - no nested divs (Streamlit strips inner divs and leaves orphan closing tags)
    st.markdown(
        f'<div class="kpi-card">'
        f'<p class="kpi-label">{label}</p>'
        f'<p class="kpi-value">{value}</p>'
        f'{delta_html}{desc_html}</div>',
        unsafe_allow_html=True,
    )


def severity_badge(level: str) -> str:
    """Return an HTML severity badge string."""
    label = level.replace("_", " ").upper()
    css_class = f"badge badge-{level.lower()}"
    return f'<span class="{css_class}">{label}</span>'


def page_header(title: str, subtitle: str = "") -> None:
    """Render a gradient page title with optional subtitle."""
    sub = f'<p class="page-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<h1 class="page-header">{title}</h1>{sub}',
        unsafe_allow_html=True,
    )


def status_dot(alive: bool) -> str:
    """Return an inline HTML status dot."""
    cls = "dot-live" if alive else "dot-dead"
    return f'<span class="{cls}"></span>'

