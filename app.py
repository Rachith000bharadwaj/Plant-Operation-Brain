# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
app.py -- the whole frontend, in pure Python via Streamlit.

Run it with:   streamlit run app.py

Tabs:
  - Ask        : the Expert Knowledge Copilot (cited, honest answers)
  - Knowledge Graph : interactive view of equipment <-> procedures <-> regulations
"""

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from src.ingest import ingest_folder
from src.embed_store import VectorStore, Reranker
from src.graph_neo4j import make_graph
from src.rag import PlantBrain
from src.maintenance import analyze_equipment
from src.compliance import audit
from src.lessons import mine_lessons
from src.conflict import find_conflicts
from src.interview import next_question, save_knowledge, knowledge_risk_map
from src.whatif import check_plan
from src.sensors import get_registry, series, status_at, explain_alert
from src.handover import generate_brief
from src.impact import analyze_impact, subgraph_html
from src.deep import deep_ask
from src.speech import tts_wav
from src.costs import fmt_inr, downtime_cost, prevention_value
from src.health import plant_health
from src.sop_fix import draft_fix, approve
from src.orchestrator import dispatch, route, AGENTS
from src.llm import provider_status, vision, transcribe

DOCS_DIR = "data/docs"
INDEX_DIR = "data/index"

# Chart series colors — mid-tone hues chosen to hold contrast on BOTH the light
# ivory and dark charcoal surfaces (charts always render on a light card).
# Fixed order; never cycled.
CHART_COLORS = ["#0891b2", "#8b5cf6", "#d97706", "#059669", "#ec4899", "#3b82f6"]


def line_chart(series_map: dict, x_title: str = "hours", height: int = 260):
    """Multi-series line chart with a 0-based y-axis (never dips negative) and a
    fixed height so the box keeps a clean, consistent shape."""
    import altair as alt, pandas as pd
    rows = [{"x": i, "value": v, "series": name}
            for name, vals in series_map.items() for i, v in enumerate(vals)]
    if not rows:
        return
    df = pd.DataFrame(rows)
    ch = (alt.Chart(df).mark_line(strokeWidth=2).encode(
            x=alt.X("x:Q", title=x_title),
            y=alt.Y("value:Q", title=None,
                    scale=alt.Scale(domainMin=0, nice=True, clamp=True)),
            color=alt.Color("series:N",
                            scale=alt.Scale(range=CHART_COLORS),
                            legend=alt.Legend(orient="bottom", title=None)))
          .properties(height=height)
          .configure_view(strokeWidth=0)
          .configure_axis(grid=True, gridOpacity=0.15))
    st.altair_chart(ch, use_container_width=True)


def bar_chart(counts: dict, x_title: str = "", height: int = 260):
    """Horizontal bar chart, 0-based, fixed height, sorted by value."""
    import altair as alt, pandas as pd
    if not counts:
        return
    df = pd.DataFrame({"category": list(counts), "count": list(counts.values())})
    ch = (alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("count:Q", title=x_title, scale=alt.Scale(domainMin=0)),
            y=alt.Y("category:N", sort="-x", title=None),
            color=alt.value(CHART_COLORS[0]))
          .properties(height=height)
          .configure_view(strokeWidth=0)
          .configure_axis(grid=True, gridOpacity=0.15))
    st.altair_chart(ch, use_container_width=True)

st.set_page_config(page_title="Plant Operations Brain", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

def inject_premium_ui(dark: bool = False):
    # Claude light (warm ivory) and Claude dark (warm charcoal) palettes.
    if dark:
        palette = """
        --bg: #262624;            /* Claude dark: warm charcoal */
        --panel: #1F1E1D;
        --card: #30302E;
        --border: #3D3C39;
        --border2: #4A4945;
        --hover: #3A3937;
        --text: #ECEBE8;
        --muted: #A8A49C;
        --accent: #D97757;        /* same coral in both modes */
        --accent-hover: #E08A6D;
        """
    else:
        palette = """
        --bg: #FAF9F5;            /* warm ivory */
        --panel: #F0EEE6;         /* warm panel */
        --card: #FFFFFF;
        --border: #E8E5DC;
        --border2: #D6D2C4;
        --hover: #E9E6DC;
        --text: #1F1E1D;
        --muted: #6B6A66;
        --accent: #D97757;        /* Claude coral */
        --accent-hover: #C4633F;
        """
    # Charts render with the native light theme; in dark mode keep them on a
    # light card so axis labels stay readable (like documents on a dark desk).
    chart_card = """
    [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
        background: #FAF9F5 !important; border-radius: 12px !important;
        padding: 8px !important;
    }
    """ if dark else ""

    st.markdown(
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family="
        "Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');"
        ":root {" + palette + "}" + chart_card + """

    /* Header: clean, light, keep sidebar toggle */
    header[data-testid="stHeader"] { background: var(--bg) !important; }
    header[data-testid="stHeader"] button { color: var(--text) !important; }
    /* Hide Streamlit's own Deploy button + menu so our theme toggle owns the
       top-right corner (covers old + new Streamlit test-ids). */
    .stDeployButton, [data-testid="stAppDeployButton"],
    [data-testid="stDeployButton"], #MainMenu, [data-testid="stMainMenu"],
    [data-testid="stToolbarActions"], [data-testid="stConnectionStatus"] {
        display: none !important;
    }
    footer { display: none !important; }

    /* App background — flat warm ivory, no gradients, no glows */
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    .stApp::before, .stApp::after { content: none !important; }

    /* Main column width — Claude keeps content narrow and readable */
    .block-container { max-width: 1180px !important; padding-top: 2.6rem !important; }

    /* Sidebar — slightly deeper warm tone, hairline divider */
    [data-testid="stSidebar"] {
        background: var(--panel) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: var(--text) !important;
    }

    /* Typography — serif display headings, like Claude */
    h1, h2, h3 {
        font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    h4, h5, h6 { color: var(--text) !important; font-weight: 600 !important; }
    p, span, div, li, label, .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {
        color: var(--text);
    }
    small, .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
        color: var(--muted) !important;
    }

    /* Cards: metrics & expanders — white, hairline border, soft radius */
    div[data-testid="stMetric"], div[data-testid="stExpander"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 3px rgba(31,30,29,0.05) !important;
        padding: 1rem 1.2rem !important;
    }
    div[data-testid="stExpander"] details { border: none !important; }
    div[data-testid="stExpander"] summary { color: var(--text) !important; }

    /* Buttons — Claude coral, rounded, quiet */
    .stButton > button, div[data-testid="stDownloadButton"] > button {
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: none !important;
        transition: background 0.15s ease !important;
    }
    .stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
        background: var(--accent-hover) !important;
        color: #FFFFFF !important;
        transform: none !important;
    }
    /* Secondary buttons (sidebar & secondary kind) — outline style */
    [data-testid="stSidebar"] .stButton > button, button[kind="secondary"] {
        background: transparent !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover, button[kind="secondary"]:hover {
        background: var(--hover) !important;
        border-color: var(--border2) !important;
    }
    /* Sidebar navigation boxes — left-aligned icon + label, compact, tile look */
    [data-testid="stSidebar"] .stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.5rem 0.85rem !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        color: #FFFFFF !important;
        border: none !important;
    }
    .nav-title {
        font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--muted); font-weight: 700; margin: 0.2rem 0 0.5rem;
    }

    /* Inputs — hairline border, coral focus ring */
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea,
    div[data-baseweb="select"] > div, div[data-testid="stAudioInput"] {
        background: var(--card) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    /* Placeholders MUST be visible in dark mode (was blending into black) */
    input::placeholder, textarea::placeholder {
        color: var(--muted) !important; opacity: 1 !important;
    }

    /* Dropdown / selectbox / multiselect POPUP menu — renders in a detached
       portal, so it needs its own rules or the options look washed-out. Force
       readable text + card background + a clear coral highlight on hover. */
    div[data-baseweb="popover"] ul[role="listbox"],
    div[data-baseweb="popover"] div[data-baseweb="menu"],
    ul[data-baseweb="menu"], div[data-baseweb="menu"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 28px rgba(0,0,0,0.28) !important;
    }
    div[data-baseweb="popover"] li[role="option"],
    ul[data-baseweb="menu"] li, div[data-baseweb="menu"] li,
    li[role="option"] {
        background: transparent !important;
        color: var(--text) !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }
    /* Hover / keyboard-highlighted option — coral tint, never faint */
    div[data-baseweb="popover"] li[role="option"]:hover,
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    li[role="option"][data-highlighted="true"] {
        background: var(--hover) !important;
        color: var(--text) !important;
    }
    /* The little text label inside each option (BaseWeb wraps it in a div) */
    div[data-baseweb="popover"] li[role="option"] *,
    li[role="option"] * { color: var(--text) !important; opacity: 1 !important; }
    /* Selected chips inside a multiselect */
    span[data-baseweb="tag"] { background: var(--accent) !important; }
    span[data-baseweb="tag"] span { color: #ffffff !important; }

    /* Chat input — kill the white corners: force every nested layer to the card
       colour and match radii so no white frame peeks through. */
    [data-testid="stChatInput"] {
        background: var(--bg) !important; border-radius: 16px !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] textarea {
        background: var(--card) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 14px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important; opacity: 1 !important;
    }
    [data-testid="stChatInput"] button {
        background: var(--accent) !important; border-radius: 10px !important;
    }
    [data-testid="stChatInput"] button svg { fill: #ffffff !important; }

    /* ===== Chat composer: ONE rounded box (input on top, ➕ / 🎤 row inside the
       bottom), styled like a modern chat composer. ===== */
    .st-key-composer {
        border: 1px solid var(--border) !important;
        border-radius: 22px !important;
        background: var(--card) !important;
        padding: 0.5rem 0.9rem 0.35rem !important;
        box-shadow: 0 2px 14px rgba(0,0,0,0.06) !important;
    }
    /* The input inside the composer is seamless — the box is the only border. */
    .st-key-composer [data-testid="stChatInput"] {
        border: none !important; background: transparent !important;
    }
    .st-key-composer [data-testid="stChatInput"] > div,
    .st-key-composer [data-testid="stChatInput"] [data-baseweb="textarea"],
    .st-key-composer [data-testid="stChatInput"] [data-baseweb="base-input"],
    .st-key-composer [data-testid="stChatInput"] textarea {
        background: transparent !important; border: none !important;
    }
    /* Control row: background spans fully from ➕ sign to mic / timer */
    .st-key-composer [data-testid="stHorizontalBlock"] {
        align-items: center !important; gap: 0.5rem !important;
        margin-top: 0.2rem !important;
        background: var(--panel) !important;
        border-radius: 16px !important;
        padding: 0.2rem 0.75rem !important;
    }
    .st-key-composer .stPopover {
        display: inline-flex !important;
        align-items: center !important;
    }
    /* ➕ button: NO circle border outline, clean flat icon button */
    .st-key-composer .stPopover button {
        background: transparent !important; border: none !important;
        outline: none !important; box-shadow: none !important;
        color: var(--accent) !important; border-radius: 0 !important;
        width: auto !important; height: auto !important;
        min-width: 0 !important; min-height: 0 !important;
        padding: 0.1rem 0.3rem !important; font-size: 1.35rem !important;
        display: inline-flex !important; align-items: center !important;
        justify-content: center !important;
    }
    .st-key-composer .stPopover button:hover,
    .st-key-composer .stPopover button:focus,
    .st-key-composer .stPopover button:active {
        background: transparent !important; border: none !important;
        color: var(--accent-hover) !important; box-shadow: none !important;
    }
    /* Completely hide any chevron down-arrow icon inside popover button */
    .st-key-composer .stPopover button svg,
    .st-key-composer .stPopover button [data-testid="stPopoverButtonIcon"],
    .st-key-composer .stPopover button [data-testid="stIconMaterial"],
    .st-key-composer .stPopover button [aria-hidden="true"] {
        display: none !important;
    }
    .st-key-composer .stPopover button p,
    .st-key-composer .stPopover button div,
    .st-key-composer .stPopover button span {
        margin: 0 !important; padding: 0 !important;
        line-height: 1 !important; display: inline-flex !important;
        align-items: center !important; justify-content: center !important;
    }
    /* Audio input / mic & timer — seamless background matching control row */
    .st-key-composer [data-testid="stAudioInput"],
    .st-key-composer [data-testid="stAudioInput"] > div,
    .st-key-composer [data-testid="stAudioInput"] [data-baseweb="base-input"] {
        background: transparent !important; border: none !important;
        box-shadow: none !important;
        min-height: 2.0rem !important; width: 100% !important;
        display: flex !important; justify-content: flex-end !important;
        align-items: center !important; overflow: visible !important;
    }
    .st-key-composer [data-testid="stAudioInput"] *,
    .st-key-composer [data-testid="stAudioInput"] div {
        background-color: transparent !important;
        overflow: visible !important;
    }
    .st-key-composer [data-testid="stColumn"]:last-child {
        display: flex !important; justify-content: flex-end !important;
    }

    /* ===== Floating theme (dark/light) toggle — top-right corner icon.
       MUST sit above Streamlit's header (z-index ~999990) or it gets hidden. ===== */
    .st-key-theme_toggle {
        position: fixed !important; top: 0.5rem; right: 1rem;
        z-index: 1000001 !important; width: auto !important; min-width: 0 !important;
    }
    .st-key-theme_toggle button {
        background: var(--accent) !important; border: 1px solid var(--accent) !important;
        color: #ffffff !important; border-radius: 50% !important;
        width: 2.6rem; height: 2.6rem; padding: 0 !important;
        font-size: 1.25rem !important; box-shadow: 0 3px 12px rgba(0,0,0,0.25) !important;
    }
    .st-key-theme_toggle button:hover { filter: brightness(1.08) !important; }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--card) !important;
        border: 1px dashed var(--border2) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }

    /* Tabs — quiet underline style, coral active. Allow WRAP + extra top space
       so tab labels are never clipped at the top of the page. */
    .stTabs { margin-top: 0.4rem !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem; background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        flex-wrap: wrap !important; overflow: visible !important;
        padding-top: 0.35rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--muted) !important; background: transparent !important;
        border-radius: 8px 8px 0 0 !important; padding: 0.45rem 0.9rem !important;
        height: auto !important; white-space: nowrap !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important; font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

    /* Page-open animation: each tab panel fades/slides in when selected. */
    @keyframes potbFadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .stTabs [data-baseweb="tab-panel"] { animation: potbFadeIn 0.28s ease both; }
    [data-testid="stChatMessage"] { animation: potbFadeIn 0.25s ease both; }

    /* Tables / dataframes */
    [data-testid="stTable"] table, .stDataFrame {
        background: var(--card) !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 12px !important;
    }
    [data-testid="stTable"] th { background: var(--panel) !important; color: var(--text) !important; }
    [data-testid="stTable"] td { color: var(--text) !important; }

    /* Alerts — soft warm tints */
    div[data-testid="stAlert"] { border-radius: 12px !important; border: 1px solid var(--border) !important; }

    /* Chat bubbles — Claude style: user in soft card, assistant plain */
    .chat-bubble-container { display: flex; margin: 0.35rem 0; }
    .user-chat-bubble {
        margin-left: auto; background: var(--panel);
        border: 1px solid var(--border);
        padding: 0.7rem 1rem; border-radius: 14px; max-width: 82%;
        color: var(--text);
    }
    .assistant-chat-bubble {
        margin-right: auto; background: transparent;
        padding: 0.3rem 0.1rem; max-width: 100%;
        color: var(--text);
        font-size: 1.0rem; line-height: 1.65;
    }

    /* Hero */
    .custom-hero { display: flex; align-items: center; gap: 1.2rem; padding: 0.6rem 0 0.4rem 0; }
    .custom-hero h1 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: 2.4rem !important; margin: 0 !important; color: var(--text) !important;
    }
    .custom-hero p { color: var(--muted) !important; margin: 0.2rem 0 0 0 !important; }
    .custom-hero img { max-height: 84px; border-radius: 12px; }

    /* Metric cards grid (hero) */
    .custom-metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin: 0.8rem 0; }
    .custom-metric-card {
        background: var(--card); border: 1px solid var(--border);
        border-radius: 14px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(31,30,29,0.05);
    }
    .custom-metric-title { color: var(--muted); font-size: 0.8rem; letter-spacing: 0.04em; text-transform: uppercase; }
    .custom-metric-value {
        color: var(--text); font-size: 1.7rem; font-weight: 600;
        font-family: 'Source Serif 4', Georgia, serif;
    }
    .custom-metric-subtitle { color: var(--accent); font-size: 0.85rem; }

    /* Toggles & progress in coral */
    [data-baseweb="checkbox"] [data-checked="true"], .stProgress > div > div { background-color: var(--accent) !important; }

    /* Scrollbar — subtle */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }

    /* ===== MOBILE / FIELD-TECHNICIAN layout (PS #8: "built to work on mobile
       for field technicians, not just desktops"). Tuned for phones ≤640px. ===== */
    @media (max-width: 640px) {
        /* Reclaim horizontal space — thin page padding on phones */
        .block-container { padding: 1rem 0.7rem 4rem !important; }
        /* Hero title fits a phone without wrapping awkwardly */
        h1 { font-size: 1.55rem !important; line-height: 1.2 !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }
        /* Dashboard 2-col layouts stack vertically instead of squishing */
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            min-width: 100% !important; flex: 1 1 100% !important;
        }
        /* Keep the composer's ➕ / mic row on ONE line (don't let it stack) */
        .st-key-composer [data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
        }
        .st-key-composer [data-testid="stColumn"] {
            min-width: auto !important; flex: 0 0 auto !important;
        }
        .st-key-composer [data-testid="stColumn"]:nth-child(2) {
            flex: 1 1 auto !important;
        }
        /* Tables/dataframes scroll horizontally instead of breaking the layout */
        [data-testid="stTable"], .stDataFrame, [data-testid="stDataFrame"] {
            overflow-x: auto !important; display: block !important;
        }
        /* Smaller, thumb-friendly theme toggle that never covers content */
        .st-key-theme_toggle button { width: 2.2rem !important; height: 2.2rem !important; }
        /* Sidebar nav boxes a touch taller for tapping */
        [data-testid="stSidebar"] .stButton > button { padding: 0.6rem 0.85rem !important; }
    }
    /* Never allow the page body itself to scroll sideways on any device */
    .stApp, .block-container { max-width: 100%; overflow-x: hidden; }
    </style>
    """, unsafe_allow_html=True)


def render_hero_with_image():
    import base64
    img_path = Path("assets/factory.png")
    img_html = ""
    if img_path.exists():
        b64_data = base64.b64encode(img_path.read_bytes()).decode()
        img_html = f'<img src="data:image/png;base64,{b64_data}" style="max-height: 120px; border-radius: 12px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.05));" />'
        
    st.markdown(f"""
    <div class="custom-hero" style="display: flex; justify-content: space-between; align-items: center; gap: 20px;">
        <div style="flex: 1;">
            <h1 style="margin: 0 0 0.5rem 0 !important; font-size: clamp(1.7rem, 7vw, 3.3rem) !important;
                       font-weight: 900 !important; letter-spacing: -2px !important;
                       line-height: 1.05 !important;
                       background: linear-gradient(92deg, #22d3ee 0%, #8a5cff 60%, #ec4899 110%) !important;
                       -webkit-background-clip: text !important;
                       background-clip: text !important;
                       -webkit-text-fill-color: transparent !important;
                       color: #22d3ee !important;">🏭 Plant Operations Brain</h1>
            <p style="margin: 0 !important; font-size: 1.1rem !important;
                      letter-spacing: 0.2px !important; color: #64748b !important;">
                Any industry. Any document. One honest brain.</p>
        </div>
        {img_html}
    </div>
    """, unsafe_allow_html=True)

# (Title/hero now renders inside the Dashboard tab only — the Copilot opens as
#  a clean full-page chat, and the tools show their own content.)


@st.cache_resource(show_spinner="Loading the knowledge base...")
def get_brain():
    """Build (or load cached) vector index + knowledge graph once, then reuse.
    Returns (PlantBrain | None, graph_backend_name)."""
    store = VectorStore()
    graph, backend = make_graph()   # Neo4j if configured, else in-memory
    reranker = Reranker()           # GPU cross-encoder second-stage retrieval
    reranker.warm()                 # load the model now so first query is fast
    if store.load(INDEX_DIR) and graph.load(INDEX_DIR):
        return PlantBrain(store, graph=graph, reranker=reranker), backend

    # First run: build everything from the documents.
    chunks = ingest_folder(DOCS_DIR)
    if not chunks:
        return None, backend
    store.build(chunks)
    store.save(INDEX_DIR)
    graph.build(chunks, cache_dir=INDEX_DIR)   # uses the LLM; cached to disk
    graph.save(INDEX_DIR)
    return PlantBrain(store, graph=graph, reranker=reranker), backend


def _clear_and_rebuild():
    """Drop the cached index so the next load re-reads all documents."""
    st.cache_resource.clear()
    import shutil
    if Path(INDEX_DIR).exists():
        shutil.rmtree(INDEX_DIR)


inject_premium_ui(st.session_state.get("ui_dark", False))

# ===== First-launch language gate: the whole site renders in this language =====
from src.i18n import t as _t, td as _td, cached_languages as _cached_langs

if "site_lang" not in st.session_state:
    st.markdown("<div style='height:14vh'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center'>🏭 Plant Operations Brain</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center'>Choose your language · "
                "अपनी भाषा चुनें · ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಆರಿಸಿ · உங்கள் மொழியைத் "
                "தேர்ந்தெடுக்கவும் · మీ భాషను ఎంచుకోండి</p>",
                unsafe_allow_html=True)
    _c1, _c2, _c3 = st.columns([1, 2, 1])
    with _c2:
        _pick = st.selectbox("Language / भाषा / ಭಾಷೆ / மொழி",
                             _cached_langs(), key="lang_pick")
        if st.button("Continue →", type="primary", use_container_width=True):
            st.session_state.site_lang = _pick
            st.rerun()
    st.stop()

LANG = st.session_state.site_lang


def T(s: str) -> str:
    """Translate a UI string into the site language (cached, zero API)."""
    return _t(s, LANG)


def TD(s) -> str:
    """Translate a dynamic DATA value (sensor/asset name, risk level) — cached."""
    return _td(s, LANG)

# --- Floating theme toggle: a single icon in the top-right corner. Clicking it
#     flips the whole site between dark and light. (No sidebar toggle anymore.) ---
_dark_now = st.session_state.get("ui_dark", False)
with st.container(key="theme_toggle"):
    if st.button("☀️" if _dark_now else "🌙", key="theme_btn",
                 help="Toggle dark / light"):
        st.session_state.ui_dark = not _dark_now
        st.rerun()

# --- Sidebar navigation: the app's sections as small boxes (like a modern app
#     shell). Selecting one shows that section; everything else is hidden. ---
NAV_ITEMS = [
    ("📊", "Dashboard"),
    ("💬", "Chat"),
    ("🛡️", "Operations"),
    ("📡", "Live Sensor Data"),
    ("🎙️", "Capture Interview"),
    ("📚", "History"),
]
NAV_KEY = {
    "Dashboard": "grp_dash", "Chat": "grp_chat", "Operations": "grp_ops",
    "Live Sensor Data": "grp_live", "Capture Interview": "grp_capture",
    "History": "grp_history",
}
st.session_state.setdefault("nav", "Dashboard")

with st.sidebar:
    st.markdown(f"<div class='nav-title'>🧭 {T('Navigate')}</div>",
                unsafe_allow_html=True)
    for _icon, _name in NAV_ITEMS:
        _active = st.session_state.nav == _name
        if st.button(f"{_icon} {T(_name)}", key=f"nav_{_name}",
                     use_container_width=True,
                     type=("primary" if _active else "secondary")):
            st.session_state.nav = _name
            st.rerun()
    st.divider()

# --- Sidebar: upload + status + rebuild control ---
with st.sidebar:
    st.header(T("Knowledge base"))
    # Language is chosen ONCE at launch and drives the whole site — it is not
    # asked again on any page. A quiet "change" link returns to the start screen.
    _lang_l, _lang_r = st.columns([3, 2])
    _lang_l.caption(f"🌐 {LANG}")
    if _lang_r.button(T("Change"), key="change_lang", use_container_width=True):
        del st.session_state["site_lang"]
        st.rerun()
    st.caption(f"AI provider: {provider_status()}")

    # Industry-agnostic by design: the documents drive the knowledge; this
    # selector rescales the business-impact (₹) model for the chosen industry.
    from src.costs import INDUSTRY_PROFILES, set_industry
    industry = st.selectbox(T("🏭 Industry profile"), list(INDUSTRY_PROFILES),
                            key="industry_profile")
    _mult = set_industry(industry)
    if _mult != 1.0:
        st.caption(f"Cost model scaled ×{_mult:g} for this industry.")

    doc_count = len(list(Path(DOCS_DIR).rglob("*"))) if Path(DOCS_DIR).exists() else 0
    st.metric(T("Files in data/docs"), doc_count)

    st.subheader(T("📤 Upload documents"))
    uploaded = st.file_uploader(
        "PDF · Word · Excel · CSV · text · email · drawings",
        type=["pdf", "docx", "xlsx", "xlsm", "csv", "tsv", "txt", "md",
              "eml", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    if uploaded and st.button(T("➕ Add files & rebuild"), type="primary"):
        Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)
        saved = 0
        for f in uploaded:
            dest = Path(DOCS_DIR) / f.name
            dest.write_bytes(f.getvalue())
            saved += 1
        _clear_and_rebuild()
        st.success(f"Added {saved} file(s). Rebuilding…")
        st.rerun()

    st.divider()
    if st.button(T("🔄 Rebuild index")):
        _clear_and_rebuild()
        st.rerun()

    with st.expander(T("🗑️ Manage files")):
        existing = sorted(p.name for p in Path(DOCS_DIR).glob("*")
                          if p.is_file()) if Path(DOCS_DIR).exists() else []
        to_delete = st.multiselect(T("Select files to remove"), existing)
        if to_delete and st.button(T("Delete selected & rebuild")):
            for name in to_delete:
                (Path(DOCS_DIR) / name).unlink(missing_ok=True)
            _clear_and_rebuild()
            st.warning(f"Deleted {len(to_delete)} file(s). Rebuilding…")
            st.rerun()

brain, graph_backend = get_brain()

with st.sidebar:
    st.caption(f"🕸️ Graph backend: **{graph_backend}**")
    # "Continuously updated": detect files added since the index was built.
    if brain is not None:
        indexed = {c.source_file for c in brain.store.chunks}
        from src.ingest import _LOADERS
        on_disk = {p.name for p in Path(DOCS_DIR).glob("*")
                   if p.suffix.lower() in _LOADERS}
        new_files = sorted(on_disk - indexed)
        if new_files:
            st.warning(f"🆕 {len(new_files)} new file(s) not yet indexed: "
                       f"{', '.join(new_files[:3])}"
                       f"{'…' if len(new_files) > 3 else ''} — click **Rebuild index**.")

    with st.expander(T("🏗️ Why this matters & how it scales")):
        st.markdown(
            "**The problem (from published studies):**\n"
            "- Engineers lose **35% of hours** searching for information (McKinsey 2024)\n"
            "- **7–12 disconnected** document systems per plant (NASSCOM-EY)\n"
            "- Fragmentation drives **18–22% of unplanned downtime** (BIS Research)\n"
            "- **25% of senior engineers retire** within a decade\n\n"
            "**How this scales — any industry:**\n"
            "- **Industry-agnostic engine**: nothing is hard-coded to one plant — "
            "upload YOUR documents and the graph, sensors (limits read from your "
            "manuals), compliance and RCA reconfigure themselves\n"
            "- Industry profile selector rescales the ₹ business-impact model\n"
            "- Embeddings run **locally on GPU** — retrieval cost ≈ ₹0, data stays on-site\n"
            "- Graph backend swaps **in-memory ⇄ Neo4j cluster** with one .env line\n"
            "- LLM provider is swappable with one .env line (no vendor lock-in)\n"
            "- Ships with a **Dockerfile** — one container per site, same engine")

    st.markdown(
        "<div style='margin-top:1.2rem; padding-top:0.6rem; "
        "border-top:1px solid var(--border); font-size:0.72rem; "
        "color:var(--muted);'>© 2026 Plant Operations Brain Team · "
        "ET AI Hackathon · All rights reserved.<br>Proprietary — "
        "not for reuse or redistribution.</div>", unsafe_allow_html=True)

if brain is None:
    st.warning("No documents found. Add files to **data/docs** and click *Rebuild index*.")
    st.stop()

# Corpus-wide scans (sensor discovery, knowledge risk, pending items) only change
# when the documents change — compute them ONCE per session, not every rerun, so
# clicking around the app stays snappy.
import re as _re
from src.sensors import get_registry as _get_registry
from src.interview import knowledge_risk_map as _krm
from src.health import _PENDING_RE as _PEND

_sig = len(brain.store.chunks)
if st.session_state.get("_static_sig") != _sig:
    st.session_state._registry = _get_registry(brain.store.chunks)
    st.session_state._risk = _krm(brain)
    st.session_state._pending = sum(bool(_PEND.search(c.text))
                                    for c in brain.store.chunks)
    st.session_state._ndocs = len({c.source_file for c in brain.store.chunks})
    st.session_state._static_sig = _sig
_registry = st.session_state._registry

# ===== Hero: Plant Health + Business Impact (the first thing judges see) =====
_n_conf = (len(st.session_state["conflicts"])
           if "conflicts" in st.session_state else None)
_health = plant_health(brain, hours=48, n_conflicts=_n_conf,
                       registry=_registry, risk=st.session_state._risk,
                       pending=st.session_state._pending)
# Corpus-aware: only counts events whose source document is actually ingested,
# so the KPI can never claim savings the current documents don't support.
_pv = prevention_value(brain.store.chunks)
_prevented = sum(v for v, _ in _pv.values())
_n_events = len(_pv)

def _render_hero_metrics():
    """The Plant-Health / cost / documents cards — shown on the Dashboard only."""
    st.markdown(f"""
    <div class="custom-metric-grid">
        <div class="custom-metric-card">
            <div class="custom-metric-title">{T("🏥 Plant Health")}</div>
            <div class="custom-metric-value">{_health.total}/100</div>
            <div class="custom-metric-subtitle">Grade {_health.grade} • {
                {"A": "Healthy", "B": "Stable", "C": "Needs attention",
                 "D": "Critical"}.get(_health.grade, "")}</div>
        </div>
        <div class="custom-metric-card">
            <div class="custom-metric-title">{T("💰 Avoided Cost (est.)")}</div>
            <div class="custom-metric-value">{fmt_inr(_prevented) if _n_events else "—"}</div>
            <div class="custom-metric-subtitle">{
                f"scenario estimate · {_n_events} documented event"
                + ("s" if _n_events != 1 else "")
                if _n_events else "no priced events in this corpus"}</div>
        </div>
        <div class="custom-metric-card">
            <div class="custom-metric-title">{T("📄 Documents Live")}</div>
            <div class="custom-metric-value">{st.session_state._ndocs}</div>
            <div class="custom-metric-subtitle">{brain.graph.g.number_of_nodes()} graph entities</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander(T("📊 How these numbers are computed (fully auditable)")):
        st.markdown("**Plant Health** — weighted, deterministic sub-scores:")
        for name, (score, note) in _health.subscores.items():
            st.progress(score, text=f"{name}: {note}")
        st.markdown("**Avoided cost** — scenario estimates from editable assumptions "
                    "in `src/costs.py`, shown only for events whose source document "
                    "is in the corpus right now:")
        if _pv:
            for name, (val, basis) in _pv.items():
                st.markdown(f"- **{fmt_inr(val)}** — {name} · _{basis}_")
        else:
            st.caption("No priced incidents in the current documents — upload plant "
                       "incident/permit records to populate this.")


def _show_sources(sources):
    if sources:
        st.divider()
        st.subheader(T("📎 Sources"))
        for i, c in enumerate(sources, start=1):
            loc = f"{c.source_file}" + (f" — page {c.page}" if c.page else "")
            with st.expander(f"Source {i}: {loc}"):
                st.write(c.text)


from contextlib import contextmanager


@contextmanager
def ai_guard():
    """Turn an API failure mid-demo into a calm message instead of a red crash."""
    try:
        yield
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ("quota", "429", "resource_exhausted",
                                  "rate limit", "limit: 0", "exhausted")):
            st.warning("⏳ The AI provider is rate-limited right now. "
                       "Wait ~30 seconds and try again.")
        elif any(x in msg for x in ("api key", "api_key", "missing", "unauthor")):
            st.error("🔑 AI provider key issue — check your key in the `.env` file.")
        else:
            st.error(f"Something went wrong handling that request: {str(e)[:180]}")
        st.stop()


# Capabilities grouped into 6 logical sections. Navigation lives in the SIDEBAR
# (small boxes); each section is a keyed container and only the active one is
# shown — the rest are hidden client-side (instant, like tabs, no reruns).
nav = st.session_state.nav
grp_dash = st.container(key="grp_dash")
grp_chat = st.container(key="grp_chat")
grp_ops = st.container(key="grp_ops")
grp_live = st.container(key="grp_live")
grp_capture = st.container(key="grp_capture")
grp_history = st.container(key="grp_history")

_active_key = NAV_KEY[nav]
_hide = "".join(f".st-key-{k}{{display:none !important;}}"
                for k in NAV_KEY.values() if k != _active_key)
st.markdown(f"<style>{_hide}</style>", unsafe_allow_html=True)

with grp_dash:
    render_hero_with_image()          # the "Plant Operations Brain" title — here only
    _render_hero_metrics()            # health / cost / documents cards
    st.divider()
    st.markdown(f'<h3 style="margin-top:0;">📊 {T("Operations & Risk Overview")}</h3>', unsafe_allow_html=True)

    # 2-column layout for dashboard visuals
    dash_col1, dash_col2 = st.columns([3, 2])
    
    with dash_col1:
        st.markdown(f'<h4 style="margin: 0.5rem 0 0.5rem 0;">📡 {T("Real-time Sensor Trends")}</h4>', unsafe_allow_html=True)
        # Load registry and plot main sensor trends side-by-side
        if _registry:
            main_sensors = list(_registry.keys())[:3]
            sensor_trends = {}
            for sname in main_sensors:
                sensor_trends[TD(sname)] = series(sname, 48, _registry)
            line_chart(sensor_trends, x_title=T("hours"))
        else:
            st.info(T("No active sensors found in documents."))

        st.markdown(f'<h4 style="margin: 1.5rem 0 0.5rem 0;">🏭 {T("Document Distribution")}</h4>', unsafe_allow_html=True)
        # Display document counts & graph structure summary
        if brain:
            nodes_data = [d.get("type", "Unknown") for n, d in brain.graph.g.nodes(data=True)]
            type_counts = {}
            for t in nodes_data:
                type_counts[T(t)] = type_counts.get(T(t), 0) + 1
            bar_chart(type_counts, x_title=T("entities"))
            
    with dash_col2:
        # Render a clean, styled unified checklist card of alerts/actions
        st.markdown(f"""
        <div class="custom-metric-card" style="padding:1.5rem; border-color: #f59e0b !important; margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 1rem 0;">⚠️ {T("Active Risk & Compliance Alerts")}</h4>
            <div style="font-size:0.95rem; line-height:1.6;">
                <div style="margin-bottom:0.75rem; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:0.5rem;">
                    <span style="color:#ef4444; font-weight:600;">🔴 {T("Open actions")}:</span> {TD(_health.subscores.get("Open actions (20%)", (0, "none found"))[1])}
                </div>
                <div style="margin-bottom:0.75rem; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:0.5rem;">
                    <span style="color:#f59e0b; font-weight:600;">🟠 {T("Document Conflicts")}:</span> {TD(_health.subscores.get("Conflicts (15%)", (0, "scan not run yet"))[1])}
                </div>
                <div style="padding-bottom:0.5rem;">
                    <span style="color:#3b82f6; font-weight:600;">🔵 {T("Pending Checklist Items")}:</span> {st.session_state._pending} {T("items need operator review.")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<h4 style="margin: 1.5rem 0 0.5rem 0;">🔧 {T("High-Risk Asset Analysis (RCA)")}</h4>', unsafe_allow_html=True)
        if st.session_state._risk:
            # Display risk levels in a readable clean table format
            risk_summary = [{T("Asset"): TD(r.equipment), T("Risk Level"): T(r.risk)}
                            for r in st.session_state._risk[:3]]
            st.table(risk_summary)
        else:
            st.info(T("No high-risk assets cataloged."))

with grp_chat:
    tab_ask = st.container()          # full-page chat, no redundant sub-tab
with grp_ops:
    tab_safety, tab_rca, tab_comp, tab_conflict, tab_impact = st.tabs(
        [T("🛡️ Safety Check"), T("🔧 Maintenance / RCA"), T("📋 Compliance"),
         T("⚠️ Conflicts → Fix"), T("💥 Failure Impact")])
with grp_live:
    tab_watch, tab_handover = st.tabs(
        [T("📡 Plant Watch"), T("📝 Shift Handover")])
with grp_capture:
    tab_tribal = st.container()       # Capture Interview — its own section now
with grp_history:
    tab_files, tab_lessons = st.tabs(
        [T("📁 Documents"), T("📚 Lessons Learned")])
# Knowledge Graph is now backend-only: it still powers GraphRAG retrieval and
# the Impact analysis, but is no longer shown as its own tab.

# ----- Tab 1: the copilot — clean full-page chat -----
with tab_ask:
    from src import chat_store as _cs
    import time as _time
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = f"c{int(_time.time()*1000)}"

    def _new_chat():
        # Persist the current conversation (if any) before starting fresh.
        _cs.save_chat(st.session_state.chat_id, st.session_state.chat)
        st.session_state.chat = []
        st.session_state.chat_id = f"c{int(_time.time()*1000)}"
        for _k in ("photo_text", "voice_text", "last_photo_sig",
                   "last_voice_sig", "last_auto_sig"):
            st.session_state.pop(_k, None)

    # Top corner: recent chat history (📚) + New chat (🖊).
    _nc_l, _nc_hist, _nc_r = st.columns([8, 1, 1])
    with _nc_hist.popover("📚", help=T("Chat history"), use_container_width=True):
        _saved = _cs.list_chats()
        st.markdown(f"**{T('Recent chats')}**")
        if not _saved:
            st.caption(T("No saved chats yet."))
        for _rec in _saved:
            _cols = st.columns([5, 1])
            if _cols[0].button("💬 " + (_rec["title"] or "Chat"),
                               key=f"open_{_rec['id']}",
                               use_container_width=True):
                # Save whatever is open, then load the selected conversation.
                _cs.save_chat(st.session_state.chat_id, st.session_state.chat)
                st.session_state.chat = _cs.get_chat(_rec["id"])
                st.session_state.chat_id = _rec["id"]
                st.rerun()
            if _cols[1].button("🗑", key=f"del_{_rec['id']}",
                               help=T("Delete this chat")):
                _cs.delete_chat(_rec["id"])
                st.rerun()
    if _nc_r.button("🖊", help=T("New chat"), key="clear_chat",
                    use_container_width=True):
        _new_chat()
        st.rerun()

    # Empty-state: greeting + chat box only.
    if not st.session_state.get("chat"):
        st.markdown(
            "<div style='text-align:center; padding:3rem 1rem 1rem; color:var(--muted);'>"
            "<div style='font-size:2.6rem;'>🏭</div>"
            f"<h3 style='margin:0.4rem 0;'>{T('How can I help with your plant today?')}</h3>"
            f"<p>{T('Try: “What is the lockout procedure for Pump 7?” or “Why does P-7 keep failing?”')}</p>"
            "</div>", unsafe_allow_html=True)

    # Replay the conversation so far (newest ends just above the input box).
    for turn in st.session_state.chat:
        role = turn[0]
        content = turn[1]
        if role == "user":
            st.markdown(f'<div class="user-chat-bubble" style="float:right; '
                        f'clear:both; margin-bottom:0.6rem;">{content}</div>',
                        unsafe_allow_html=True)
            continue
        # assistant turn: ("assistant", text, sources, extras)
        sources = turn[2] if len(turn) > 2 else []
        extras = turn[3] if len(turn) > 3 else {}
        if extras.get("route"):
            st.caption(f"🤖 {extras['route']}")
        if extras.get("confident") is False:
            st.error(T("No documented answer — the system refuses to guess."))
        st.markdown(f'<div class="assistant-chat-bubble" style="float:left; '
                    f'clear:both; margin-bottom:0.3rem;">{content}</div>',
                    unsafe_allow_html=True)
        if extras.get("score"):
            st.caption(f"🎯 Retrieval confidence: **{extras['score']:.0%}**")
        if extras.get("audio"):
            st.audio(extras["audio"], format="audio/wav")
        if extras.get("trace"):
            with st.expander("🔬 Reasoning trace — how the agent got here"):
                for i, (sq, a) in enumerate(extras["trace"], 1):
                    st.markdown(f"**Step {i} — {sq}**")
                    st.caption(a)
        if sources:
            with st.expander("📎 Sources"):
                for i, c in enumerate(sources, start=1):
                    loc = f"{c.source_file}" + (f" — p.{c.page}" if c.page else "")
                    st.markdown(f"**Source {i}: {loc}**")
                    st.caption(c.text)

    # Answers use the language already chosen at launch — never ask again.
    language = LANG

    # --- Composer: ONE rounded box. The text field sits on top; a control row
    #     with ➕ (options + photo) on the left and the 🎤 mic on the right sits
    #     inside the bottom of the same box — like a modern chat composer. ---
    with st.container(key="composer"):
        typed = st.chat_input(T("Ask a question (follow-ups are remembered)…"))
        _cl, _cmid, _cr = st.columns([1, 4.5, 2.5])
        with _cl.popover("➕", help=T("Tools & options")):
            auto_mode = st.toggle(T("🤖 Agentic"), value=True,
                                  help="The AI decides which specialist agent "
                                       "(RCA, Safety, Compliance…) should handle "
                                       "your question — autonomous tool selection.")
            deep_mode = st.toggle(T("🔬 Deep analysis"),
                                  help="Force multi-step investigation with a "
                                       "visible reasoning trace.")
            speak_mode = st.toggle(T("🔊 Speak answers"),
                                   help="Reads answers aloud (offline TTS).")
            st.markdown("---")
            photo = st.file_uploader(T("📷 Photo of equipment / nameplate / gauge"),
                                     type=["jpg", "jpeg", "png"], key="ask_photo")
        voice = _cr.audio_input(T("🎤 Speak"), key="ask_voice",
                                label_visibility="collapsed")

    # Photo -> equipment id (cached per image).
    photo_question = None
    if photo is not None:
        psig = hash(photo.getvalue())
        if st.session_state.get("last_photo_sig") != psig:
            with st.spinner(T("Reading the photo...")):
                seen = vision(
                    "Identify the equipment tag, gauge reading, or nameplate text "
                    "in this industrial photo. Reply with the key identifier only.",
                    photo.getvalue(), mime=photo.type or "image/jpeg")
            st.session_state.last_photo_sig = psig
            st.session_state.photo_text = seen
        seen = st.session_state.get("photo_text")
        if seen:
            st.info(f"📷 {T('Read from photo')}: **{seen}**")
            photo_question = f"Tell me about {seen}"

    # Voice -> text (cached per clip; Gemini auto-detects the spoken language).
    voice_question = None
    if voice is not None:
        vsig = hash(voice.getvalue())
        if st.session_state.get("last_voice_sig") != vsig:
            with st.spinner(T("Transcribing your voice...")):
                st.session_state.voice_text = transcribe(
                    voice.getvalue(), mime=voice.type or "audio/wav")
            st.session_state.last_voice_sig = vsig
        voice_question = st.session_state.get("voice_text")
        if voice_question:
            st.info(f"🎤 {T('Heard')}: **{voice_question}**")

    # Decide what to submit. Typed always submits; voice/photo submit ONCE per
    # new clip/image (guarded by signature) so reruns don't loop.
    user_q, auto_sig = None, None
    if typed:
        user_q = typed
    elif voice_question:
        user_q, auto_sig = voice_question, ("voice", st.session_state.get("last_voice_sig"))
    elif photo_question:
        user_q, auto_sig = photo_question, ("photo", st.session_state.get("last_photo_sig"))

    if user_q and (typed or st.session_state.get("last_auto_sig") != auto_sig):
        st.markdown(f'<div class="user-chat-bubble" style="float: right; clear: both; margin-bottom: 1rem;">{user_q}</div>', unsafe_allow_html=True)
        
        with ai_guard():
            extras = {}                       # trace / sources / audio for replay
            if deep_mode:
                with st.spinner("🔬 Deep analysis: planning → investigating → synthesising..."):
                    da = deep_ask(brain, user_q, language=language)
                answer_text, answer_sources = da.final_md, da.sources
                extras["trace"] = da.steps
            elif auto_mode:
                # AGENTIC: the router decides which specialist handles this.
                # rules-only routing = 0 extra API calls (100% on the benchmark);
                # only the chosen agent spends quota.
                r = route(user_q, use_llm_fallback=False)
                with st.spinner(f"🤖 Routing → {AGENTS[r.agent].split('—')[0].strip()}…"):
                    res = dispatch(brain, user_q, r, language=language)
                extras["route"] = f"Routed to {AGENTS[res.agent].split('—')[0].strip()} — {res.route.reason}"
                answer_text, answer_sources = res.markdown, res.sources
            else:
                # Normalise history to (role, text) pairs (assistant turns also
                # carry sources/extras, which the brain doesn't need).
                _hist = [(t[0], t[1]) for t in st.session_state.chat]
                with st.spinner("Searching documents and reasoning..."):
                    answer = brain.ask(user_q, language=language, history=_hist)
                answer_text, answer_sources = answer.text, answer.sources
                extras["confident"] = answer.confident
                extras["score"] = answer.score

            # Speak only makes sense for English (Windows TTS voice is English).
            if speak_mode and language == "English":
                wav = tts_wav(answer_text)
                if wav:
                    extras["audio"] = wav

        # Persist the turn (with its extras) and rerun so it appears in the
        # conversation history ABOVE the input box — not stranded below it.
        st.session_state.chat.append(("user", user_q))
        st.session_state.chat.append(("assistant", answer_text, answer_sources, extras))
        if auto_sig is not None:
            st.session_state.last_auto_sig = auto_sig   # don't re-ask this clip
        # Auto-save so the conversation is stored even without "New chat".
        _cs.save_chat(st.session_state.chat_id, st.session_state.chat)
        st.rerun()

# ----- Tab: What-If Safety Check -----
with tab_safety:
    st.write(T("Describe work **before you do it** — the brain cross-checks every "
             "procedure, incident and permit, and answers **STOP / CAUTION / SAFE**."))
    example = st.selectbox(T("Try an example (or write your own below)"), [
        "",
        "Replacing the mechanical seal on P-7. I closed the suction valve SV-7; "
        "discharge valve DV-7 is still open.",
        "Doing welding on a bracket near compressor C-12. Gas detector shows "
        "LEL at 12%.",
        "Starting boiler B-3 with one cracked gauge glass; will replace it "
        "next week.",
        "Routine oil top-up on P-7 while it is running.",
    ])
    plan = st.text_area(T("Planned work"), value=example, height=90,
                        placeholder="e.g. About to open the C-12 valve cover "
                                    "for inspection…")
    if st.button(T("🛡️ Check before starting"), type="primary") and plan.strip():
        with ai_guard(), st.spinner("Cross-checking procedures, incidents and permits..."):
            v = check_plan(brain, plan.strip(), language=LANG)
        badge = {"STOP": st.error, "CAUTION": st.warning, "SAFE": st.success}
        badge.get(v.verdict, st.warning)(f"**{v.verdict}** — {v.headline}")
        if v.reasons:
            st.markdown("**Why:**")
            for r in v.reasons:
                st.markdown(f"- {r}")
        if v.required_actions:
            st.markdown("**Required actions before/during the work:**")
            for i, a in enumerate(v.required_actions, 1):
                st.markdown(f"{i}. {a}")
        _show_sources(v.sources)

# ----- Tab: Plant Watch (live sensor simulation + proactive alerts) -----
with tab_watch:
    st.write(T("Simulated live SCADA feed. The watchlist configures itself from "
               "the ingested documents: any alarm/trip limit written in a manual "
               "or SOP becomes a live sensor automatically."))
    registry = _registry   # reuse the once-per-session cached registry
    n_auto = sum(1 for c in registry.values() if c.get("source") != "curated")
    st.caption(f"📡 {len(registry)} {T('sensors on watch')} — "
               f"**{n_auto} {T('auto-discovered from documents')}**, "
               f"{len(registry) - n_auto} {T('curated')}.")
    wc1, wc2 = st.columns([2, 1])
    hours = wc1.slider(T("Hours into the shift simulation"), 1, 96, 48)
    _scen_opts = [T("Degrading (default)"), T("Healthy week")]
    scenario = wc2.radio(T("Scenario"), _scen_opts, horizontal=False,
                         help=T("'Healthy week' shows the same assets stable — "
                                "proof the readings aren't scripted to fail."))
    _drift = 0.0 if scenario == _scen_opts[1] else 1.0

    statuses = [status_at(name, hours, registry, _drift) for name in registry]
    icon = {"OK": "🟢", "WARNING": "🟠", "ALARM": "🔴"}
    st.table([{
        T("Sensor"): TD(s.name), T("Live value"): s.value,
        T("Alarm at"): s.alarm, T("Trip at"): s.trip,
        T("State"): f"{icon[s.state]} {T(s.state)}",
        T("Limits from"): (T("curated") if s.source == "curated"
                           else f"📄 {s.source}"),
    } for s in statuses])

    trending = [s for s in statuses if s.state in ("WARNING", "ALARM")]
    if trending:
        s = trending[0]
        st.error(f"{icon[s.state]} **{s.name} = {s.value}** "
                 f"(alarm {s.alarm} / trip {s.trip})")
        if st.button(T("🧠 Explain this alert from the documents"), key="alert_btn"):
            with ai_guard(), st.spinner("Fusing live reading with documented history..."):
                text, srcs = explain_alert(brain, s)
            st.markdown(text)
            _show_sources(srcs)
    else:
        st.success(T("All sensors within documented limits."))

    pick = st.selectbox(T("Trend view"), list(registry), format_func=TD)
    _s = status_at(pick, hours, registry, _drift)
    line_chart({TD(pick): series(pick, hours, registry, _drift)}, height=300)
    st.caption(f"{T('Alarm at')} {_s.alarm} · {T('Trip at')} {_s.trip} · "
               f"{T('now')} {_s.value} → {T(_s.state)}")

# ----- Tab: Shift Handover Brief -----
with tab_handover:
    st.write(T("Auto-generated brief for the **incoming shift**: live trends to "
             "watch, open actions from the documents, and lessons to remember. "
             "Skimmable in 60 seconds."))
    ho_hours = st.slider(T("Generate for this simulation time (hours)"), 1, 96, 48,
                         key="ho_hours")
    if st.button(T("📝 Generate handover brief"), type="primary"):
        with ai_guard(), st.spinner("Fusing live readings with documented open items..."):
            ho = generate_brief(brain, hours=ho_hours, language=LANG)
        st.markdown(ho.brief_md)
        st.download_button(T("⬇️ Download brief (markdown)"), ho.brief_md,
                           file_name="shift_handover.md")
        _show_sources(ho.sources)

# ----- Tab: Failure Impact Analysis -----
with tab_impact:
    st.write(T("Pick an asset — the knowledge graph traversal shows everything connected to it (procedures, permits, regulations, incidents, people), then summarises the failure blast-radius."))
    equip_nodes = sorted(n for n, d in brain.graph.g.nodes(data=True)
                         if d.get("type") == "Equipment")
    ic1, ic2 = st.columns([2, 1])
    target = ic1.selectbox(T("Equipment"), equip_nodes or ["P-7"])
    down_h = ic2.number_input(T("Downtime to price (hours)"), min_value=0.5,
                              max_value=72.0, value=4.0, step=0.5)
    if st.button(T("💥 Analyse failure impact"), type="primary"):
        with ai_guard(), st.spinner("Traversing the knowledge graph..."):
            imp = analyze_impact(brain, target, language=LANG)
        if imp.neighbours:
            cols = st.columns(len(imp.neighbours) or 1)
            for col, (etype, names) in zip(cols, imp.neighbours.items()):
                col.metric(etype, len(names))
        cl = downtime_cost(target, down_h)
        st.warning(f"💰 **Estimated cost if {target} is down {down_h:g} h: "
                   f"{fmt_inr(cl.amount)}** — basis: {cl.basis} (editable "
                   f"assumption in src/costs.py)")
        if imp.summary_md:
            st.markdown(imp.summary_md)
        html = subgraph_html(brain, target)
        if html:
            components.html(html, height=500, scrolling=True)
        _show_sources(imp.sources)

# Knowledge Graph is backend-only now — it powers GraphRAG retrieval and the
# Impact analysis, but has no visible tab (removed at the team's request).


# ----- Tab 3: Maintenance / RCA -----
with tab_rca:
    st.write(T("Root-cause analysis & predictive maintenance for one asset."))
    # Suggest equipment tags discovered in the graph.
    tags = [n for n, d in brain.graph.g.nodes(data=True)
            if d.get("type") == "Equipment"]
    tag = st.text_input(T("Equipment tag"), value=(tags[0] if tags else "P-7"))
    if st.button(T("Run RCA"), key="rca"):
        with ai_guard(), st.spinner("Analyzing equipment history..."):
            rep = analyze_equipment(brain, tag, language=LANG)
        st.markdown(rep.report_md)
        _show_sources(rep.sources)

# ----- Tab 4: Compliance -----
with tab_comp:
    st.write(T("Detect regulatory compliance gaps and generate audit evidence."))
    focus = st.text_input(T("Optional focus (e.g. 'lockout', 'P-7')"), value="")
    if st.button(T("Run compliance audit"), key="comp"):
        with ai_guard(), st.spinner("Auditing against Factory Act / OISD / PESO..."):
            rep = audit(brain, focus, language=LANG)
        st.markdown(rep.report_md)
        _show_sources(rep.sources)

# ----- Tab 5: Lessons Learned -----
with tab_lessons:
    st.write(T("Mine incident & near-miss history for recurring systemic patterns."))
    if st.button(T("Mine lessons"), key="lessons"):
        with ai_guard(), st.spinner("Mining failure history..."):
            rep = mine_lessons(brain, language=LANG)
        st.markdown(rep.report_md)
        _show_sources(rep.sources)

# ----- Tab 6: Conflict detection -> auto-drafted fix (closed loop) -----
with tab_conflict:
    st.write(T("Scan documents for contradictions — then let the brain draft the corrected procedure for engineer approval. Detection → fix, closed loop."))
    if st.button(T("Scan for conflicts"), key="conflict"):
        with ai_guard(), st.spinner("Cross-checking documents..."):
            st.session_state["conflicts"] = find_conflicts(brain)

    for i, c in enumerate(st.session_state.get("conflicts", [])):
        color = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}.get(c.severity, "⚪")
        st.error(f"{color} **{c.entity}** — {c.severity} severity")
        st.write(c.summary)

        dkey = f"draft_{i}"
        if st.button(f"🛠️ Draft the corrected document for {c.entity}",
                     key=f"fixbtn_{i}"):
            with ai_guard(), st.spinner("Drafting the corrected procedure..."):
                draft_md, path = draft_fix(c.entity, c.summary, c.sources)
            st.session_state[dkey] = (draft_md, path)

        if dkey in st.session_state:
            draft_md, path = st.session_state[dkey]
            with st.expander("📄 Revised draft (pending approval)", expanded=True):
                st.markdown(draft_md)
            a1, a2 = st.columns(2)
            a1.download_button(T("⬇️ Download draft"), draft_md,
                               file_name=Path(path).name, key=f"dl_{i}")
            if a2.button(T("✅ Approve → publish to corpus"), key=f"appr_{i}"):
                dest = approve(path)
                del st.session_state[dkey]
                st.success(f"Approved and published: {dest}. "
                           f"Click **Rebuild index** to make it the live "
                           f"source of truth.")
        _show_sources(c.sources)

    if "conflicts" in st.session_state and not st.session_state["conflicts"]:
        st.success(T("No contradictions found across the documents."))

# ----- Tab 7: Tribal Knowledge -----
with tab_tribal:
    st.write(T("Capture a retiring engineer's undocumented knowledge before it walks out the door — and see which assets are at risk."))

    st.subheader(T("🗺️ Knowledge Risk Map"))
    risk = knowledge_risk_map(brain)
    if risk:
        st.table([{"Equipment": r.equipment, "Documents": r.doc_chunks,
                   "Risk": r.risk} for r in risk])
    else:
        st.info(T("No equipment in the graph yet."))

    st.divider()
    st.subheader(T("🎙️ Capture Interview"))
    asset = st.text_input(T("Asset to document"), value="P-7", key="tribal_asset")

    # Keep the interview transcript in session state.
    if "transcript" not in st.session_state:
        st.session_state.transcript = []
        st.session_state.current_q = None

    if st.button(T("Start / next question"), key="tribal_q"):
        with ai_guard(), st.spinner("Thinking of the next question..."):
            st.session_state.current_q = next_question(asset, st.session_state.transcript)

    if st.session_state.current_q:
        st.markdown(f"**Interviewer:** {st.session_state.current_q}")

        # Dynamic keys per answer -> each new question gets fresh, empty inputs.
        idx = len(st.session_state.transcript)
        ans = st.text_area(T("Type the expert's answer"), key=f"tribal_ans_{idx}")

        # The expert can SPEAK the answer instead of typing it.
        ans_voice = st.audio_input("🎤 Or speak the answer", key=f"tribal_voice_{idx}")
        voice_text = ""
        if ans_voice is not None:
            vsig = hash(ans_voice.getvalue())
            if st.session_state.get(f"tribal_vsig_{idx}") != vsig:
                with st.spinner("Transcribing the spoken answer..."):
                    st.session_state[f"tribal_vtext_{idx}"] = transcribe(
                        ans_voice.getvalue(), mime=ans_voice.type or "audio/wav")
                st.session_state[f"tribal_vsig_{idx}"] = vsig
            voice_text = st.session_state.get(f"tribal_vtext_{idx}", "")
            if voice_text:
                st.info(f"🎤 Heard: {voice_text}")

        # Typed text wins if present; otherwise use the transcribed speech.
        final_ans = ans.strip() or voice_text.strip()
        if st.button(T("Save answer"), key=f"tribal_save_{idx}") and final_ans:
            st.session_state.transcript.append(
                (st.session_state.current_q, final_ans))
            st.session_state.current_q = None
            st.rerun()

    if st.session_state.transcript:
        st.caption(f"{len(st.session_state.transcript)} answers captured")
        with st.expander("View transcript"):
            for q, a in st.session_state.transcript:
                st.markdown(f"**Q:** {q}\n\n**A:** {a}")
        if st.button(T("💾 Save as knowledge document"), key="tribal_done"):
            with ai_guard(), st.spinner("Writing structured knowledge document..."):
                path = save_knowledge(asset, st.session_state.transcript)
            st.success(f"Saved to {path}. Click *Rebuild index* (sidebar) to add "
                       f"it to the knowledge base.")
            st.session_state.transcript = []

# ----- Tab: Documents library (see everything that's uploaded) -----
with tab_files:
    st.write(T("Every document in the knowledge base — status, size, and a quick preview. Upload more from the sidebar."))
    from src.ingest import _LOADERS as _EXTS
    import datetime as _dt

    _files = sorted([p for p in Path(DOCS_DIR).glob("*")
                     if p.is_file() and p.suffix.lower() in _EXTS])
    _chunk_count = {}
    for _c in brain.store.chunks:
        _chunk_count[_c.source_file] = _chunk_count.get(_c.source_file, 0) + 1

    _icon = {".pdf": "📕", ".docx": "📝", ".xlsx": "📊", ".xlsm": "📊",
             ".csv": "📊", ".tsv": "📊", ".txt": "📄", ".md": "📄",
             ".eml": "📧", ".png": "🖼️", ".jpg": "🖼️", ".jpeg": "🖼️",
             ".webp": "🖼️"}
    st.dataframe(
        [{"File": f"{_icon.get(p.suffix.lower(), '📄')} {p.name}",
          "Type": p.suffix.lstrip('.').upper(),
          "Size (KB)": round(p.stat().st_size / 1024, 1),
          "Modified": _dt.datetime.fromtimestamp(p.stat().st_mtime)
                        .strftime("%Y-%m-%d %H:%M"),
          "Indexed": (f"✅ {_chunk_count[p.name]} chunks"
                      if p.name in _chunk_count else "🆕 rebuild to index")}
         for p in _files],
        use_container_width=True, hide_index=True)
    st.caption(f"{len(_files)} files · "
               f"{sum(p.stat().st_size for p in _files)/1_048_576:.1f} MB total · "
               f"{len(brain.store.chunks)} searchable chunks")

    _pick = st.selectbox(T("🔍 Preview / download a file"),
                         [p.name for p in _files])
    if _pick:
        _sel = Path(DOCS_DIR) / _pick
        st.download_button(f"⬇️ Download {_pick}", _sel.read_bytes(),
                           file_name=_pick, key="doc_dl")
        _ext = _sel.suffix.lower()
        if _ext in (".png", ".jpg", ".jpeg", ".webp"):
            st.image(str(_sel), caption=_pick)
        elif _ext in (".txt", ".md", ".eml", ".csv", ".tsv"):
            st.code(_sel.read_text(encoding="utf-8", errors="ignore")[:3000],
                    language=None)
        elif _ext == ".pdf":
            import fitz as _fitz
            _d = _fitz.open(_sel)
            _np = len(_d)
            _pg = st.number_input(f"Page (1–{_np})", min_value=1, max_value=_np,
                                  value=1, step=1, key="pdf_page") if _np > 1 else 1
            st.caption(f"{_np} page(s) — showing page {_pg}:")
            st.code(_d[int(_pg) - 1].get_text()[:3000] or "(no extractable text "
                    "on this page — likely a scanned image)", language=None)
            _d.close()
        else:
            st.caption(T("No inline preview for this format — download to view."))
