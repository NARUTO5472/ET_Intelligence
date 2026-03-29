"""
app.py — ET AI-Native News Platform  (UPGRADED)
Main Streamlit application entry point.

Upgrades:
  • Groq API status indicator in masthead (shows speed tier)
  • "Fetch Live" → "Fetch Today's News" (date-smart ingestion)
  • Shows article count by ingestion date
  • All other UI identical to original
"""

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import PERSONAS, ET_RSS_FEEDS
from utils.helpers import format_relative_time, vertical_color, sentiment_to_emoji
from utils.ingestion_orchestrator import ingest_articles
from vector_store.lancedb_manager import article_count
from llm.ollama_client import check_ollama_alive, check_groq_alive

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ET Intelligence Platform",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
    --bg-primary:    #0A0A0F;
    --bg-surface:    #111118;
    --bg-elevated:   #16161F;
    --bg-card:       #1A1A25;
    --border:        #2A2A3A;
    --border-accent: #FF6B00;
    --text-primary:  #F0EDE8;
    --text-secondary:#9B96A8;
    --text-muted:    #5C5870;
    --accent:        #FF6B00;
    --accent-amber:  #F59E0B;
    --accent-blue:   #4F9CF9;
    --accent-green:  #34D399;
    --accent-red:    #F87171;
    --accent-purple: #A78BFA;
    --positive:      #34D399;
    --negative:      #F87171;
    --neutral:       #9B96A8;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
.stApp { background-color: var(--bg-primary) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.masthead { background: linear-gradient(135deg, #0D0D18 0%, #110A05 50%, #0D0A18 100%); border-bottom: 2px solid var(--accent); padding: 24px 32px 20px; margin: -1rem -1rem 2rem; display: flex; align-items: center; justify-content: space-between; }
.masthead-title { font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 900; color: var(--text-primary); letter-spacing: -1px; line-height: 1; }
.masthead-title span { color: var(--accent); }
.masthead-subtitle { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.75rem; font-weight: 300; color: var(--text-muted); letter-spacing: 3px; text-transform: uppercase; margin-top: 4px; }
.masthead-status { display: flex; align-items: center; gap: 16px; }
.status-pill { padding: 4px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.5px; font-family: 'IBM Plex Mono', monospace; }
.status-live  { background: rgba(52,211,153,0.12); color: var(--positive); border: 1px solid rgba(52,211,153,0.3); }
.status-warn  { background: rgba(245,158,11,0.12);  color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.3); }
.status-error { background: rgba(248,113,113,0.12); color: var(--negative); border: 1px solid rgba(248,113,113,0.3); }
.status-groq  { background: rgba(79,156,249,0.12); color: var(--accent-blue); border: 1px solid rgba(79,156,249,0.3); }
.css-1d391kg, [data-testid="stSidebar"] { background: var(--bg-surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] .stRadio > label { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.8rem; color: var(--text-secondary); letter-spacing: 1.5px; text-transform: uppercase; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg-surface); border-bottom: 1px solid var(--border); gap: 0; padding: 0; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; font-size: 0.82rem; letter-spacing: 1px; text-transform: uppercase; color: var(--text-muted); padding: 14px 24px; border-bottom: 3px solid transparent; transition: all 0.2s ease; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; background: transparent !important; }
.news-card { background: var(--bg-card); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 8px; padding: 20px 22px; margin-bottom: 14px; transition: border-color 0.2s, transform 0.15s; cursor: pointer; }
.news-card:hover { border-left-color: var(--accent-amber); transform: translateX(2px); }
.news-card-title { font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; color: var(--text-primary); line-height: 1.35; margin-bottom: 8px; }
.news-card-meta { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
.vertical-badge { font-size: 0.65rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; padding: 2px 8px; border-radius: 3px; font-family: 'IBM Plex Mono', monospace; }
.news-card-lead { font-size: 0.88rem; line-height: 1.6; color: var(--text-secondary); }
.score-badge { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: var(--text-muted); padding: 1px 6px; background: var(--bg-elevated); border-radius: 3px; }
.briefing-box { background: var(--bg-elevated); border: 1px solid var(--border); border-top: 2px solid var(--accent-blue); border-radius: 10px; padding: 28px 32px; font-size: 0.92rem; line-height: 1.8; color: var(--text-primary); font-family: 'IBM Plex Sans', sans-serif; }
.briefing-box h1, .briefing-box h2, .briefing-box h3 { font-family: 'Playfair Display', serif; color: var(--text-primary); }
.briefing-box strong { color: var(--accent-amber); }
.briefing-box em { color: var(--text-secondary); }
.briefing-box code { font-family: 'IBM Plex Mono', monospace; background: var(--bg-card); padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }
.source-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 20px; font-size: 0.72rem; margin: 3px; font-family: 'IBM Plex Mono', monospace; color: var(--text-secondary); }
.entity-tag { display: inline-block; padding: 2px 8px; background: rgba(79,156,249,0.1); border: 1px solid rgba(79,156,249,0.25); border-radius: 3px; font-size: 0.68rem; color: var(--accent-blue); margin: 2px; font-family: 'IBM Plex Mono', monospace; }
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.kpi-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 16px 22px; min-width: 140px; flex: 1; }
.kpi-value { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 900; color: var(--accent); line-height: 1; }
.kpi-label { font-size: 0.7rem; font-weight: 500; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text-muted); margin-top: 4px; font-family: 'IBM Plex Sans', sans-serif; }
.timeline-event { position: relative; padding-left: 20px; padding-bottom: 20px; border-left: 2px solid var(--border); margin-left: 8px; }
.timeline-event::before { content: ''; position: absolute; left: -5px; top: 6px; width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }
.timeline-date { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--text-muted); margin-bottom: 2px; }
.timeline-title { font-size: 0.88rem; color: var(--text-primary); font-weight: 500; line-height: 1.3; }
.stButton > button { background: var(--accent) !important; color: white !important; border: none !important; font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; font-size: 0.82rem !important; letter-spacing: 0.5px !important; border-radius: 6px !important; padding: 8px 20px !important; transition: opacity 0.2s !important; }
.stButton > button:hover { opacity: 0.88 !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background: var(--bg-elevated) !important; border: 1px solid var(--border) !important; color: var(--text-primary) !important; border-radius: 6px !important; font-family: 'IBM Plex Sans', sans-serif !important; }
.stSelectbox > div > div, .stMultiSelect > div > div { background: var(--bg-elevated) !important; border-color: var(--border) !important; color: var(--text-primary) !important; }
.section-header { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px; }
.section-subheader { font-size: 0.8rem; color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase; font-family: 'IBM Plex Sans', sans-serif; margin-bottom: 20px; }
.divider { height: 1px; background: var(--border); margin: 24px 0; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "persona":          "Mutual Fund Investor",
        "user_profile":     None,
        "nav_conversation": None,
        "nav_history":      [],
        "nav_query":        "",
        "arc_result":       None,
        "arc_topic":        "",
        "ingestion_done":   False,
        "follow_up_query":  "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.user_profile is None:
        from modules.my_et import UserProfileManager
        st.session_state.user_profile = UserProfileManager()
    if st.session_state.nav_conversation is None:
        from modules.news_navigator import ConversationManager
        st.session_state.nav_conversation = ConversationManager()


init_session()


# ── Masthead ──────────────────────────────────────────────────────────────────
groq_ok   = check_groq_alive()
ollama_ok = check_ollama_alive()
db_count  = article_count()

# LLM status: Groq beats Ollama; show whichever is active
if groq_ok:
    llm_status_html = '<span class="status-pill status-groq">⚡ GROQ · 1200 tok/s</span>'
elif ollama_ok:
    llm_status_html = '<span class="status-pill status-live">● OLLAMA ONLINE</span>'
else:
    llm_status_html = '<span class="status-pill status-warn">⚠ NO LLM</span>'

db_status_html = f'<span class="status-pill status-live">DB: {db_count} articles</span>'

st.markdown(f"""
<div class="masthead">
  <div>
    <div class="masthead-title"><span>ET</span> Intelligence</div>
    <div class="masthead-subtitle">AI-Native News Platform · Economic Times</div>
  </div>
  <div class="masthead-status">
    {llm_status_html}
    {db_status_html}
  </div>
</div>
""", unsafe_allow_html=True)

if not groq_ok and not ollama_ok:
    st.warning(
        "⚠️ **No LLM available.** Set `GROQ_API_KEY` (recommended, free & instant) "
        "or run `ollama serve` for local inference."
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Platform Settings")
    st.markdown("---")

    st.markdown("**YOUR PERSONA**")
    persona_options = list(PERSONAS.keys())
    persona_labels  = [f"{PERSONAS[p]['icon']} {p}" for p in persona_options]
    selected_idx    = st.radio(
        "Select persona",
        range(len(persona_options)),
        format_func=lambda i: persona_labels[i],
        index=persona_options.index(st.session_state.persona),
        label_visibility="collapsed",
    )
    st.session_state.persona = persona_options[selected_idx]
    persona_data = PERSONAS[st.session_state.persona]
    st.caption(persona_data["description"])
    st.markdown("---")

    st.markdown("**📥 DATA INGESTION**")

    # Speed tip based on current LLM
    if groq_ok:
        st.caption("⚡ Groq active — briefings in ~15 seconds")
    elif ollama_ok:
        st.caption("🐢 Ollama CPU mode — briefings take ~4-8 min")
    else:
        st.caption("💡 Set GROQ_API_KEY for instant briefings")

    col1, col2 = st.columns(2)
    with col1:
        # "Fetch Today's News" — only ingests today's RSS articles
        if st.button("📅 Today's News", use_container_width=True):
            with st.spinner("Fetching today's ET articles…"):
                progress_bar = st.progress(0)
                def _progress(step, pct):
                    progress_bar.progress(pct)
                result = ingest_articles(
                    use_mock=False, today_only=True,
                    progress_callback=_progress,
                )
            if result["upserted"] > 0:
                st.success(f"✅ +{result['upserted']} new articles")
            else:
                st.info("ℹ️ Already up to date for today")
            st.session_state.ingestion_done = True
            st.rerun()

    with col2:
        if st.button("🎭 Load Demo", use_container_width=True):
            with st.spinner("Loading 20 demo articles…"):
                result = ingest_articles(use_mock=True, force_refresh=True)
            st.success(f"✅ {result['upserted']} demo articles loaded")
            st.session_state.ingestion_done = True
            st.rerun()

    st.markdown(f"**{article_count()} articles** in knowledge base")
    st.markdown("---")

    st.markdown("**🗂 VERTICAL FILTERS**")
    available_verticals = list(ET_RSS_FEEDS.keys())
    selected_verticals  = st.multiselect(
        "Focus on verticals",
        options=available_verticals,
        default=[],
        label_visibility="collapsed",
        placeholder="All verticals (default)",
    )
    verticals_filter = selected_verticals if selected_verticals else None

    st.markdown("---")
    st.markdown("**📅 DATE RANGE**")
    days_back = st.slider("Days back", min_value=1, max_value=90, value=7)

    st.markdown("---")
    if groq_ok:
        st.caption("**ET AI Intelligence Platform**\nGroq · llama-3.1-8b-instant\n\nBuilt for ET GenAI Hackathon 2026")
    else:
        st.caption("**ET AI Intelligence Platform**\nOllama · llama3.2:3b CPU\n\nBuilt for ET GenAI Hackathon 2026")


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📰  MY ET — Personalized Newsroom",
    "🔬  News Navigator — Intelligence Briefings",
    "🕸️  Story Arc Tracker",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: MY ET
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    from modules.my_et import (
        get_personalized_feed, format_article_card,
        generate_eli5_summary, generate_persona_summary,
    )
    persona_key  = st.session_state.persona
    persona_info = PERSONAS[persona_key]

    st.markdown(f"""
    <div class="section-header">{persona_info['icon']} Your Personalized Newsroom</div>
    <div class="section-subheader">Curated for · {persona_key} · {persona_info['description']}</div>
    """, unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        refresh_feed = st.button("🔄 Refresh Feed", key="refresh_myET")
    with ctrl_col2:
        feed_size = st.selectbox("Articles", [5, 10, 15, 20], index=1, label_visibility="collapsed")
    with ctrl_col3:
        show_ai_summaries = st.toggle("AI Summaries", value=False)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if article_count() == 0:
        st.info(
            "📭 **Knowledge base is empty.**\n\n"
            "Click **Load Demo** in the sidebar to load 20 high-quality sample articles, "
            "or **Today's News** to pull live ET articles from today."
        )
    else:
        with st.spinner(f"Generating your {persona_key} feed…"):
            feed = get_personalized_feed(
                persona_key=persona_key,
                verticals=verticals_filter,
                top_n=feed_size,
                days_back=days_back,
            )

        if not feed:
            st.warning("No articles matched your filters. Try broadening vertical or date range.")
        else:
            avg_sentiment = sum(a.get("sentiment_compound", 0) for a in feed) / max(len(feed), 1)
            market_tone   = "📈 Bullish" if avg_sentiment > 0.1 else ("📉 Bearish" if avg_sentiment < -0.1 else "➡️ Neutral")
            top_entities  = []
            for a in feed:
                top_entities.extend(a.get("entities", [])[:2])
            unique_entities = list(dict.fromkeys(top_entities))[:5]

            st.markdown(f"""
            <div class="kpi-row">
              <div class="kpi-card"><div class="kpi-value">{len(feed)}</div><div class="kpi-label">Stories curated</div></div>
              <div class="kpi-card"><div class="kpi-value">{market_tone}</div><div class="kpi-label">Market tone</div></div>
              <div class="kpi-card"><div class="kpi-value">{len(set(a.get('vertical','') for a in feed))}</div><div class="kpi-label">Verticals covered</div></div>
            </div>
            """, unsafe_allow_html=True)

            if unique_entities:
                entity_html = " ".join(f'<span class="entity-tag">{e}</span>' for e in unique_entities)
                st.markdown(f"**Key entities in your feed:** {entity_html}", unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            for article in feed:
                card    = format_article_card(article, persona_key)
                vc      = vertical_color(card["vertical"])
                time_ago = format_relative_time(card["published_at"])

                with st.container():
                    st.markdown(f"""
                    <div class="news-card">
                      <div class="news-card-meta">
                        <span class="vertical-badge" style="background:{'rgba'+str(tuple(int(vc.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (25,)).replace('(','(').replace(', ',',')};color:{vc};border:1px solid {'rgba'+str(tuple(int(vc.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (60,)).replace('(','(').replace(', ',',')};">
                          {card["vertical"].upper()}
                        </span>
                        <span style="color:var(--text-muted);font-size:0.72rem;font-family:'IBM Plex Mono',monospace;">{time_ago}</span>
                        <span style="color:{('#34D399' if card['sentiment']=='positive' else '#F87171' if card['sentiment']=='negative' else '#9B96A8')};font-size:0.8rem;">{card['sentiment_icon']} {card['sentiment'].title()}</span>
                        <span class="score-badge">Relevance: {card['score']:.2f}</span>
                      </div>
                      <div class="news-card-title">{card['title']}</div>
                      <div class="news-card-lead">{card['lead']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander(f"🤖 AI Summary for {persona_key}", expanded=False):
                    if not groq_ok and not ollama_ok:
                        st.warning("No LLM available. Set GROQ_API_KEY or start Ollama.")
                    else:
                        with st.spinner("Generating summary…"):
                            if persona_key == "Student / Learner":
                                summary = generate_eli5_summary(article)
                                st.info(f"**ELI5:** {summary}")
                            else:
                                summary = generate_persona_summary(article, persona_key)
                                st.markdown(f"> {summary}")

                entity_tags = " ".join(f'<span class="entity-tag">{e}</span>' for e in card["entities"][:5])
                st.markdown(
                    f"{entity_tags} &nbsp; <a href='{card['url']}' target='_blank' style='color:var(--accent);font-size:0.75rem;font-family:IBM Plex Mono,monospace;text-decoration:none;'>Read full article ↗</a>",
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: NEWS NAVIGATOR
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    from modules.news_navigator import run_navigator_briefing, ConversationManager

    st.markdown("""
    <div class="section-header">🔬 News Navigator</div>
    <div class="section-subheader">Multi-document synthesis · Parallel intelligence briefings · Follow-up Q&A</div>
    """, unsafe_allow_html=True)

    with st.container():
        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            default_q = st.session_state.get("follow_up_query", "")
            nav_query = st.text_area(
                "Intelligence Query",
                value=default_q,
                placeholder="e.g. What are the key implications of the Union Budget 2026 on equity markets?",
                height=90,
                label_visibility="collapsed",
                key="nav_query_input",
            )
            st.session_state.follow_up_query = ""
        with q_col2:
            nav_persona = st.selectbox(
                "Output Style",
                options=list(PERSONAS.keys()),
                format_func=lambda k: f"{PERSONAS[k]['icon']} {k}",
                index=list(PERSONAS.keys()).index(st.session_state.persona),
                label_visibility="collapsed",
            )
            nav_vertical = st.multiselect(
                "Verticals", options=list(ET_RSS_FEEDS.keys()),
                default=[], placeholder="All", label_visibility="collapsed",
            )

    act_col1, act_col2, act_col3 = st.columns([2, 1, 1])
    with act_col1:
        run_briefing = st.button("🚀 Generate Intelligence Briefing", type="primary", key="run_nav")
    with act_col2:
        clear_history = st.button("🗑 Clear Conversation", key="clear_nav")
    with act_col3:
        st.caption(f"💬 {len(st.session_state.nav_history)//2} turns")

    if clear_history:
        st.session_state.nav_history = []
        st.session_state.nav_conversation.clear()
        st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.session_state.nav_history:
        for turn in st.session_state.nav_history:
            if turn["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(turn["content"])
            else:
                with st.chat_message("assistant", avatar="📰"):
                    st.markdown(f'<div class="briefing-box">{turn["content"]}</div>', unsafe_allow_html=True)
                    if "sources" in turn:
                        with st.expander("📚 Source Articles", expanded=False):
                            for src in turn["sources"]:
                                sent_color = "#34D399" if src["sentiment"] == "positive" else ("#F87171" if src["sentiment"] == "negative" else "#9B96A8")
                                st.markdown(
                                    f'<span class="source-chip">{src["label"]}</span> '
                                    f'<a href="{src["url"]}" target="_blank" style="color:var(--accent-blue);font-size:0.82rem;">{src["title"][:70]}</a> '
                                    f'<span style="color:{sent_color};font-size:0.75rem;margin-left:8px;">{src["sentiment"]}</span>',
                                    unsafe_allow_html=True,
                                )
                    if "follow_ups" in turn and turn["follow_ups"]:
                        st.markdown("**Suggested follow-up questions:**")
                        for fq in turn["follow_ups"]:
                            if st.button(f"↩ {fq}", key=f"fq_{hash(fq)}", use_container_width=False):
                                st.session_state.follow_up_query = fq
                                st.rerun()

    if run_briefing and nav_query.strip():
        if article_count() == 0:
            st.error("Knowledge base is empty. Load demo data or fetch today's news first.")
        else:
            st.session_state.nav_history.append({"role": "user", "content": nav_query})
            st.session_state.nav_conversation.add_turn("user", nav_query)

            with st.chat_message("assistant", avatar="📰"):
                with st.spinner("🔍 Retrieving articles and synthesising briefing (parallel MAP)…"):
                    try:
                        result = run_navigator_briefing(
                            query=nav_query,
                            persona_key=nav_persona,
                            verticals=nav_vertical if nav_vertical else None,
                            days_back=days_back,
                            conversation=st.session_state.nav_conversation,
                        )
                    except Exception as _nav_err:
                        logger.error("Navigator briefing failed: %s", _nav_err)
                        result = {
                            "briefing": f"⚠️ **Briefing generation failed.**\n\nError: `{_nav_err}`",
                            "sources": [], "article_count": 0,
                            "follow_up_questions": [], "llm_available": False,
                        }

                briefing       = result["briefing"]
                sources        = result.get("sources", [])
                follow_ups     = result.get("follow_up_questions", [])
                art_count_used = result.get("article_count", 0)
                llm_available  = result.get("llm_available", True)

                if not llm_available:
                    st.warning("⚠️ LLM unavailable — showing extracted summaries. Set GROQ_API_KEY for full synthesis.")

                st.info(f"📄 Synthesised from **{art_count_used}** relevant articles")
                st.markdown(f'<div class="briefing-box">{briefing}</div>', unsafe_allow_html=True)

                if sources:
                    with st.expander(f"📚 {len(sources)} Source Articles", expanded=False):
                        for src in sources:
                            st.markdown(
                                f'<span class="source-chip">{src["label"]}</span> '
                                f'<a href="{src["url"]}" target="_blank" style="color:var(--accent-blue);font-size:0.82rem;">{src["title"][:70]}</a>',
                                unsafe_allow_html=True,
                            )

                if follow_ups:
                    st.markdown("---")
                    st.markdown("**💡 Suggested follow-up questions:**")
                    cols = st.columns(min(len(follow_ups), 3))
                    for i, fq in enumerate(follow_ups[:3]):
                        with cols[i]:
                            if st.button(f"↩ {fq[:60]}…" if len(fq) > 60 else f"↩ {fq}", key=f"fq2_{hash(fq)}", use_container_width=True):
                                st.session_state.follow_up_query = fq
                                st.rerun()

            st.session_state.nav_conversation.add_turn("assistant", briefing)
            history_entry = {"role": "assistant", "content": briefing, "sources": sources, "follow_ups": follow_ups}
            st.session_state.nav_history.append(history_entry)
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: STORY ARC TRACKER
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    from modules.story_arc import discover_story_arcs, get_graph_stats

    st.markdown("""
    <div class="section-header">🕸️ Story Arc Tracker</div>
    <div class="section-subheader">Knowledge graph · Narrative evolution · Sentiment trajectories · Contrarian perspectives</div>
    """, unsafe_allow_html=True)

    arc_col1, arc_col2, arc_col3 = st.columns([3, 1, 1])
    with arc_col1:
        arc_topic = st.text_input(
            "Track a story",
            value=st.session_state.arc_topic,
            placeholder="e.g.  RBI interest rates  |  Zepto IPO  |  Union Budget telecom",
            label_visibility="collapsed",
        )
    with arc_col2:
        arc_days = st.slider("Days back", 7, 90, 30, key="arc_days", label_visibility="collapsed")
    with arc_col3:
        run_arc = st.button("🕵️ Track Arc", type="primary", key="run_arc")

    try:
        gstats = get_graph_stats()
        st.caption(
            f"Knowledge Graph: **{gstats['entity_count']}** entities · **{gstats['article_count']}** article nodes · "
            f"**{gstats['total_edges']}** relationships"
        )
    except Exception:
        pass

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if run_arc and arc_topic.strip():
        st.session_state.arc_topic = arc_topic
        if article_count() == 0:
            st.error("Knowledge base is empty. Load demo data or fetch today's news first.")
        else:
            with st.spinner(f"Building story arc for '{arc_topic}'…"):
                try:
                    arc_result = discover_story_arcs(arc_topic, days_back=arc_days)
                except Exception as _arc_err:
                    logger.error("Story arc failed: %s", _arc_err)
                    arc_result = {"error": f"Story arc analysis failed: `{_arc_err}`"}
            st.session_state.arc_result = arc_result

    arc_result = st.session_state.arc_result
    if arc_result and not arc_result.get("error"):
        llm_ok = arc_result.get("llm_available", True)
        if not llm_ok:
            st.warning("⚠️ LLM unavailable — showing extracted data without AI narration.")
        st.success(
            f"🗺 Found **{arc_result['arc_count']} arc(s)** from **{arc_result['article_count']}** articles · "
            f"Dominant arc: **{arc_result['dominant_article_count']} articles**"
        )

        evo_col, right_col = st.columns([3, 2])

        with evo_col:
            st.markdown("#### 📖 Evolution Briefing")
            st.markdown(f'<div class="briefing-box">{arc_result["evolution_briefing"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 🔭 What to Watch Next")
            st.markdown(f'<div class="briefing-box">{arc_result["what_to_watch"]}</div>', unsafe_allow_html=True)

        with right_col:
            st.markdown("#### 👥 Key Players")
            players = arc_result.get("key_players", [])
            if players:
                max_score = max(p["score"] for p in players) or 1.0
                for p in players[:8]:
                    bar_pct = int((p["score"] / max_score) * 100)
                    st.markdown(
                        f"""<div style="margin-bottom:8px;">
                          <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                            <span style="font-size:0.82rem;color:var(--text-primary);font-weight:500;">{p['name']}</span>
                            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:var(--text-muted);">{p['mentions']} mentions</span>
                          </div>
                          <div style="height:4px;background:var(--border);border-radius:2px;">
                            <div style="height:4px;width:{bar_pct}%;background:var(--accent);border-radius:2px;"></div>
                          </div></div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 📊 Sentiment Trajectory")
            sentiment_traj = arc_result.get("sentiment_trajectory", [])
            if sentiment_traj:
                import plotly.graph_objects as go
                dates   = [t["published_at"][:10] for t in sentiment_traj]
                raw     = [t["sentiment"] for t in sentiment_traj]
                rolling = [t.get("rolling_avg", t["sentiment"]) for t in sentiment_traj]
                titles  = [t["title"][:40] for t in sentiment_traj]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=raw, mode="markers",
                    marker=dict(size=8, color=["#34D399" if s > 0.05 else "#F87171" if s < -0.05 else "#9B96A8" for s in raw]),
                    name="Article sentiment", text=titles, hovertemplate="%{text}<br>%{y:.2f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=dates, y=rolling, mode="lines",
                    line=dict(color="#FF6B00", width=2), name="Rolling average"))
                fig.add_hline(y=0, line_dash="dot", line_color="#2A2A3A")
                fig.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=30),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9B96A8", size=10),
                    xaxis=dict(gridcolor="#1A1A25", tickangle=30),
                    yaxis=dict(gridcolor="#1A1A25", range=[-1,1]), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        contrarians = arc_result.get("contrarian_articles", [])
        if contrarians:
            st.markdown("#### ⚔️ Contrarian Perspectives")
            for pair in contrarians[:3]:
                c1, c2 = st.columns(2)
                with c1:
                    s = pair["article_a"]["sentiment"]
                    colour = "#34D399" if s > 0.05 else "#F87171"
                    st.markdown(f'<div class="news-card" style="border-left-color:{colour};"><div style="font-size:0.7rem;color:{colour};margin-bottom:4px;">{"📈 Bullish" if s > 0.05 else "📉 Bearish"} · {s:.2f}</div><div class="news-card-title" style="font-size:0.9rem;">{pair["article_a"]["title"]}</div></div>', unsafe_allow_html=True)
                with c2:
                    s = pair["article_b"]["sentiment"]
                    colour = "#34D399" if s > 0.05 else "#F87171"
                    st.markdown(f'<div class="news-card" style="border-left-color:{colour};"><div style="font-size:0.7rem;color:{colour};margin-bottom:4px;">{"📈 Bullish" if s > 0.05 else "📉 Bearish"} · {s:.2f}</div><div class="news-card-title" style="font-size:0.9rem;">{pair["article_b"]["title"]}</div></div>', unsafe_allow_html=True)

        st.markdown("#### 🗓 Story Timeline")
        if sentiment_traj:
            for event in sentiment_traj:
                date = event["published_at"][:10] if event["published_at"] else "—"
                s    = event["sentiment"]
                icon = "📈" if s > 0.05 else ("📉" if s < -0.05 else "➡️")
                st.markdown(f'<div class="timeline-event"><div class="timeline-date">{date}</div><div class="timeline-title">{icon} {event["title"]}</div></div>', unsafe_allow_html=True)

    elif arc_result and arc_result.get("error"):
        st.warning(arc_result["error"])
    else:
        st.info(
            "🔍 Enter a topic above and click **Track Arc** to build its knowledge graph.\n\n"
            "**Example topics:** `RBI rate cut` · `Zepto IPO` · `Union Budget 2026` · `Tata Motors EV` · `SEBI regulations`"
        )
