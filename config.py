"""
config.py — Central configuration for the ET AI-Native News Platform.
UPGRADED: Groq API support, higher article limits, RAM optimization.
"""

import os
from pathlib import Path

# ── Directory Layout ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
LANCE_DIR  = DATA_DIR / "lancedb"
GRAPH_DIR  = DATA_DIR / "graphs"
CACHE_DIR  = DATA_DIR / "cache"

for d in [DATA_DIR, LANCE_DIR, GRAPH_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Groq (PRIMARY — 1,200+ tok/s, free tier) ────────────────────────────────
# Get a free key at https://console.groq.com  (takes 30 seconds)
# Set:  export GROQ_API_KEY="gsk_..."  or add to a .env file
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")   # ~1,200 tok/s
GROQ_MODEL_ALT = "mixtral-8x7b-32768"                              # backup: richer output
USE_GROQ       = bool(GROQ_API_KEY)

# ── LLM / Ollama (FALLBACK — CPU-only) ──────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:3b")
# Generous timeout for CPU-only fallback path
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# Shared LLM knobs (apply to both backends)
LLM_TEMPERATURE = 0.3
# Groq can handle 2× tokens in the same clock time — we use the headroom.
LLM_MAX_TOKENS  = 800   # was 400
LLM_CTX_WINDOW  = 8192  # Groq supports large contexts; Ollama: 4096 fallback

# ── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, ~90 MB, stays on CPU
EMBEDDING_DIM   = 384
EMBEDDING_BATCH = 64   # doubled from 32 — safe with 14 GB RAM

# ── LanceDB ─────────────────────────────────────────────────────────────────
LANCEDB_TABLE              = "et_articles"
LANCEDB_IVF_NLIST          = 64
LANCEDB_TOP_K_FIRST_PASS   = 80    # was 50  → richer candidate pool
LANCEDB_TOP_K_NAVIGATOR    = 8     # was 4   → more sources = better briefing
LANCEDB_TOP_K_STORY        = 30    # was 20

# ── Ingestion ────────────────────────────────────────────────────────────────
RSS_POLL_INTERVAL_MINUTES = 15
MAX_ARTICLES_PER_FEED     = 50     # was 20 — fetch more per poll
ARTICLE_CHUNK_SIZE        = 800
ARTICLE_CHUNK_OVERLAP     = 150

# Date-based smart ingestion (new)
# "Fetch Live" only ingests articles published TODAY.
# LanceDB retains all historical articles permanently.
INGEST_TODAY_ONLY = True   # set False to re-fetch all available articles

ET_RSS_FEEDS = {
    "Markets":      "https://economictimes.indiatimes.com/markets/rss.cms",
    "Tech":         "https://economictimes.indiatimes.com/tech/rss.cms",
    "Startups":     "https://economictimes.indiatimes.com/tech/startups/rss.cms",
    "Economy":      "https://economictimes.indiatimes.com/news/economy/rss.cms",
    "Finance":      "https://economictimes.indiatimes.com/wealth/rss.cms",
    "Mutual Funds": "https://economictimes.indiatimes.com/mf/rss.cms",
    "Politics":     "https://economictimes.indiatimes.com/news/politics-and-nation/rss.cms",
    "Healthcare":   "https://economictimes.indiatimes.com/industry/healthcare/rss.cms",
    "Auto":         "https://economictimes.indiatimes.com/industry/auto/rss.cms",
    "Energy":       "https://economictimes.indiatimes.com/industry/energy/rss.cms",
}

# ── Entity Extraction (GLiNER / SpaCy) ──────────────────────────────────────
GLINER_MODEL     = "urchade/gliner_medium-v2.1"
NER_ENTITY_TYPES = [
    "organization", "person", "location",
    "product", "event", "financial_instrument",
    "regulation", "government_body",
]
NER_THRESHOLD = 0.45
# Speed optimisation: run NER only on first N chars (covers 95% of entities)
NER_MAX_CHARS = 1200   # was unbounded

# ── Persona Definitions ──────────────────────────────────────────────────────
PERSONAS = {
    "Mutual Fund Investor": {
        "icon": "📈",
        "description": "Portfolio-relevant stories, NAV movements, fund performance, macro policy",
        "keywords": ["mutual fund", "NAV", "SIP", "equity", "debt fund", "SEBI", "portfolio",
                     "returns", "asset management", "AUM", "market cap", "dividend"],
        "boost_entities": ["SEBI", "AMC", "NSE", "BSE", "RBI"],
        "summary_style": "data-driven with numbers and percentages",
    },
    "Startup Founder": {
        "icon": "🚀",
        "description": "Funding rounds, competitor intelligence, regulatory impact, VC trends",
        "keywords": ["funding", "startup", "venture capital", "Series A", "unicorn",
                     "pivot", "runway", "valuation", "acquisition", "IPO", "angel"],
        "boost_entities": ["DPIIT", "Nasscom", "TiE", "Sequoia", "Accel"],
        "summary_style": "strategic and action-oriented",
    },
    "Student / Learner": {
        "icon": "📚",
        "description": "Explainer-first content, simplified jargon, educational context",
        "keywords": ["economy", "GDP", "inflation", "RBI", "budget", "policy",
                     "interest rate", "trade", "employment", "growth"],
        "boost_entities": [],
        "summary_style": "simple, jargon-free with analogies",
    },
    "Executive / CXO": {
        "icon": "💼",
        "description": "Strategic intelligence, industry disruption, regulatory landscape",
        "keywords": ["strategy", "M&A", "disruption", "leadership", "board",
                     "quarterly results", "ESG", "supply chain", "enterprise"],
        "boost_entities": [],
        "summary_style": "executive-level with strategic implications",
    },
}

# ── Story Arc Tracker ────────────────────────────────────────────────────────
ARC_ASSOCIATION_THRESHOLD = 0.45
ARC_TEMPORAL_WEIGHT       = 0.25
ARC_SEMANTIC_WEIGHT       = 0.50
ARC_ENTITY_WEIGHT         = 0.25
ARC_MAX_AGE_DAYS          = 90

# ── Sentiment ────────────────────────────────────────────────────────────────
SENTIMENT_WINDOW = 5

# ── Memory / Context Guards ──────────────────────────────────────────────────
# Groq can handle much larger contexts; Ollama fallback uses tighter cap.
MAX_CONTEXT_TOKENS        = 4096 if USE_GROQ else 2000
MAP_REDUCE_CHUNK_ARTICLES = 8    # was 2 — parallel Groq calls make this safe
EMBEDDING_CACHE_SIZE      = 2000  # was 1000

# ── Parallel Processing ──────────────────────────────────────────────────────
MAP_MAX_WORKERS = 6   # ThreadPoolExecutor workers for parallel MAP calls
