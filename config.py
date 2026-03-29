"""
config.py — Central configuration for the ET AI-Native News Platform.
All hardware-aware settings are tuned for 16 GB RAM / CPU-only execution.
"""

import os
from pathlib import Path

# ── Directory Layout ────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
LANCE_DIR    = DATA_DIR / "lancedb"
GRAPH_DIR    = DATA_DIR / "graphs"
CACHE_DIR    = DATA_DIR / "cache"

for d in [DATA_DIR, LANCE_DIR, GRAPH_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM / Ollama ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2:3b")   # q4_k_m via Ollama
# CPU inference at 10-15 tok/s: 400 tokens ≈ 27-40s; 300s gives generous
# headroom for prompt processing + generation on a 16 GB / no-VRAM machine.
OLLAMA_TIMEOUT     = int(os.getenv("OLLAMA_TIMEOUT", "300"))
LLM_TEMPERATURE    = 0.3          # low temp = factual, grounded output
# 400 tokens ≈ 26-40s on CPU — keeps every LLM call well inside the timeout.
# Briefings are still substantive; we just cut padding and filler.
LLM_MAX_TOKENS     = 400
# Keep context under 2 k chars to avoid RAM thrashing and long prompt eval times.
LLM_CTX_WINDOW     = 4096

# ── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"   # 384-dim, ~90 MB on disk
EMBEDDING_DIM      = 384
EMBEDDING_BATCH    = 32            # small batch to keep RAM flat

# ── LanceDB ─────────────────────────────────────────────────────────────────
LANCEDB_TABLE      = "et_articles"
LANCEDB_IVF_NLIST  = 64            # IVF partitions — sensible for <500 k docs
LANCEDB_TOP_K_FIRST_PASS  = 50     # candidate pool for My-ET re-ranking
# Reduced from 8 → 4 to halve the number of sequential MAP LLM calls.
# 4 articles still produces a comprehensive briefing; 8 was causing timeouts.
LANCEDB_TOP_K_NAVIGATOR   = 4
LANCEDB_TOP_K_STORY       = 20     # recent articles scanned for arc clustering

# ── Ingestion ────────────────────────────────────────────────────────────────
RSS_POLL_INTERVAL_MINUTES  = 15
MAX_ARTICLES_PER_FEED      = 20    # cap per poll cycle to stay under RAM budget
ARTICLE_CHUNK_SIZE         = 800   # chars for Parent-Child splitting
ARTICLE_CHUNK_OVERLAP      = 150

# ET RSS feed catalogue
ET_RSS_FEEDS = {
    "Markets":         "https://economictimes.indiatimes.com/markets/rss.cms",
    "Tech":            "https://economictimes.indiatimes.com/tech/rss.cms",
    "Startups":        "https://economictimes.indiatimes.com/tech/startups/rss.cms",
    "Economy":         "https://economictimes.indiatimes.com/news/economy/rss.cms",
    "Finance":         "https://economictimes.indiatimes.com/wealth/rss.cms",
    "Mutual Funds":    "https://economictimes.indiatimes.com/mf/rss.cms",
    "Politics":        "https://economictimes.indiatimes.com/news/politics-and-nation/rss.cms",
    "Healthcare":      "https://economictimes.indiatimes.com/industry/healthcare/rss.cms",
    "Auto":            "https://economictimes.indiatimes.com/industry/auto/rss.cms",
    "Energy":          "https://economictimes.indiatimes.com/industry/energy/rss.cms",
}

# ── Entity Extraction (GLiNER) ───────────────────────────────────────────────
GLINER_MODEL       = "urchade/gliner_medium-v2.1"  # CPU-friendly size
NER_ENTITY_TYPES   = [
    "organization", "person", "location",
    "product", "event", "financial_instrument",
    "regulation", "government_body",
]
NER_THRESHOLD      = 0.45          # confidence floor for entity acceptance

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
ARC_ASSOCIATION_THRESHOLD = 0.45    # composite score to cluster article into arc
ARC_TEMPORAL_WEIGHT       = 0.25
ARC_SEMANTIC_WEIGHT       = 0.50
ARC_ENTITY_WEIGHT         = 0.25
ARC_MAX_AGE_DAYS          = 90      # arcs older than this are archived

# ── Sentiment ────────────────────────────────────────────────────────────────
SENTIMENT_WINDOW          = 5       # rolling window for arc sentiment tracking

# ── Memory Management Guards ─────────────────────────────────────────────────
# These prevent OOM on 16 GB RAM.
# Reduced from 3500 → 2000 to keep prompt eval fast on CPU.
MAX_CONTEXT_TOKENS        = 2000
# Reduced from 3 → 2: each MAP call is now cheaper; 2 at a time keeps us well
# under the per-request context budget while still processing all articles.
MAP_REDUCE_CHUNK_ARTICLES = 2
EMBEDDING_CACHE_SIZE      = 1000    # LRU cache for repeated embed calls
