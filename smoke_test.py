#!/usr/bin/env python3
"""
smoke_test.py — Offline verification of all platform modules.
Run this BEFORE launching the Streamlit app to catch import / config issues.
Does NOT require Ollama or internet access.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = "  ✅"
FAIL = "  ❌"
SKIP = "  ⚠️"

results = []


def test(name, fn):
    try:
        fn()
        results.append((name, True, None))
        print(f"{PASS} {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"{FAIL} {name}\n       → {e}")


print("\n" + "="*60)
print("  ET AI News Platform — Smoke Test")
print("="*60 + "\n")

# ── 1. Config ─────────────────────────────────────────────────────────────────
print("[1/8] Configuration")

def test_config():
    from config import (
        BASE_DIR, DATA_DIR, LANCE_DIR, GRAPH_DIR, CACHE_DIR,
        OLLAMA_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM,
        PERSONAS, ET_RSS_FEEDS,
    )
    assert EMBEDDING_DIM == 384
    assert len(PERSONAS) >= 4
    assert len(ET_RSS_FEEDS) >= 5
    assert DATA_DIR.exists()
    assert LANCE_DIR.exists()

test("Config loads and directories exist", test_config)


# ── 2. Ingestion — article processor ─────────────────────────────────────────
print("\n[2/8] Ingestion Pipeline")

def test_chunker():
    from ingestion.article_processor import chunk_text
    text = "This is sentence one. This is sentence two, which is a bit longer. " * 20
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c) > 20

test("Text chunking (Parent-Child)", test_chunker)


def test_sentiment():
    from ingestion.article_processor import compute_sentiment
    result = compute_sentiment("RBI cuts rates boosting markets and growth outlook.")
    assert "compound" in result
    assert result["compound"] >= -1.0
    assert result["compound"] <= 1.0

test("Sentiment analysis (VADER)", test_sentiment)


def test_readability():
    from ingestion.article_processor import compute_readability_score
    score = compute_readability_score("The cat sat on the mat. It was a sunny day.")
    assert 0 <= score <= 100

test("Readability scoring", test_readability)


def test_mock_articles():
    from ingestion.rss_fetcher import get_mock_articles
    articles = get_mock_articles()
    assert len(articles) == 8
    for a in articles:
        assert "id" in a
        assert "title" in a
        assert "full_text" in a
        assert len(a["full_text"]) > 100

test("Mock article generation (8 articles)", test_mock_articles)


# ── 3. Embeddings ─────────────────────────────────────────────────────────────
print("\n[3/8] Embedding Model")

def test_embedding_import():
    from sentence_transformers import SentenceTransformer
    # Don't load model (heavy), just verify import
    assert SentenceTransformer is not None

test("sentence-transformers import", test_embedding_import)


def test_embed_single():
    from ingestion.article_processor import embed_single
    vec = embed_single("RBI cuts repo rate by 25 basis points")
    assert vec.shape == (384,)
    import numpy as np
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 0.01, f"Vector not normalised: norm={norm}"

test("Single embedding (384-dim, normalised)", test_embed_single)


def test_embed_batch():
    from ingestion.article_processor import embed_texts
    import numpy as np
    texts = ["First article about markets.", "Second article about startups.", "Third about budget."]
    vecs = embed_texts(texts)
    assert vecs.shape == (3, 384)
    assert vecs.dtype == np.float32

test("Batch embedding (3 texts → 3×384 matrix)", test_embed_batch)


# ── 4. LanceDB ────────────────────────────────────────────────────────────────
print("\n[4/8] LanceDB Vector Store")

def test_lancedb_create():
    import lancedb
    from config import LANCE_DIR
    db = lancedb.connect(str(LANCE_DIR))
    assert db is not None

test("LanceDB connection", test_lancedb_create)


def test_lancedb_upsert():
    from ingestion.rss_fetcher import get_mock_articles
    from ingestion.article_processor import process_article
    from vector_store.lancedb_manager import upsert_articles, article_count

    raw = get_mock_articles()[:2]  # Only 2 for speed
    processed = [process_article(a) for a in raw]
    n = upsert_articles(processed)
    assert n >= 0  # May be 0 if articles already exist
    total = article_count()
    assert total >= 0

test("LanceDB upsert (2 articles)", test_lancedb_upsert)


def test_lancedb_search():
    from vector_store.lancedb_manager import semantic_search, article_count
    if article_count() == 0:
        print(f"{SKIP}  Skipping search test — DB empty")
        return
    results = semantic_search("interest rate monetary policy", top_k=3)
    # Results may be empty if no matching articles
    assert isinstance(results, list)

test("LanceDB vector search", test_lancedb_search)


# ── 5. Persona & My ET ────────────────────────────────────────────────────────
print("\n[5/8] My ET — Personalized Newsroom")

def test_temporal_decay():
    from modules.my_et import _temporal_decay
    from datetime import datetime, timezone, timedelta
    now_iso   = datetime.now(timezone.utc).isoformat()
    old_iso   = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    score_now = _temporal_decay(now_iso)
    score_old = _temporal_decay(old_iso)
    assert score_now > score_old, "Recent articles must score higher than old ones"
    assert 0 <= score_now <= 1
    assert 0 <= score_old <= 1

test("Temporal decay function", test_temporal_decay)


def test_persona_score():
    from modules.my_et import _persona_score
    article = {
        "title": "SEBI tightens mutual fund regulations; SIP inflows hit record ₹23,000 crore",
        "summary": "SEBI has issued new guidelines for mutual fund NAV calculation.",
        "entities": ["SEBI", "AMC", "NSE"],
        "readability_score": 65.0,
    }
    score_mf      = _persona_score(article, "Mutual Fund Investor")
    score_student = _persona_score(article, "Student / Learner")
    assert 0 <= score_mf <= 1
    assert 0 <= score_student <= 1

test("Persona relevance scoring", test_persona_score)


def test_user_profile():
    from modules.my_et import UserProfileManager
    import numpy as np
    mgr = UserProfileManager()
    assert mgr.is_cold()
    vec = list(np.random.randn(384).astype("float32"))
    mgr.update(vec, weight=1.0)
    mgr.update(vec, weight=0.8)
    mgr.update(vec, weight=0.9)
    assert not mgr.is_cold()
    pv = mgr.get_vector()
    assert pv.shape == (384,)

test("User profile vector (decay update)", test_user_profile)


def test_feed_generation():
    from modules.my_et import get_personalized_feed
    from vector_store.lancedb_manager import article_count
    if article_count() == 0:
        print(f"{SKIP}  Skipping feed test — DB empty")
        return
    feed = get_personalized_feed("Mutual Fund Investor", top_n=3, days_back=365)
    assert isinstance(feed, list)

test("Personalized feed generation", test_feed_generation)


# ── 6. News Navigator ─────────────────────────────────────────────────────────
print("\n[6/8] News Navigator")

def test_conversation_manager():
    from modules.news_navigator import ConversationManager
    cm = ConversationManager(max_turns=3)
    assert cm.is_empty()
    cm.add_turn("user", "What is the RBI policy?")
    cm.add_turn("assistant", "The RBI cut rates by 25 bps.")
    assert not cm.is_empty()
    ctx = cm.get_context_string()
    assert "RBI" in ctx
    cm.clear()
    assert cm.is_empty()

test("Conversation manager (multi-turn history)", test_conversation_manager)


def test_llm_prompts():
    from llm.ollama_client import (
        build_eli5_prompt, build_persona_summary_prompt,
        build_navigator_map_prompt, build_navigator_reduce_prompt,
        build_arc_evolution_prompt, build_what_to_watch_prompt,
    )
    text = "The RBI cut repo rates by 25 bps to 6.25%."
    assert "ELI5" in build_eli5_prompt(text) or "simple" in build_eli5_prompt(text).lower()
    assert "Mutual Fund Investor" in build_persona_summary_prompt(text, "Mutual Fund Investor", "data-driven")
    assert "KEY FACTS" in build_navigator_map_prompt(text, "rate cut impact")
    reduce_prompt = build_navigator_reduce_prompt("facts here", "query here", "executive", "sources here")
    assert "Chain-of-Thought" in reduce_prompt or "CORE NARRATIVE" in reduce_prompt
    assert "ORIGIN" in build_arc_evolution_prompt("event 1\nevent 2", "RBI policy")
    assert "Watch" in build_what_to_watch_prompt("summary", "RBI, SEBI")

test("All LLM prompt templates", test_llm_prompts)


def test_extract_follow_ups():
    from modules.news_navigator import _extract_follow_up_questions
    briefing = """
This is the briefing content.

Suggested follow-up questions:
1. How will this impact equity markets in Q2?
2. What sectors benefit most from the rate cut?
3. Should retail investors rebalance their portfolio now?
"""
    qs = _extract_follow_up_questions(briefing)
    assert len(qs) >= 1
    assert any("?" in q for q in qs)

test("Follow-up question extraction from briefing", test_extract_follow_ups)


# ── 7. Story Arc Tracker ──────────────────────────────────────────────────────
print("\n[7/8] Story Arc Tracker")

def test_graph_creation():
    import networkx as nx
    from modules.story_arc import (
        add_article_to_graph, _article_node_id, _entity_node_id,
    )
    G = nx.DiGraph()
    article = {
        "id": "test_art_001",
        "title": "RBI cuts rates",
        "url": "https://example.com/rbi",
        "published_at": "2026-03-18T10:00:00+00:00",
        "vertical": "Economy",
        "summary": "RBI cuts repo rate by 25 bps.",
        "sentiment_compound": 0.45,
        "entities": ["RBI", "Sanjay Malhotra", "Nifty"],
        "vector": [0.1] * 50,
    }
    node_id = add_article_to_graph(G, article)
    assert G.has_node(node_id)
    assert G.number_of_nodes() >= 4  # 1 article + 3 entities

test("Knowledge graph node creation", test_graph_creation)


def test_jaccard_similarity():
    from modules.story_arc import _jaccard_similarity
    a = {"RBI", "SEBI", "NSE"}
    b = {"RBI", "SEBI", "BSE"}
    j = _jaccard_similarity(a, b)
    assert abs(j - 0.5) < 0.01, f"Expected 0.5, got {j}"
    assert _jaccard_similarity(set(), set()) == 0.0
    assert _jaccard_similarity({"A"}, {"A"}) == 1.0

test("Jaccard entity similarity", test_jaccard_similarity)


def test_association_score():
    from modules.story_arc import compute_association_score
    art_a = {
        "published_at": "2026-03-15T10:00:00+00:00",
        "vector": [0.8, 0.6] + [0.0] * 48,
        "entities": ["RBI", "SEBI", "Nifty"],
    }
    art_b = {
        "published_at": "2026-03-16T12:00:00+00:00",
        "vector": [0.7, 0.7] + [0.0] * 48,
        "entities": ["RBI", "Sensex", "Nifty"],
    }
    art_c = {
        "published_at": "2026-01-01T00:00:00+00:00",
        "vector": [-0.9, 0.1] + [0.0] * 48,
        "entities": ["Zepto", "SoftBank", "Sequoia"],
    }
    score_ab = compute_association_score(art_a, art_b)
    score_ac = compute_association_score(art_a, art_c)
    assert 0 <= score_ab <= 1
    assert 0 <= score_ac <= 1
    assert score_ab > score_ac, "Close related articles must score higher than distant unrelated ones"

test("Arc association scoring (temporal + semantic + entity)", test_association_score)


def test_sentiment_trajectory():
    import networkx as nx
    from modules.story_arc import add_article_to_graph, get_sentiment_trajectory

    G = nx.DiGraph()
    articles = [
        {"id": f"tst_{i}", "title": f"Article {i}", "url": f"https://ex.com/{i}",
         "published_at": f"2026-03-{10+i:02d}T10:00:00+00:00",
         "vertical": "Economy", "summary": "test",
         "sentiment_compound": 0.1 * i - 0.2,
         "entities": ["RBI"], "vector": [0.1] * 50}
        for i in range(5)
    ]
    art_nodes = [add_article_to_graph(G, a) for a in articles]
    traj = get_sentiment_trajectory(G, art_nodes)
    assert len(traj) == 5
    assert "rolling_avg" in traj[0]

test("Sentiment trajectory calculation", test_sentiment_trajectory)


def test_contrarian_detection():
    import networkx as nx
    from modules.story_arc import add_article_to_graph, find_contrarian_articles

    G = nx.DiGraph()
    bullish = {"id": "bull1", "title": "Markets surge on rate cut optimism", "url": "#",
               "published_at": "2026-03-17T10:00:00+00:00", "vertical": "Markets",
               "summary": "...", "sentiment_compound": 0.65, "entities": ["Nifty", "Sensex"], "vector": [0.1]*50}
    bearish = {"id": "bear1", "title": "Rate cut risks inflation resurgence warns economist", "url": "#",
               "published_at": "2026-03-17T11:00:00+00:00", "vertical": "Economy",
               "summary": "...", "sentiment_compound": -0.55, "entities": ["RBI", "Inflation"], "vector": [0.1]*50}
    n1 = add_article_to_graph(G, bullish)
    n2 = add_article_to_graph(G, bearish)
    contrarians = find_contrarian_articles(G, [n1, n2])
    assert len(contrarians) == 1
    assert contrarians[0]["divergence"] >= 0.4

test("Contrarian article detection", test_contrarian_detection)


# ── 8. Ollama connectivity (soft check) ──────────────────────────────────────
print("\n[8/8] LLM Connectivity")

def test_ollama_check():
    from llm.ollama_client import check_ollama_alive, list_available_models
    alive = check_ollama_alive()
    if alive:
        models = list_available_models()
        print(f"       Ollama online · Models: {models}")
    else:
        print(f"  ⚠️   Ollama is offline — start with `ollama serve` and pull llama3.2:3b")
        print(f"       AI summaries and briefings will be unavailable until Ollama is running.")

test("Ollama server ping", test_ollama_check)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  Results: {passed} passed · {failed} failed / {len(results)} total")

if failed > 0:
    print("\n  FAILURES:")
    for name, ok, err in results:
        if not ok:
            print(f"  ❌ {name}: {err}")
    print()
    sys.exit(1)
else:
    print("\n  All tests passed! 🎉  Launch with: streamlit run app.py")
    print("="*60 + "\n")
