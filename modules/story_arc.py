"""
modules/story_arc.py
Story Arc Tracker

Builds and traverses a dynamic Knowledge Graph (NetworkX) to track
long-running business narratives — mapping entities, sentiment shifts,
contrarian perspectives, and predicting "what to watch next."

CPU token budget:
  evolution_briefing → ARC_MAX_TOKENS  (350)  ≈ 23-35 s
  what_to_watch      → WATCH_MAX_TOKENS(180)  ≈ 12-18 s
  Total LLM time                              ≈ 35-53 s  (safe on 300 s timeout)
"""

from __future__ import annotations
import logging
import pickle
from datetime import datetime, timezone, timedelta

import networkx as nx
import numpy as np

from config import (
    GRAPH_DIR,
    ARC_ASSOCIATION_THRESHOLD,
    ARC_TEMPORAL_WEIGHT, ARC_SEMANTIC_WEIGHT, ARC_ENTITY_WEIGHT,
    SENTIMENT_WINDOW,
    LANCEDB_TOP_K_STORY,
)
from vector_store.lancedb_manager import semantic_search
from llm.ollama_client import (
    generate_with_fallback,
    SYSTEM_NEWS_ANALYST,
    build_arc_evolution_prompt,
    build_what_to_watch_prompt,
    ARC_MAX_TOKENS,
    WATCH_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

_GRAPH_PICKLE = GRAPH_DIR / "knowledge_graph.pkl"


# ── Knowledge Graph Persistence ───────────────────────────────────────────────

def load_graph() -> nx.DiGraph:
    """Load the persisted knowledge graph or create a fresh one."""
    if _GRAPH_PICKLE.exists():
        try:
            with open(_GRAPH_PICKLE, "rb") as f:
                G = pickle.load(f)
            logger.info(
                "Loaded knowledge graph: %d nodes, %d edges.",
                G.number_of_nodes(), G.number_of_edges(),
            )
            return G
        except Exception as e:
            logger.warning("Graph load failed (%s) — creating fresh graph.", e)
    return nx.DiGraph()


def save_graph(G: nx.DiGraph) -> None:
    """Persist the knowledge graph to disk."""
    try:
        with open(_GRAPH_PICKLE, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error("Graph save failed: %s", e)


# ── Node Naming Helpers ───────────────────────────────────────────────────────

def _article_node_id(article_id: str) -> str: return f"ART:{article_id}"
def _entity_node_id(entity: str)       -> str: return f"ENT:{entity.lower().strip()}"


# ── Graph Construction ────────────────────────────────────────────────────────

def add_article_to_graph(G: nx.DiGraph, article: dict) -> str:
    """
    Adds an article node + entity nodes + edges to the knowledge graph.
    Returns the article node ID.
    """
    art_id    = _article_node_id(article["id"])
    published = article.get("published_at", datetime.now(timezone.utc).isoformat())
    sentiment = article.get("sentiment_compound", 0.0)
    vector    = article.get("vector", [])

    G.add_node(art_id, **{
        "type":         "article",
        "article_id":   article["id"],
        "title":        article.get("title", ""),
        "url":          article.get("url", "#"),
        "published_at": published,
        "vertical":     article.get("vertical", ""),
        "summary":      article.get("summary", "")[:300],
        "sentiment":    sentiment,
        "vector":       vector[:50] if vector else [],
        "entities":     article.get("entities", []),
    })

    for entity in article.get("entities", [])[:15]:
        ent_id = _entity_node_id(entity)
        if not G.has_node(ent_id):
            G.add_node(ent_id, type="entity", name=entity, article_count=0)
        G.nodes[ent_id]["article_count"] = G.nodes[ent_id].get("article_count", 0) + 1
        G.add_edge(ent_id, art_id, relation="MENTIONED_IN")
        G.add_edge(art_id, ent_id, relation="MENTIONS")

    return art_id


# ── Arc Association Scoring ───────────────────────────────────────────────────

def _jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union if union else 0.0


def _temporal_closeness(ts_a: str, ts_b: str, max_days: float = 30.0) -> float:
    try:
        a = datetime.fromisoformat(ts_a.replace("Z", "+00:00"))
        b = datetime.fromisoformat(ts_b.replace("Z", "+00:00"))
        days_apart = abs((a - b).total_seconds()) / 86400
        return max(0.0, 1.0 - (days_apart / max_days))
    except Exception:
        return 0.5


def _cosine_similarity_from_lists(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.array(a, dtype="float32"), np.array(b, dtype="float32")
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def compute_association_score(article_a: dict, article_b: dict) -> float:
    """
    Composite arc association score between two articles.
    S = w_t * temporal + w_s * semantic + w_e * entity_jaccard
    """
    t_score = _temporal_closeness(
        article_a.get("published_at", ""),
        article_b.get("published_at", ""),
    )
    vec_a   = article_a.get("vector", [])[:50]
    vec_b   = article_b.get("vector", [])[:50]
    s_score = _cosine_similarity_from_lists(vec_a, vec_b)
    ents_a  = set(article_a.get("entities", []))
    ents_b  = set(article_b.get("entities", []))
    e_score = _jaccard_similarity(ents_a, ents_b)

    composite = (
        ARC_TEMPORAL_WEIGHT * t_score
        + ARC_SEMANTIC_WEIGHT * s_score
        + ARC_ENTITY_WEIGHT * e_score
    )
    return round(composite, 4)


# ── Story Arc Clustering ──────────────────────────────────────────────────────

def build_story_arcs(G: nx.DiGraph, articles: list[dict]) -> dict[str, list[str]]:
    """
    Clusters articles into story arcs using association scoring.
    Returns mapping: arc_id → [article_node_id, ...]
    """
    arcs: dict[str, list[str]] = {}
    arc_representatives: dict[str, dict] = {}

    for article in sorted(
        articles, key=lambda x: x.get("published_at", ""), reverse=False
    ):
        art_node   = add_article_to_graph(G, article)
        best_arc   = None
        best_score = 0.0

        for arc_id, rep_art in arc_representatives.items():
            score = compute_association_score(article, rep_art)
            if score > best_score:
                best_score = score
                best_arc   = arc_id

        if best_arc and best_score >= ARC_ASSOCIATION_THRESHOLD:
            arcs[best_arc].append(art_node)
            arc_representatives[best_arc] = article
            if len(arcs[best_arc]) >= 2:
                G.add_edge(arcs[best_arc][-2], art_node, relation="SUBSEQUENT_TO")
        else:
            arc_id = f"arc_{article['id'][:8]}"
            arcs[arc_id] = [art_node]
            arc_representatives[arc_id] = article

    return arcs


# ── Arc Analysis ──────────────────────────────────────────────────────────────

def get_key_players(G: nx.DiGraph, arc_articles: list[str]) -> list[dict]:
    """
    Uses PageRank on the entity sub-graph to identify key players in the arc.
    """
    sub_nodes = set()
    for art_node in arc_articles:
        sub_nodes.add(art_node)
        sub_nodes.update(G.successors(art_node))

    subgraph      = G.subgraph(sub_nodes)
    entity_nodes  = [n for n in subgraph.nodes if G.nodes[n].get("type") == "entity"]

    if len(entity_nodes) < 2:
        return [
            {
                "name":     G.nodes[n].get("name", n),
                "score":    1.0,
                "mentions": G.nodes[n].get("article_count", 1),
            }
            for n in entity_nodes
        ]

    try:
        pagerank = nx.pagerank(subgraph, alpha=0.85)
    except Exception:
        pagerank = {n: 1.0 for n in entity_nodes}

    players = [
        {
            "name":     G.nodes[n].get("name", n.replace("ENT:", "")),
            "score":    round(pagerank.get(n, 0.0), 6),
            "mentions": G.nodes[n].get("article_count", 1),
        }
        for n in entity_nodes
    ]
    players.sort(key=lambda x: x["score"], reverse=True)
    return players[:10]


def get_sentiment_trajectory(G: nx.DiGraph, arc_articles: list[str]) -> list[dict]:
    """Returns chronological sentiment scores with rolling average."""
    timeline = []
    for art_node in arc_articles:
        node_data = G.nodes.get(art_node, {})
        if node_data.get("type") == "article":
            timeline.append({
                "title":        node_data.get("title", "")[:50],
                "published_at": node_data.get("published_at", ""),
                "sentiment":    node_data.get("sentiment", 0.0),
            })

    timeline.sort(key=lambda x: x["published_at"])

    scores = [t["sentiment"] for t in timeline]
    for i, t in enumerate(timeline):
        window_start = max(0, i - SENTIMENT_WINDOW + 1)
        window       = scores[window_start: i + 1]
        t["rolling_avg"] = round(sum(window) / len(window), 4)

    return timeline


def find_contrarian_articles(G: nx.DiGraph, arc_articles: list[str]) -> list[dict]:
    """Identifies articles with high semantic proximity but opposite sentiment."""
    article_nodes = [
        (n, G.nodes[n])
        for n in arc_articles
        if G.nodes.get(n, {}).get("type") == "article"
    ]

    contrarians = []
    for i, (n_a, d_a) in enumerate(article_nodes):
        for n_b, d_b in article_nodes[i + 1:]:
            sent_a = d_a.get("sentiment", 0.0)
            sent_b = d_b.get("sentiment", 0.0)
            if abs(sent_a - sent_b) >= 0.4 and sent_a * sent_b < 0:
                contrarians.append({
                    "article_a": {
                        "title":     d_a.get("title", "")[:60],
                        "url":       d_a.get("url", "#"),
                        "sentiment": sent_a,
                    },
                    "article_b": {
                        "title":     d_b.get("title", "")[:60],
                        "url":       d_b.get("url", "#"),
                        "sentiment": sent_b,
                    },
                    "divergence": round(abs(sent_a - sent_b), 4),
                })

    contrarians.sort(key=lambda x: x["divergence"], reverse=True)
    return contrarians[:5]


# ── High-Level Story Arc API ──────────────────────────────────────────────────

def discover_story_arcs(topic: str, days_back: int = 30) -> dict:
    """
    Main entry point: given a topic, find related articles, build/update
    the knowledge graph, and return a comprehensive arc analysis.

    LLM calls are made via generate_with_fallback so a timeout produces
    a graceful text fallback rather than a raw traceback in the UI.
    """
    # ── Retrieve related articles ────────────────────────────────────────────
    candidates = semantic_search(
        query_text=topic,
        top_k=LANCEDB_TOP_K_STORY,
        days_back=days_back,
    )

    if not candidates:
        return {
            "topic":         topic,
            "article_count": 0,
            "arcs":          [],
            "error":         "No articles found. Please ingest news first.",
        }

    # ── Build / update graph ─────────────────────────────────────────────────
    G    = load_graph()
    arcs = build_story_arcs(G, candidates)
    save_graph(G)

    if not arcs:
        return {
            "topic": topic, "article_count": len(candidates),
            "arcs": [], "error": "No arcs formed from retrieved articles.",
        }

    dominant_arc_id  = max(arcs, key=lambda k: len(arcs[k]))
    dominant_articles = arcs[dominant_arc_id]

    # ── Analysis ─────────────────────────────────────────────────────────────
    key_players    = get_key_players(G, dominant_articles)
    sentiment_traj = get_sentiment_trajectory(G, dominant_articles)
    contrarians    = find_contrarian_articles(G, dominant_articles)

    # Build timeline text for LLM
    timeline_lines = []
    for item in sentiment_traj:
        date  = item["published_at"][:10]
        title = item["title"]
        sent  = (
            "📈 Positive" if item["sentiment"] > 0.05
            else ("📉 Negative" if item["sentiment"] < -0.05 else "➡️ Neutral")
        )
        timeline_lines.append(
            f"[{date}] {title} — {sent} (score: {item['sentiment']:.2f})"
        )

    events_text  = "\n".join(timeline_lines) or "No timeline data available."
    entity_names = ", ".join(p["name"] for p in key_players[:6])

    # ── LLM: Evolution briefing ───────────────────────────────────────────────
    evolution_briefing, evo_ok = generate_with_fallback(
        build_arc_evolution_prompt(events_text, topic),
        fallback_text=(
            f"**Story arc for '{topic}'** spans {len(dominant_articles)} articles "
            f"with key players: {entity_names}.\n\n"
            f"**Timeline:**\n{events_text}"
        ),
        system=SYSTEM_NEWS_ANALYST,
        temperature=0.3,
        max_tokens=ARC_MAX_TOKENS,
    )
    if not evo_ok:
        logger.warning("Evolution briefing LLM unavailable — using fallback.")

    # ── LLM: What to watch ────────────────────────────────────────────────────
    what_to_watch, watch_ok = generate_with_fallback(
        build_what_to_watch_prompt(evolution_briefing, entity_names),
        fallback_text=(
            "1. **Monitor regulatory filings** — Watch for policy announcements "
            "from relevant authorities.\n"
            "2. **Track earnings & results** — Next quarterly results will "
            "confirm or reverse current trends.\n"
            "3. **Watch competitor moves** — Any rival announcement could "
            "shift the narrative significantly."
        ),
        system=SYSTEM_NEWS_ANALYST,
        temperature=0.4,
        max_tokens=WATCH_MAX_TOKENS,
    )
    if not watch_ok:
        logger.warning("What-to-watch LLM unavailable — using fallback.")

    # ── Per-arc summaries ─────────────────────────────────────────────────────
    arc_summaries = []
    for arc_id, arc_arts in sorted(
        arcs.items(), key=lambda x: len(x[1]), reverse=True
    )[:5]:
        arc_node_data = [
            G.nodes.get(n, {})
            for n in arc_arts
            if G.nodes.get(n, {}).get("type") == "article"
        ]
        titles = [d.get("title", "")[:60] for d in arc_node_data]
        dates  = sorted([d.get("published_at", "")[:10] for d in arc_node_data])
        arc_summaries.append({
            "arc_id":        arc_id,
            "article_count": len(arc_arts),
            "titles":        titles[:3],
            "date_range": (
                f"{dates[0] if dates else '?'} → "
                f"{dates[-1] if len(dates) > 1 else '?'}"
            ),
            "is_dominant": arc_id == dominant_arc_id,
        })

    return {
        "topic":                 topic,
        "article_count":         len(candidates),
        "arc_count":             len(arcs),
        "dominant_arc_id":       dominant_arc_id,
        "dominant_article_count": len(dominant_articles),
        "key_players":           key_players,
        "sentiment_trajectory":  sentiment_traj,
        "contrarian_articles":   contrarians,
        "evolution_briefing":    evolution_briefing,
        "what_to_watch":         what_to_watch,
        "arc_summaries":         arc_summaries,
        "llm_available":         evo_ok and watch_ok,
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
    }


def get_graph_stats() -> dict:
    """Returns summary statistics about the current knowledge graph."""
    G             = load_graph()
    entity_nodes  = [n for n in G.nodes if G.nodes[n].get("type") == "entity"]
    article_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "article"]
    return {
        "total_nodes":  G.number_of_nodes(),
        "total_edges":  G.number_of_edges(),
        "entity_count": len(entity_nodes),
        "article_count": len(article_nodes),
    }
