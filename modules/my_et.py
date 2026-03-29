"""
modules/my_et.py
My ET — The Personalized Newsroom

Implements dual-pass retrieval and persona-based re-ranking:
  Pass 1: Vector cosine similarity + temporal decay → top-50 candidates
  Pass 2: Persona keyword boost + entity cross-reference → final ranked feed
"""

from __future__ import annotations
import logging
import math
from datetime import datetime, timezone

import numpy as np

from config import (
    PERSONAS, LANCEDB_TOP_K_FIRST_PASS,
    EMBEDDING_DIM,
)
from vector_store.lancedb_manager import semantic_search, get_recent_articles
from ingestion.article_processor import embed_single
from llm.ollama_client import (
    generate_with_fallback,
    build_eli5_prompt,
    build_persona_summary_prompt,
    SUMMARY_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


# ── User Profile Vector Manager ───────────────────────────────────────────────

class UserProfileManager:
    """
    Maintains an exponentially decayed User Profile Vector.
    In a real system this persists to disk/DB; here it lives in st.session_state.
    """

    def __init__(self, decay_factor: float = 0.85):
        self.decay_factor      = decay_factor
        self.interaction_count = 0
        self.profile_vector    = np.zeros(EMBEDDING_DIM, dtype="float32")

    def update(self, article_vector: list[float], weight: float = 1.0) -> None:
        vec  = np.array(article_vector, dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.profile_vector = (
            self.decay_factor * self.profile_vector
            + (1 - self.decay_factor) * weight * vec
        )
        self.interaction_count += 1

    def get_vector(self) -> np.ndarray:
        norm = np.linalg.norm(self.profile_vector)
        if norm > 0:
            return self.profile_vector / norm
        return self.profile_vector

    def is_cold(self) -> bool:
        return self.interaction_count < 3


# ── Temporal Decay ────────────────────────────────────────────────────────────

def _temporal_decay(published_at_str: str, lambda_hours: float = 24.0) -> float:
    """
    Exponential time-decay: score × e^(−λ × age_hours).
    λ=24h → article ~1 day old retains e^-1 ≈ 37%.
    Breaking news (< 2h) retains > 92%.
    """
    try:
        published = datetime.fromisoformat(
            published_at_str.replace("Z", "+00:00")
        )
        now       = datetime.now(timezone.utc)
        age_hours = max((now - published).total_seconds() / 3600, 0)
        return math.exp(-age_hours / lambda_hours)
    except Exception:
        return 0.5


# ── Persona Re-ranking ────────────────────────────────────────────────────────

def _persona_score(article: dict, persona_key: str) -> float:
    """
    Compute persona relevance boost ∈ [0, 1].
    Based on keyword overlap + entity boost matching.
    """
    if persona_key not in PERSONAS:
        return 0.0

    persona  = PERSONAS[persona_key]
    text     = (article.get("title", "") + " " + article.get("summary", "")).lower()
    entities = [e.lower() for e in article.get("entities", [])]

    kw_matches  = sum(1 for kw in persona["keywords"] if kw.lower() in text)
    kw_score    = min(kw_matches / max(len(persona["keywords"]), 1), 1.0)

    entity_matches = sum(
        1 for be in persona["boost_entities"]
        if be.lower() in entities or be.lower() in text
    )
    entity_score = (
        min(entity_matches / max(len(persona["boost_entities"]), 1), 1.0)
        if persona["boost_entities"] else 0.0
    )

    readability_bonus = 0.0
    if persona_key == "Student / Learner":
        read_score        = float(article.get("readability_score", 50.0))
        readability_bonus = (read_score / 100.0) * 0.3

    return 0.5 * kw_score + 0.3 * entity_score + 0.2 * readability_bonus


# ── Main Ranking Function ─────────────────────────────────────────────────────

def get_personalized_feed(
    persona_key: str,
    verticals: list[str] | None = None,
    user_profile: UserProfileManager | None = None,
    top_n: int = 10,
    days_back: int = 7,
) -> list[dict]:
    """
    Dual-pass personalized news feed.

    Pass 1: Semantic search using persona keywords
    Pass 2: Score = α·cosine + β·temporal_decay + γ·persona_score
    """
    persona    = PERSONAS.get(persona_key, PERSONAS["Mutual Fund Investor"])
    query_base = " ".join(persona["keywords"][:8])

    candidates = semantic_search(
        query_text=query_base,
        top_k=LANCEDB_TOP_K_FIRST_PASS,
        verticals=verticals,
        days_back=days_back,
    )

    if not candidates:
        logger.warning("No candidates from vector search — returning recent articles.")
        candidates = get_recent_articles(n=20, days_back=days_back)

    scored = []
    for art in candidates:
        distance   = art.get("_distance", 0.5)
        cosine_sim = max(0.0, 1.0 - distance)
        td         = _temporal_decay(art.get("published_at", ""))
        ps         = _persona_score(art, persona_key)
        final_score = 0.45 * cosine_sim + 0.30 * td + 0.25 * ps
        scored.append({**art, "_score": round(final_score, 4)})

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_n]


# ── Persona-Specific Augmentations ───────────────────────────────────────────

def generate_eli5_summary(article: dict) -> str:
    """
    Generate an ELI5 2-sentence summary for the student persona.
    Falls back to the article summary if Ollama is unavailable.
    """
    text = article.get("full_text") or article.get("summary", "")
    if not text:
        return ""
    summary, _ = generate_with_fallback(
        build_eli5_prompt(text),
        fallback_text=article.get("summary", "")[:200],
        temperature=0.4,
        max_tokens=SUMMARY_MAX_TOKENS,
    )
    return summary


def generate_persona_summary(article: dict, persona_key: str) -> str:
    """
    Generate a persona-tailored summary for the given article.
    Falls back to the raw article summary on LLM error.
    """
    text    = article.get("full_text") or article.get("summary", "")
    persona = PERSONAS.get(persona_key, {})
    style   = persona.get("summary_style", "professional and concise")
    if not text:
        return article.get("summary", "")[:300]
    summary, _ = generate_with_fallback(
        build_persona_summary_prompt(text, persona_key, style),
        fallback_text=article.get("summary", "")[:300],
        temperature=0.3,
        max_tokens=SUMMARY_MAX_TOKENS,
    )
    return summary


def format_article_card(article: dict, persona_key: str) -> dict:
    """
    Returns a display-ready dict with persona-adapted fields.
    Does NOT call the LLM — summaries are generated on-demand via expanders.
    """
    sentiment      = article.get("sentiment_label", "neutral")
    sentiment_icon = {
        "positive": "📈", "negative": "📉", "neutral": "➡️"
    }.get(sentiment, "➡️")
    compound = article.get("sentiment_compound", 0.0)

    return {
        "id":              article.get("id", ""),
        "title":           article.get("title", "Untitled"),
        "lead":            article.get("lead", article.get("summary", ""))[:200],
        "url":             article.get("url", "#"),
        "vertical":        article.get("vertical", "General"),
        "published_at":    article.get("published_at", ""),
        "entities":        article.get("entities", [])[:6],
        "sentiment":       sentiment,
        "sentiment_icon":  sentiment_icon,
        "sentiment_score": compound,
        "readability":     article.get("readability_score", 50.0),
        "score":           article.get("_score", 0.0),
        "word_count":      article.get("word_count", 0),
    }
