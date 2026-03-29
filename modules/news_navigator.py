"""
modules/news_navigator.py  (UPGRADED)
News Navigator — Interactive Intelligence Briefings

Key upgrade: MAP phase is now PARALLEL using ThreadPoolExecutor.
  Before: 8 articles × 30s/call (Ollama CPU) = ~4 min just for MAP
  After:  8 articles, 6 workers, Groq at 1200 tok/s = ~8 seconds total

Also: top_k raised from 4 → 8 for richer, more comprehensive briefings.
"""

from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import Generator

from config import (
    PERSONAS, MAP_REDUCE_CHUNK_ARTICLES,
    LANCEDB_TOP_K_NAVIGATOR, MAX_CONTEXT_TOKENS,
    MAP_MAX_WORKERS,
)
from vector_store.lancedb_manager import semantic_search
from llm.ollama_client import (
    generate,
    generate_stream,
    generate_with_fallback,
    SYSTEM_NEWS_ANALYST,
    build_navigator_map_prompt,
    build_navigator_reduce_prompt,
    MAP_MAX_TOKENS,
    REDUCE_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


# ── Conversation History Manager ──────────────────────────────────────────────

class ConversationManager:
    """Sliding-window conversation history for multi-turn Q&A."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content[:600]})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[2:]

    def get_context_string(self) -> str:
        if not self.history:
            return ""
        lines = []
        for turn in self.history[-6:]:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history.clear()

    def is_empty(self) -> bool:
        return len(self.history) == 0


# ── MAP Phase — per-article fact extraction (parallelised) ────────────────────

def _map_article_to_facts(article: dict, query: str, source_index: int) -> dict:
    """
    Extracts 3-5 core facts from a single article relevant to the query.
    Runs inside a ThreadPoolExecutor worker — must be thread-safe (it is,
    because each call makes its own HTTP request to Groq/Ollama).
    """
    text   = (article.get("full_text") or article.get("summary", ""))[:1200]
    title  = article.get("title", f"Article {source_index}")
    url    = article.get("url", "#")
    source = f"Source {source_index}"
    prompt = build_navigator_map_prompt(text, query)

    facts, ok = generate_with_fallback(
        prompt,
        fallback_text=f"• {article.get('summary', 'No summary available.')[:300]}",
        system=SYSTEM_NEWS_ANALYST,
        temperature=0.1,
        max_tokens=MAP_MAX_TOKENS,
    )
    if not ok:
        logger.warning("MAP fallback used for article %d.", source_index)

    return {
        "source_label": source,
        "title":        title,
        "url":          url,
        "facts":        facts,
        "vertical":     article.get("vertical", ""),
        "published_at": article.get("published_at", ""),
        "sentiment":    article.get("sentiment_label", "neutral"),
    }


def _run_map_parallel(candidates: list[dict], query: str) -> list[dict]:
    """
    Runs MAP phase in parallel across all candidate articles.
    Uses up to MAP_MAX_WORKERS threads. Falls back to sequential on errors.
    """
    n = len(candidates)
    if n == 0:
        return []

    fact_cards: list[dict | None] = [None] * n

    try:
        with ThreadPoolExecutor(max_workers=min(MAP_MAX_WORKERS, n)) as executor:
            future_to_idx = {
                executor.submit(_map_article_to_facts, art, query, i + 1): i
                for i, art in enumerate(candidates)
            }
            # 90s timeout per batch — Groq is fast; Ollama may be slower
            for future in as_completed(future_to_idx, timeout=120):
                idx = future_to_idx[future]
                try:
                    fact_cards[idx] = future.result()
                except Exception as e:
                    logger.error("MAP worker error for idx %d: %s", idx, e)
                    art = candidates[idx]
                    fact_cards[idx] = {
                        "source_label": f"Source {idx + 1}",
                        "title":        art.get("title", ""),
                        "url":          art.get("url", "#"),
                        "facts":        art.get("summary", "")[:300],
                        "vertical":     art.get("vertical", ""),
                        "published_at": art.get("published_at", ""),
                        "sentiment":    art.get("sentiment_label", "neutral"),
                    }
    except FuturesTimeout:
        logger.warning("MAP phase timed out — using available results.")

    # Fill any remaining None slots with basic summaries
    for i, card in enumerate(fact_cards):
        if card is None:
            art = candidates[i]
            fact_cards[i] = {
                "source_label": f"Source {i + 1}",
                "title":        art.get("title", ""),
                "url":          art.get("url", "#"),
                "facts":        art.get("summary", "")[:300],
                "vertical":     art.get("vertical", ""),
                "published_at": art.get("published_at", ""),
                "sentiment":    art.get("sentiment_label", "neutral"),
            }

    return fact_cards


# ── Reduce Phase — multi-article synthesis ────────────────────────────────────

def _reduce_to_briefing(
    fact_cards: list[dict],
    query: str,
    persona_key: str,
    conversation_context: str = "",
) -> tuple[str, bool]:
    persona = PERSONAS.get(persona_key, PERSONAS["Mutual Fund Investor"])
    style   = persona.get("summary_style", "professional and concise")

    chars_per_source = max(
        400,
        (MAX_CONTEXT_TOKENS * 4) // max(len(fact_cards), 1) - 200,
    )

    facts_parts, source_list_parts = [], []
    for card in fact_cards:
        facts_parts.append(
            f"[{card['source_label']}] {card['title']}\n"
            f"Date: {card['published_at'][:10]} | Vertical: {card['vertical']}\n"
            f"{card['facts'][:chars_per_source]}"
        )
        source_list_parts.append(
            f"{card['source_label']}: {card['title']} ({card['url']})"
        )

    facts_combined = "\n\n---\n\n".join(facts_parts)
    source_list    = "\n".join(source_list_parts)

    if conversation_context:
        facts_combined += (
            f"\n\nPREVIOUS CONTEXT (do not repeat):\n{conversation_context[:500]}"
        )

    prompt = build_navigator_reduce_prompt(
        facts_combined=facts_combined,
        query=query,
        persona_style=style,
        source_list=source_list,
    )

    briefing, ok = generate_with_fallback(
        prompt,
        fallback_text=_build_fallback_briefing(fact_cards, query),
        system=SYSTEM_NEWS_ANALYST,
        temperature=0.25,
        max_tokens=REDUCE_MAX_TOKENS,
    )
    return briefing, ok


def _build_fallback_briefing(fact_cards: list[dict], query: str) -> str:
    lines = [
        f"**Intelligence Briefing — {query}**\n",
        "_⚠️ LLM synthesis unavailable. Showing extracted article summaries._\n",
    ]
    for card in fact_cards:
        lines.append(
            f"\n**[{card['source_label']}] {card['title']}**\n"
            f"{card['facts']}\n"
        )
    lines.append(
        "\n---\n_To enable full AI synthesis, ensure Ollama is running or set GROQ_API_KEY._"
    )
    return "\n".join(lines)


# ── Main Navigator API ────────────────────────────────────────────────────────

def run_navigator_briefing(
    query: str,
    persona_key: str,
    verticals: list[str] | None = None,
    days_back: int | None = 7,
    conversation: ConversationManager | None = None,
) -> dict:
    """
    Full News Navigator pipeline.
    MAP is now parallel — all articles processed concurrently.
    """
    # Step 1: Retrieve candidates
    candidates = semantic_search(
        query_text=query,
        top_k=LANCEDB_TOP_K_NAVIGATOR,
        verticals=verticals,
        days_back=days_back,
    )

    if not candidates:
        return {
            "briefing": (
                "**No articles found** in the knowledge base for your query.\n\n"
                "Click **Load Demo** or **Fetch Today's News** in the sidebar first."
            ),
            "sources": [], "article_count": 0,
            "follow_up_questions": [], "llm_available": True,
        }

    logger.info("Navigator: %d candidates for '%s'.", len(candidates), query[:60])

    # Step 2: Parallel MAP — extract facts from all articles at once
    fact_cards = _run_map_parallel(candidates, query)

    # Step 3: REDUCE — synthesise into coherent briefing
    conversation_ctx = conversation.get_context_string() if conversation else ""
    briefing, llm_ok = _reduce_to_briefing(
        fact_cards=fact_cards,
        query=query,
        persona_key=persona_key,
        conversation_context=conversation_ctx,
    )

    # Step 4: Extract follow-up questions
    follow_ups = _extract_follow_up_questions(briefing)

    # Step 5: Build source cards
    sources = [
        {
            "label":     card["source_label"],
            "title":     card["title"],
            "url":       card["url"],
            "vertical":  card["vertical"],
            "sentiment": card["sentiment"],
            "published": card["published_at"][:10] if card["published_at"] else "",
        }
        for card in fact_cards
    ]

    return {
        "briefing":            briefing,
        "sources":             sources,
        "article_count":       len(fact_cards),
        "follow_up_questions": follow_ups,
        "llm_available":       llm_ok,
    }


def stream_navigator_response(
    query: str,
    persona_key: str,
    verticals: list[str] | None = None,
    days_back: int | None = 7,
    conversation: ConversationManager | None = None,
) -> Generator[str, None, None]:
    """Streaming version — MAP runs parallel, then REDUCE streams tokens."""
    candidates = semantic_search(
        query_text=query,
        top_k=LANCEDB_TOP_K_NAVIGATOR,
        verticals=verticals,
        days_back=days_back,
    )
    if not candidates:
        yield "No articles found. Please ingest news first."
        return

    yield f"_Found {len(candidates)} relevant articles. Extracting facts in parallel…_\n\n"

    fact_cards = _run_map_parallel(candidates, query)
    for card in fact_cards:
        yield f"✅ Processed: **{card['title'][:70]}**\n\n"

    yield "\n---\n\n**Synthesising your intelligence briefing…**\n\n"

    persona   = PERSONAS.get(persona_key, PERSONAS["Mutual Fund Investor"])
    style     = persona.get("summary_style", "professional")
    chars_per = max(400, (MAX_CONTEXT_TOKENS * 4) // max(len(fact_cards), 1) - 200)

    facts_parts, source_list_parts = [], []
    for card in fact_cards:
        facts_parts.append(
            f"[{card['source_label']}] {card['title']}\n"
            f"{card['facts'][:chars_per]}"
        )
        source_list_parts.append(f"{card['source_label']}: {card['title']}")

    ctx    = conversation.get_context_string() if conversation else ""
    prompt = build_navigator_reduce_prompt(
        "\n\n---\n\n".join(facts_parts) + (f"\n\nHISTORY:\n{ctx[:500]}" if ctx else ""),
        query, style, "\n".join(source_list_parts),
    )

    try:
        for token in generate_stream(prompt, system=SYSTEM_NEWS_ANALYST):
            yield token
    except RuntimeError as e:
        yield f"\n\n⚠️ **LLM Error:** {e}"


def _extract_follow_up_questions(briefing_text: str) -> list[str]:
    """Parse up to 3 follow-up questions from the briefing output."""
    lines     = briefing_text.split("\n")
    questions = []
    capturing = False

    for line in lines:
        stripped = line.strip()
        if not capturing and any(
            kw in stripped.lower()
            for kw in ("follow-up", "follow up", "suggested question", "questions:")
        ):
            capturing = True
            continue

        if capturing and stripped:
            if stripped[0].isdigit() and len(stripped) > 2 and stripped[1] in ".):":
                q = stripped[2:].strip().lstrip(" -")
                if q and len(q) > 10:
                    questions.append(q)
            elif "?" in stripped and len(stripped) > 15:
                questions.append(stripped)

        if len(questions) >= 3:
            break

    return questions[:3]
