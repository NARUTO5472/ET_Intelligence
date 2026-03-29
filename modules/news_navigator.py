"""
modules/news_navigator.py
News Navigator — Interactive Intelligence Briefings

Implements Advanced RAG with:
  - Parent-Child document retrieval
  - Map-Reduce summarisation (memory-safe for 16 GB RAM / CPU-only)
  - Chain-of-Thought synthesis across multiple articles
  - Conversational context window (last 3-5 turns)

CPU-inference budget per run:
  MAP:    LANCEDB_TOP_K_NAVIGATOR(4) × _MAP_MAX_TOKENS(250) ≈ 4 × ~20s = ~80s
  REDUCE: _REDUCE_MAX_TOKENS(400)                           ≈ ~30s
  Total:  ~110s  ← comfortably inside OLLAMA_TIMEOUT(300s)
"""

from __future__ import annotations
import logging
from typing import Generator

from config import (
    PERSONAS, MAP_REDUCE_CHUNK_ARTICLES,
    LANCEDB_TOP_K_NAVIGATOR, MAX_CONTEXT_TOKENS,
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
    """
    Sliding-window conversation history.
    Maintains last N turns and produces a context string for prompt injection.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content[:600]})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[2:]     # drop oldest turn pair

    def get_context_string(self) -> str:
        if not self.history:
            return ""
        lines = []
        for turn in self.history[-6:]:   # last 3 complete turns
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history.clear()

    def is_empty(self) -> bool:
        return len(self.history) == 0


# ── Map Phase — per-article fact extraction ───────────────────────────────────

def _map_article_to_facts(article: dict, query: str, source_index: int) -> dict:
    """
    Extracts 3-5 core facts from a single article relevant to the query.
    Uses MAP_MAX_TOKENS to stay fast on CPU (≈ 250 tokens ≈ 17-25 s).
    Falls back to a truncated summary on LLM failure so the pipeline
    always continues rather than aborting.
    """
    # Truncate article text — MAP prompt doesn't need the full article
    text    = (article.get("full_text") or article.get("summary", ""))[:1200]
    title   = article.get("title", f"Article {source_index}")
    url     = article.get("url", "#")
    source  = f"Source {source_index}"

    prompt  = build_navigator_map_prompt(text, query)

    facts, ok = generate_with_fallback(
        prompt,
        fallback_text=f"• {article.get('summary', 'No summary available.')[:300]}",
        system=SYSTEM_NEWS_ANALYST,
        temperature=0.1,
        max_tokens=MAP_MAX_TOKENS,
    )
    if not ok:
        logger.warning("MAP step LLM unavailable for article %d — using summary fallback.", source_index)

    return {
        "source_label": source,
        "title":        title,
        "url":          url,
        "facts":        facts,
        "vertical":     article.get("vertical", ""),
        "published_at": article.get("published_at", ""),
        "sentiment":    article.get("sentiment_label", "neutral"),
    }


# ── Reduce Phase — multi-article synthesis ────────────────────────────────────

def _reduce_to_briefing(
    fact_cards: list[dict],
    query: str,
    persona_key: str,
    conversation_context: str = "",
) -> tuple[str, bool]:
    """
    Combines fact cards from all articles into a single unified briefing.
    Returns (briefing_text, llm_success_flag).
    """
    persona = PERSONAS.get(persona_key, PERSONAS["Mutual Fund Investor"])
    style   = persona.get("summary_style", "professional and concise")

    # Allocate chars per source so the combined prompt stays under budget.
    # Budget: MAX_CONTEXT_TOKENS * 4 chars, split across all sources.
    chars_per_source = max(
        300,
        (MAX_CONTEXT_TOKENS * 4) // max(len(fact_cards), 1) - 200,
    )

    facts_parts = []
    source_list_parts = []

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

    # Inject conversation context if available (keep it short)
    if conversation_context:
        facts_combined += (
            f"\n\nPREVIOUS CONTEXT (do not repeat):\n{conversation_context[:400]}"
        )

    prompt = build_navigator_reduce_prompt(
        facts_combined=facts_combined,
        query=query,
        persona_style=style,
        source_list=source_list,
    )

    # Use generate_with_fallback so a timeout produces a graceful error card
    # instead of an unhandled exception / traceback in the UI.
    briefing, ok = generate_with_fallback(
        prompt,
        fallback_text=_build_fallback_briefing(fact_cards, query),
        system=SYSTEM_NEWS_ANALYST,
        temperature=0.25,
        max_tokens=REDUCE_MAX_TOKENS,
    )
    return briefing, ok


def _build_fallback_briefing(fact_cards: list[dict], query: str) -> str:
    """
    Produces a readable (non-LLM) summary when Ollama is unavailable / slow.
    Shown as a graceful degradation rather than a raw traceback.
    """
    lines = [
        f"**Intelligence Briefing — {query}**\n",
        "_⚠️ LLM synthesis unavailable (Ollama timeout). "
        "Showing extracted article summaries instead._\n",
    ]
    for card in fact_cards:
        lines.append(
            f"\n**[{card['source_label']}] {card['title']}**\n"
            f"{card['facts']}\n"
        )
    lines.append(
        "\n---\n_To enable full AI synthesis, ensure Ollama is running and "
        "the model is loaded: `ollama pull llama3.2:3b`_"
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
    Full News Navigator pipeline for a given query.

    Returns:
        {
          "briefing":             str,        # synthesised briefing
          "sources":              list[dict], # source cards
          "article_count":        int,
          "follow_up_questions":  list[str],
          "llm_available":        bool,       # False when Ollama timed out
        }
    """
    # ── Step 1: Retrieve candidate articles ──────────────────────────────────
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
                "Please click **Load Demo** or **Fetch Live** in the sidebar to "
                "populate the database, then try again."
            ),
            "sources": [],
            "article_count": 0,
            "follow_up_questions": [],
            "llm_available": True,
        }

    logger.info(
        "Navigator: %d candidate articles for query '%s'",
        len(candidates), query[:60],
    )

    # ── Step 2: MAP — extract facts from each article ────────────────────────
    # Process in chunks of MAP_REDUCE_CHUNK_ARTICLES to control memory.
    fact_cards: list[dict] = []
    source_index = 1

    for i in range(0, len(candidates), MAP_REDUCE_CHUNK_ARTICLES):
        chunk = candidates[i: i + MAP_REDUCE_CHUNK_ARTICLES]
        for article in chunk:
            card = _map_article_to_facts(article, query, source_index)
            fact_cards.append(card)
            source_index += 1

    # ── Step 3: REDUCE — synthesise into coherent briefing ───────────────────
    conversation_ctx = conversation.get_context_string() if conversation else ""

    briefing, llm_ok = _reduce_to_briefing(
        fact_cards=fact_cards,
        query=query,
        persona_key=persona_key,
        conversation_context=conversation_ctx,
    )

    # ── Step 4: Extract follow-up questions ──────────────────────────────────
    follow_ups = _extract_follow_up_questions(briefing)

    # ── Step 5: Build source cards for UI ────────────────────────────────────
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
    """
    Streaming version of the briefing for real-time Streamlit output.
    Streams only the reduce phase; the map phase runs silently first.
    """
    candidates = semantic_search(
        query_text=query,
        top_k=LANCEDB_TOP_K_NAVIGATOR,
        verticals=verticals,
        days_back=days_back,
    )

    if not candidates:
        yield "No articles found. Please ingest news first."
        return

    yield f"_Found {len(candidates)} relevant articles. Extracting key facts…_\n\n"

    fact_cards = []
    for i, article in enumerate(candidates, 1):
        card = _map_article_to_facts(article, query, i)
        fact_cards.append(card)
        yield f"✅ Processed: **{article.get('title', 'Article')[:70]}**\n\n"

    yield "\n---\n\n**Synthesising your intelligence briefing…**\n\n"

    persona   = PERSONAS.get(persona_key, PERSONAS["Mutual Fund Investor"])
    style     = persona.get("summary_style", "professional")
    chars_per = max(300, (MAX_CONTEXT_TOKENS * 4) // max(len(fact_cards), 1) - 200)

    facts_parts, source_list_parts = [], []
    for card in fact_cards:
        facts_parts.append(
            f"[{card['source_label']}] {card['title']}\n"
            f"{card['facts'][:chars_per]}"
        )
        source_list_parts.append(f"{card['source_label']}: {card['title']}")

    ctx = conversation.get_context_string() if conversation else ""
    prompt = build_navigator_reduce_prompt(
        "\n\n---\n\n".join(facts_parts) + (f"\n\nHISTORY:\n{ctx[:400]}" if ctx else ""),
        query, style,
        "\n".join(source_list_parts),
    )

    try:
        for token in generate_stream(prompt, system=SYSTEM_NEWS_ANALYST):
            yield token
    except RuntimeError as e:
        yield f"\n\n⚠️ **LLM Error:** {e}"
    except Exception as e:
        yield f"\n\n⚠️ **Unexpected error during streaming:** {e}"


def _extract_follow_up_questions(briefing_text: str) -> list[str]:
    """Parse up to 3 follow-up questions from the briefing output."""
    lines      = briefing_text.split("\n")
    questions  = []
    capturing  = False

    for line in lines:
        stripped = line.strip()
        # Detect the follow-up section header (various phrasings)
        if not capturing and any(
            kw in stripped.lower()
            for kw in ("follow-up", "follow up", "suggested question", "questions:")
        ):
            capturing = True
            continue

        if capturing and stripped:
            # Match numbered items:  "1. ..."  "2) ..."
            if stripped[0].isdigit() and len(stripped) > 2 and stripped[1] in ".):":
                q = stripped[2:].strip().lstrip(" -")
                if q and len(q) > 10:
                    questions.append(q)
            elif "?" in stripped and len(stripped) > 15:
                # Un-numbered question line
                questions.append(stripped)

        if len(questions) >= 3:
            break

    return questions[:3]
