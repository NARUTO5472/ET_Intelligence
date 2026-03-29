"""
llm/ollama_client.py
Thin, memory-safe wrapper around the local Ollama inference server.
Handles context-window budgeting, streaming, and error recovery.

CPU-inference tuning (16 GB RAM / no VRAM):
  • llama3.2:3b at q4_k_m → ~10-15 tok/s
  • 400 tokens output ≈ 27-40 s  ← safe default
  • 300 s global timeout gives generous headroom for any single call
"""

from __future__ import annotations
import time
import logging
from typing import Generator, Optional

import requests

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, MAX_CONTEXT_TOKENS
)

logger = logging.getLogger(__name__)

# ── CPU-speed constants ───────────────────────────────────────────────────────
# Keep every call clearly inside the timeout.  Values here are *maximums*;
# individual call-sites can pass smaller values via the max_tokens argument.
_MAP_MAX_TOKENS     = 250   # per-article fact extraction: bullet list → short
_REDUCE_MAX_TOKENS  = 400   # full briefing synthesis
_SUMMARY_MAX_TOKENS = 200   # persona / ELI5 one-shot summaries
_ARC_MAX_TOKENS     = 350   # arc evolution briefing
_WATCH_MAX_TOKENS   = 180   # "what to watch" 3-signal list


def _truncate_to_token_budget(text: str, budget: int = MAX_CONTEXT_TOKENS) -> str:
    """
    Rough token estimator (1 token ≈ 4 chars).
    Hard-truncates text to stay within the context budget, preventing RAM thrashing.
    """
    char_limit = budget * 4
    if len(text) > char_limit:
        logger.warning(
            "Context truncated from %d → %d chars to stay within budget.",
            len(text), char_limit,
        )
        return text[:char_limit] + "\n\n[... truncated for RAM efficiency ...]"
    return text


def check_ollama_alive() -> bool:
    """Ping the Ollama server.  Returns True if running."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def list_available_models() -> list[str]:
    """Return model names available in the local Ollama registry."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def generate(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    stream: bool = False,
) -> str:
    """
    Single-turn generation.  Returns the full response string.
    Raises RuntimeError with a helpful message when Ollama is unavailable
    or the request times out.

    The caller is responsible for passing a max_tokens value that is
    achievable within OLLAMA_TIMEOUT on CPU at ~10-15 tok/s.
    Rule of thumb:  max_tokens × 0.08 ≤ OLLAMA_TIMEOUT  (i.e. max ≈ 3 750)
    """
    if not check_ollama_alive():
        raise RuntimeError(
            "Ollama is not running. Start it with:\n"
            "  ollama serve\n"
            "Then pull the model with:\n"
            "  ollama pull llama3.2:3b"
        )

    prompt = _truncate_to_token_budget(prompt)

    payload = {
        "model":   model,
        "prompt":  prompt,
        "system":  system,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx":     MAX_CONTEXT_TOKENS + 512,   # slight headroom
            "num_thread":  6,                           # leave 2 threads for OS
        },
    }

    try:
        t0 = time.time()
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        elapsed = time.time() - t0
        result  = response.json().get("response", "").strip()
        tokens  = response.json().get("eval_count", 0)
        logger.info(
            "LLM generated %d tokens in %.1fs (%.1f tok/s)",
            tokens, elapsed, tokens / max(elapsed, 0.01),
        )
        return result

    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama timed out after {OLLAMA_TIMEOUT}s on CPU.\n"
            f"Tip: reduce max_tokens (current={max_tokens}) or use a shorter prompt."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama. Make sure it is running:\n"
            "  ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama generation failed: {e}")


def generate_with_fallback(
    prompt: str,
    fallback_text: str = "",
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> tuple[str, bool]:
    """
    Like generate(), but returns (text, success_flag) instead of raising.
    Falls back to fallback_text on any error — safe to call from UI code.

    Returns:
        (generated_text, True)  on success
        (fallback_text,  False) on any Ollama error
    """
    try:
        return generate(prompt, system=system, model=model,
                        temperature=temperature, max_tokens=max_tokens), True
    except RuntimeError as e:
        logger.warning("generate_with_fallback: LLM unavailable — %s", e)
        return fallback_text, False


def generate_stream(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> Generator[str, None, None]:
    """
    Streaming generation — yields token-by-token for Streamlit's st.write_stream.
    """
    prompt = _truncate_to_token_budget(prompt)

    payload = {
        "model":   model,
        "prompt":  prompt,
        "system":  system,
        "stream":  True,
        "options": {
            "temperature": temperature,
            "num_predict": _REDUCE_MAX_TOKENS,
            "num_ctx":     MAX_CONTEXT_TOKENS + 512,
            "num_thread":  6,
        },
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=OLLAMA_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        import json
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


# ── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_NEWS_ANALYST = (
    "You are a senior business journalist at the Economic Times. "
    "Provide factual, grounded analysis based ONLY on the context provided. "
    "Never fabricate facts. Cite sources as [Source N]. Be concise."
)


def build_eli5_prompt(article_text: str) -> str:
    return (
        "Explain this business news in 2 simple sentences for a curious 18-year-old. "
        "Use plain English and one helpful analogy. No jargon.\n\n"
        f"ARTICLE:\n{article_text[:800]}\n\n"
        "ELI5 (2 sentences):"
    )


def build_persona_summary_prompt(article_text: str, persona: str, style: str) -> str:
    return (
        f"Summarise for a {persona} in {style} style. "
        f"Focus only on what matters to a {persona}. Be specific and practical. 3 sentences max.\n\n"
        f"ARTICLE:\n{article_text[:1200]}\n\n"
        f"SUMMARY:"
    )


def build_navigator_map_prompt(article_text: str, query: str) -> str:
    return (
        f"Extract 3-5 key facts from this article that directly answer: {query}\n"
        "Output a numbered list. Be very concise — one line per fact. No filler.\n\n"
        f"ARTICLE:\n{article_text[:1200]}\n\n"
        "KEY FACTS:"
    )


def build_navigator_reduce_prompt(
    facts_combined: str, query: str, persona_style: str, source_list: str
) -> str:
    return (
        f"Synthesise these news facts into a briefing. Query: {query}\n"
        f"Style: {persona_style}\n\n"
        f"FACTS:\n{facts_combined}\n\n"
        f"SOURCES: {source_list}\n\n"
        "Write a structured briefing with:\n"
        "1. CORE NARRATIVE (1 para)\n"
        "2. KEY EVIDENCE (2-3 bullets)\n"
        "3. IMPLICATIONS for the reader\n"
        "4. WHAT TO WATCH\n\n"
        "End with:\nFollow-up questions:\n1. ...\n2. ...\n3. ..."
    )


def build_arc_evolution_prompt(events_timeline: str, topic: str) -> str:
    return (
        f"Analyse this story arc about \"{topic}\".\n\n"
        f"EVENTS:\n{events_timeline}\n\n"
        "Write a brief evolution briefing (max 300 words):\n"
        "**ORIGIN** — how it started\n"
        "**KEY TURNING POINTS** — 2 pivotal developments\n"
        "**CURRENT STATE** — where we are now\n"
        "**WHAT TO WATCH NEXT** — 3 specific signals"
    )


def build_what_to_watch_prompt(arc_summary: str, entity_list: str) -> str:
    return (
        f"Based on this story and key players ({entity_list}), "
        "list 3 specific 'Watch for...' signals.\n\n"
        f"STORY: {arc_summary[:600]}\n\n"
        "Format: numbered list, bold signal + 1-sentence explanation. "
        "Be specific, not vague."
    )


# Expose per-call token budgets for importers
MAP_MAX_TOKENS     = _MAP_MAX_TOKENS
REDUCE_MAX_TOKENS  = _REDUCE_MAX_TOKENS
SUMMARY_MAX_TOKENS = _SUMMARY_MAX_TOKENS
ARC_MAX_TOKENS     = _ARC_MAX_TOKENS
WATCH_MAX_TOKENS   = _WATCH_MAX_TOKENS
