"""
llm/ollama_client.py  (UPGRADED)
Unified LLM client: Groq (primary, 1,200+ tok/s) → Ollama (fallback, CPU).

Why Groq?
  llama3.2:3b on CPU  → 10-15 tok/s  → 400-token response ≈ 30-40s
  llama-3.1-8b on Groq → 1,200 tok/s → 800-token response ≈ 0.7s
  Net result: Navigator briefing drops from ~8 min to ~15-20 seconds.

Set up:  export GROQ_API_KEY="gsk_..."   (free at console.groq.com)
No key?  Falls back silently to local Ollama — zero code changes needed.
"""

from __future__ import annotations
import time
import logging
from typing import Generator, Optional

import requests

from config import (
    GROQ_API_KEY, GROQ_MODEL, USE_GROQ,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, MAX_CONTEXT_TOKENS,
)

logger = logging.getLogger(__name__)

# ── Groq client (lazy-init) ───────────────────────────────────────────────────
_groq_client = None

def _get_groq():
    global _groq_client
    if _groq_client is None and USE_GROQ:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialised (model: %s).", GROQ_MODEL)
        except ImportError:
            logger.warning(
                "groq package not installed. Run: pip install groq\n"
                "Falling back to Ollama."
            )
    return _groq_client


# ── Per-call token budgets ────────────────────────────────────────────────────
# Groq handles 2× the tokens in a fraction of the time, so we raise the limits.
_MAP_MAX_TOKENS    = 300    # per-article fact extraction
_REDUCE_MAX_TOKENS = 800    # full briefing synthesis
_SUMMARY_MAX_TOKENS= 250    # persona / ELI5 one-shot
_ARC_MAX_TOKENS    = 500    # arc evolution briefing
_WATCH_MAX_TOKENS  = 250    # "what to watch" signals

# Expose for importers
MAP_MAX_TOKENS     = _MAP_MAX_TOKENS
REDUCE_MAX_TOKENS  = _REDUCE_MAX_TOKENS
SUMMARY_MAX_TOKENS = _SUMMARY_MAX_TOKENS
ARC_MAX_TOKENS     = _ARC_MAX_TOKENS
WATCH_MAX_TOKENS   = _WATCH_MAX_TOKENS


def _truncate_to_token_budget(text: str, budget: int = MAX_CONTEXT_TOKENS) -> str:
    char_limit = budget * 4
    if len(text) > char_limit:
        logger.warning("Context truncated %d → %d chars.", len(text), char_limit)
        return text[:char_limit] + "\n\n[... truncated ...]"
    return text


# ── Groq generation ───────────────────────────────────────────────────────────

def _groq_generate(
    prompt: str,
    system: str = "",
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    client = _get_groq()
    if client is None:
        raise RuntimeError("Groq client not available.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - t0
    tokens  = resp.usage.completion_tokens if resp.usage else 0
    logger.info(
        "Groq: %d tokens in %.2fs (%.0f tok/s).",
        tokens, elapsed, tokens / max(elapsed, 0.001),
    )
    return resp.choices[0].message.content.strip()


# ── Ollama generation ─────────────────────────────────────────────────────────

def check_ollama_alive() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def list_available_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def check_groq_alive() -> bool:
    """Returns True if Groq API key is configured and client is importable."""
    return USE_GROQ and _get_groq() is not None


def _ollama_generate(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    if not check_ollama_alive():
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve\n"
            "Then pull the model: ollama pull llama3.2:3b"
        )

    prompt  = _truncate_to_token_budget(prompt)
    payload = {
        "model":   model,
        "prompt":  prompt,
        "system":  system,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx":     min(MAX_CONTEXT_TOKENS + 512, 4096),
            "num_thread":  8,   # use more threads — we have headroom
        },
    }

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
        "Ollama: %d tokens in %.1fs (%.1f tok/s).",
        tokens, elapsed, tokens / max(elapsed, 0.01),
    )
    return result


# ── Unified generate() — Groq first, Ollama fallback ─────────────────────────

def generate(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    stream: bool = False,
) -> str:
    """
    Generate text via Groq (if key set) or Ollama (CPU fallback).
    Groq is ~100× faster on the same prompts.
    """
    # ── Try Groq first ────────────────────────────────────────────────────────
    if USE_GROQ:
        try:
            return _groq_generate(
                prompt, system=system,
                max_tokens=max_tokens, temperature=temperature,
            )
        except Exception as groq_err:
            logger.warning("Groq failed (%s) — trying Ollama fallback.", groq_err)

    # ── Ollama fallback ───────────────────────────────────────────────────────
    return _ollama_generate(
        prompt, system=system, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )


def generate_with_fallback(
    prompt: str,
    fallback_text: str = "",
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> tuple[str, bool]:
    """
    Like generate() but never raises — returns (text, success_flag).
    Safe to call from UI code.
    """
    try:
        return generate(
            prompt, system=system, model=model,
            temperature=temperature, max_tokens=max_tokens,
        ), True
    except Exception as e:
        logger.warning("generate_with_fallback: LLM unavailable — %s", e)
        return fallback_text, False


def generate_stream(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> Generator[str, None, None]:
    """
    Streaming generation for Streamlit st.write_stream.
    Uses Groq streaming if available, else Ollama streaming.
    """
    # ── Groq streaming ────────────────────────────────────────────────────────
    if USE_GROQ:
        try:
            client = _get_groq()
            if client:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": _truncate_to_token_budget(prompt)})
                with client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    max_tokens=_REDUCE_MAX_TOKENS,
                    temperature=temperature,
                    stream=True,
                ) as stream:
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                return
        except Exception as e:
            logger.warning("Groq streaming failed (%s) — falling back to Ollama.", e)

    # ── Ollama streaming fallback ─────────────────────────────────────────────
    prompt  = _truncate_to_token_budget(prompt)
    payload = {
        "model":   model,
        "prompt":  prompt,
        "system":  system,
        "stream":  True,
        "options": {
            "temperature": temperature,
            "num_predict": _REDUCE_MAX_TOKENS,
            "num_ctx":     min(MAX_CONTEXT_TOKENS + 512, 4096),
            "num_thread":  8,
        },
    }
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload, stream=True, timeout=OLLAMA_TIMEOUT,
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


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_NEWS_ANALYST = (
    "You are a senior business journalist at the Economic Times. "
    "Provide factual, grounded analysis based ONLY on the context provided. "
    "Never fabricate facts. Cite sources as [Source N]. Be concise."
)


# ── Prompt Templates ──────────────────────────────────────────────────────────

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
        "SUMMARY:"
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
        "Write a structured briefing using Chain-of-Thought reasoning:\n"
        "**CORE NARRATIVE** (1 para)\n"
        "**KEY EVIDENCE** (2-3 bullets)\n"
        "**IMPLICATIONS** for the reader\n"
        "**WHAT TO WATCH** (3 signals)\n\n"
        "End with:\nFollow-up questions:\n1. ...\n2. ...\n3. ..."
    )


def build_arc_evolution_prompt(events_timeline: str, topic: str) -> str:
    return (
        f"Analyse this story arc about \"{topic}\".\n\n"
        f"EVENTS:\n{events_timeline}\n\n"
        "Write a brief evolution briefing (max 400 words):\n"
        "**ORIGIN** — how it started\n"
        "**KEY TURNING POINTS** — 2 pivotal developments\n"
        "**CURRENT STATE** — where we are now\n"
        "**WHAT TO WATCH NEXT** — 3 specific signals"
    )


def build_what_to_watch_prompt(arc_summary: str, entity_list: str) -> str:
    return (
        f"Based on this story and key players ({entity_list}), "
        "list 3 specific 'Watch for...' signals.\n\n"
        f"STORY: {arc_summary[:800]}\n\n"
        "Format: numbered list, bold signal + 1-sentence explanation. "
        "Be specific, not vague."
    )
