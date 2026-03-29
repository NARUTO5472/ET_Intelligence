"""
ingestion/article_processor.py
Semantic processing pipeline: embedding generation, NER extraction,
sentiment scoring, and Parent-Child document chunking.
"""

from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import numpy as np

from config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH,
    ARTICLE_CHUNK_SIZE, ARTICLE_CHUNK_OVERLAP,
    GLINER_MODEL, NER_ENTITY_TYPES, NER_THRESHOLD,
    EMBEDDING_CACHE_SIZE,
)

logger = logging.getLogger(__name__)

# ── Lazy-loaded models (prevent import-time RAM spike) ───────────────────────

_embedder = None
_gliner    = None
_vader     = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Loading SentenceTransformer: %s …", EMBEDDING_MODEL)
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedder


def _get_gliner():
    global _gliner
    if _gliner is None:
        try:
            logger.info("Loading GLiNER NER model: %s …", GLINER_MODEL)
            from gliner import GLiNER
            _gliner = GLiNER.from_pretrained(GLINER_MODEL)
            logger.info("GLiNER loaded.")
        except Exception as e:
            logger.warning("GLiNER failed to load (%s) — falling back to SpaCy NER.", e)
            import spacy
            try:
                _gliner = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("SpaCy model not found. Run: python -m spacy download en_core_web_sm")
                _gliner = None
    return _gliner


def _get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


# ── Embedding ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
def embed_text_cached(text: str) -> tuple:
    """LRU-cached embedding for repeated queries (saves CPU cycles)."""
    emb = _get_embedder().encode(text, normalize_embeddings=True)
    return tuple(emb.tolist())


def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch-embed a list of texts. Returns (N, 384) float32 array."""
    model = _get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def embed_single(text: str) -> np.ndarray:
    """Single-text embedding with caching."""
    cached = embed_text_cached(text[:512])  # Cache on truncated text for efficiency
    return np.array(cached, dtype="float32")


# ── Text Chunking (Parent-Child pattern) ─────────────────────────────────────

def chunk_text(text: str, chunk_size: int = ARTICLE_CHUNK_SIZE, overlap: int = ARTICLE_CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping semantic chunks.
    Respects sentence boundaries where possible.
    """
    # Sentence split on period/newline boundaries
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N chars worth of sentences
            overlap_text = " ".join(current)[-overlap:]
            current = [overlap_text]
            current_len = len(overlap_text)
        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c) > 50]


# ── Named Entity Recognition ──────────────────────────────────────────────────

def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extracts named entities using GLiNER (preferred) or SpaCy (fallback).
    Returns dict mapping entity_type → list of entity strings.
    """
    model = _get_gliner()
    if model is None:
        return {}

    text_input = text[:2000]   # NER on first 2k chars (covers most entities)

    try:
        # GLiNER interface
        if hasattr(model, "predict_entities"):
            predictions = model.predict_entities(text_input, NER_ENTITY_TYPES, threshold=NER_THRESHOLD)
            result: dict[str, list[str]] = {}
            for pred in predictions:
                etype  = pred["label"]
                entity = pred["text"].strip()
                if len(entity) > 2:
                    result.setdefault(etype, [])
                    if entity not in result[etype]:
                        result[etype].append(entity)
            return result
        else:
            # SpaCy fallback
            doc = model(text_input)
            result = {}
            spacy_map = {
                "ORG": "organization", "PERSON": "person",
                "GPE": "location", "LOC": "location",
                "PRODUCT": "product", "EVENT": "event",
                "LAW": "regulation", "MONEY": "financial_instrument",
            }
            for ent in doc.ents:
                etype = spacy_map.get(ent.label_, ent.label_.lower())
                result.setdefault(etype, [])
                if ent.text.strip() not in result[etype]:
                    result[etype].append(ent.text.strip())
            return result
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)
        return {}


def flatten_entities(entity_dict: dict) -> list[str]:
    """Returns a flat list of all entity strings."""
    all_entities = []
    for entities in entity_dict.values():
        all_entities.extend(entities)
    return list(set(all_entities))


# ── Sentiment Analysis ────────────────────────────────────────────────────────

def compute_sentiment(text: str) -> dict:
    """
    Returns VADER compound score + label.
    compound ∈ [-1, 1]. Label: positive / negative / neutral.
    """
    vader = _get_vader()
    scores = vader.polarity_scores(text[:1000])
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"compound": round(compound, 4), "label": label, **scores}


# ── Readability ───────────────────────────────────────────────────────────────

def compute_readability_score(text: str) -> float:
    """
    Simplified Flesch Reading Ease proxy.
    Higher score = easier to read. Used for student persona ranking.
    """
    words = text.split()
    if not words:
        return 50.0
    sentences = max(len(re.findall(r'[.!?]+', text)), 1)
    syllables  = sum(_count_syllables(w) for w in words[:200])  # Sample first 200 words
    asl = len(words) / sentences
    asw = syllables / max(len(words[:200]), 1)
    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return max(0.0, min(100.0, round(score, 1)))


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:")
    vowels = "aeiou"
    count = sum(1 for i, c in enumerate(word) if c in vowels and (i == 0 or word[i-1] not in vowels))
    return max(1, count)


# ── Master Processing Function ────────────────────────────────────────────────

def process_article(article: dict) -> dict:
    """
    Full processing pipeline for a raw article dict.
    Adds: embedding vector, entity dict, sentiment, readability, chunks.
    Returns enriched article dict ready for LanceDB ingestion.
    """
    text       = article.get("full_text") or article.get("summary", "")
    title      = article.get("title", "")
    embed_text = f"{title}. {text[:600]}"   # Embed title + lead for best retrieval

    # Embedding
    vector = embed_single(embed_text).tolist()

    # Entity extraction
    entity_dict = extract_entities(title + " " + text)
    entity_list = flatten_entities(entity_dict)

    # Sentiment
    sentiment = compute_sentiment(text)

    # Readability
    readability = compute_readability_score(text)

    # Child chunks for RAG
    chunks = chunk_text(text)

    return {
        **article,
        "vector":         vector,
        "entities":       entity_list[:30],   # Cap at 30 to limit storage
        "entity_dict":    entity_dict,
        "sentiment_compound": sentiment["compound"],
        "sentiment_label":    sentiment["label"],
        "readability_score":  readability,
        "chunk_count":        len(chunks),
        "chunks":             chunks,
        "processed_at":       datetime.now(timezone.utc).isoformat(),
    }
