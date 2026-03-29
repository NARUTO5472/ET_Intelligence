"""
vector_store/lancedb_manager.py
LanceDB interface — disk-based IVF-PQ vector index + metadata filtering.
Designed for larger-than-RAM corpora on a 16 GB / CPU-only system.
"""

from __future__ import annotations
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pyarrow as pa

import lancedb

from config import (
    LANCE_DIR, LANCEDB_TABLE, EMBEDDING_DIM,
    LANCEDB_TOP_K_FIRST_PASS, LANCEDB_TOP_K_NAVIGATOR, LANCEDB_TOP_K_STORY,
)
from ingestion.article_processor import embed_single

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────
ARTICLE_SCHEMA = pa.schema([
    pa.field("id",                   pa.string()),
    pa.field("url",                  pa.string()),
    pa.field("title",                pa.string()),
    pa.field("summary",              pa.string()),
    pa.field("full_text",            pa.string()),
    pa.field("lead",                 pa.string()),
    pa.field("published_at",         pa.string()),
    pa.field("vertical",             pa.string()),
    pa.field("source",               pa.string()),
    pa.field("word_count",           pa.int32()),
    pa.field("entities",             pa.list_(pa.string())),
    pa.field("sentiment_compound",   pa.float32()),
    pa.field("sentiment_label",      pa.string()),
    pa.field("readability_score",    pa.float32()),
    pa.field("chunk_count",          pa.int32()),
    pa.field("processed_at",         pa.string()),
    pa.field("vector",               pa.list_(pa.float32(), EMBEDDING_DIM)),
])

_db    = None
_table = None


def _get_db():
    global _db
    if _db is None:
        _db = lancedb.connect(str(LANCE_DIR))
    return _db


def get_table():
    global _table
    if _table is None:
        db = _get_db()
        if LANCEDB_TABLE in db.table_names():
            _table = db.open_table(LANCEDB_TABLE)
            logger.info("Opened existing LanceDB table '%s' (%d rows).", LANCEDB_TABLE, len(_table))
        else:
            _table = db.create_table(LANCEDB_TABLE, schema=ARTICLE_SCHEMA)
            logger.info("Created new LanceDB table '%s'.", LANCEDB_TABLE)
    return _table


def _safe_str(val, default="") -> str:
    return str(val) if val is not None else default


def _safe_list(val) -> list:
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


def upsert_articles(processed_articles: list[dict]) -> int:
    """
    Upsert enriched article dicts into LanceDB.
    Skips articles with missing or malformed vectors.
    Returns count of successfully upserted records.
    """
    if not processed_articles:
        return 0

    table  = get_table()
    rows   = []
    errors = 0

    for art in processed_articles:
        vec = art.get("vector")
        if not vec or len(vec) != EMBEDDING_DIM:
            logger.warning("Skipping article '%s' — invalid vector.", art.get("id"))
            errors += 1
            continue

        rows.append({
            "id":                  _safe_str(art.get("id")),
            "url":                 _safe_str(art.get("url")),
            "title":               _safe_str(art.get("title")),
            "summary":             _safe_str(art.get("summary"))[:1000],
            "full_text":           _safe_str(art.get("full_text"))[:6000],
            "lead":                _safe_str(art.get("lead"))[:400],
            "published_at":        _safe_str(art.get("published_at")),
            "vertical":            _safe_str(art.get("vertical")),
            "source":              _safe_str(art.get("source"), "Economic Times"),
            "word_count":          int(art.get("word_count", 0)),
            "entities":            _safe_list(art.get("entities", [])),
            "sentiment_compound":  float(art.get("sentiment_compound", 0.0)),
            "sentiment_label":     _safe_str(art.get("sentiment_label"), "neutral"),
            "readability_score":   float(art.get("readability_score", 50.0)),
            "chunk_count":         int(art.get("chunk_count", 1)),
            "processed_at":        _safe_str(art.get("processed_at")),
            "vector":              [float(v) for v in vec],
        })

    if rows:
        try:
            table.add(rows)
            logger.info("Upserted %d articles into LanceDB (%d errors).", len(rows), errors)
        except Exception as e:
            logger.error("LanceDB upsert failed: %s", e)
            raise

    return len(rows)


def article_count() -> int:
    try:
        return len(get_table())
    except Exception:
        return 0


def id_exists(article_id: str) -> bool:
    try:
        results = get_table().search().where(f"id = '{article_id}'").limit(1).to_list()
        return len(results) > 0
    except Exception:
        return False


# ── Vector Search ─────────────────────────────────────────────────────────────

def semantic_search(
    query_text: str,
    top_k: int = LANCEDB_TOP_K_FIRST_PASS,
    verticals: Optional[list[str]] = None,
    days_back: Optional[int] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> list[dict]:
    """
    First-pass vector retrieval with optional metadata filtering.
    Returns list of article dicts sorted by cosine similarity (descending).
    """
    table = get_table()
    if len(table) == 0:
        logger.warning("LanceDB table is empty — no results.")
        return []

    query_vec = embed_single(query_text)

    search = table.search(query_vec.tolist()).limit(top_k * 2)  # Over-fetch then filter

    # Metadata filters
    filters = []

    if verticals:
        vert_quoted = ", ".join(f"'{v}'" for v in verticals)
        filters.append(f"vertical IN ({vert_quoted})")

    if days_back is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        filters.append(f"published_at >= '{cutoff}'")

    if min_date:
        filters.append(f"published_at >= '{min_date}'")

    if max_date:
        filters.append(f"published_at <= '{max_date}'")

    if filters:
        where_clause = " AND ".join(filters)
        try:
            search = search.where(where_clause)
        except Exception as e:
            logger.warning("Filter failed (%s), running unfiltered search.", e)

    try:
        results = search.to_list()
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return []

    # Convert to plain dicts and deduplicate by ID
    seen_ids = set()
    articles = []
    for r in results:
        rid = r.get("id", "")
        if rid not in seen_ids:
            seen_ids.add(rid)
            articles.append(dict(r))

    return articles[:top_k]


def get_recent_articles(
    n: int = 20,
    vertical: Optional[str] = None,
    days_back: int = 7,
) -> list[dict]:
    """
    Fetch the most recently published N articles (no vector search).
    Used for the Story Arc article feed and homepage.
    """
    table = get_table()
    if len(table) == 0:
        return []

    try:
        scanner = table.search().limit(n * 3)

        filters = []
        if vertical:
            filters.append(f"vertical = '{vertical}'")

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        filters.append(f"published_at >= '{cutoff}'")

        if filters:
            scanner = scanner.where(" AND ".join(filters))

        rows = scanner.to_list()
        rows.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return [dict(r) for r in rows[:n]]
    except Exception as e:
        logger.error("get_recent_articles failed: %s", e)
        return []


def get_all_articles_for_arc(days_back: int = 60) -> list[dict]:
    """
    Fetch all articles within window for Story Arc construction.
    Uses simple table scan — acceptable for small corpora.
    """
    table = get_table()
    if len(table) == 0:
        return []

    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        rows = table.search().where(f"published_at >= '{cutoff}'").limit(2000).to_list()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error("get_all_articles_for_arc failed: %s", e)
        return []


def get_article_by_id(article_id: str) -> Optional[dict]:
    try:
        results = get_table().search().where(f"id = '{article_id}'").limit(1).to_list()
        return dict(results[0]) if results else None
    except Exception:
        return None
