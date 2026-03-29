"""
utils/ingestion_orchestrator.py
Ties together RSS fetching → article processing → LanceDB upsert.
Also provides the APScheduler-based background polling loop.
"""

from __future__ import annotations
import logging
from typing import Optional

from config import RSS_POLL_INTERVAL_MINUTES
from ingestion.rss_fetcher import fetch_new_articles, get_mock_articles
from ingestion.article_processor import process_article
from vector_store.lancedb_manager import upsert_articles, id_exists, article_count

logger = logging.getLogger(__name__)


def ingest_articles(
    use_mock: bool = False,
    verticals: Optional[list[str]] = None,
    force_refresh: bool = False,
    progress_callback=None,
) -> dict:
    """
    Full ingestion cycle:
      1. Fetch raw articles (RSS or mock)
      2. Process each: embed + NER + sentiment
      3. Upsert into LanceDB

    Args:
        use_mock:          Use mock articles (for offline / demo mode)
        verticals:         Filter to specific ET verticals
        force_refresh:     Re-process even if URL was seen before
        progress_callback: Optional callable(step: str, pct: float) for UI updates

    Returns:
        {"fetched": N, "processed": N, "upserted": N, "errors": N}
    """
    def _cb(step, pct):
        if progress_callback:
            progress_callback(step, pct)

    _cb("Fetching articles from RSS feeds…", 0.05)

    if use_mock:
        raw_articles = get_mock_articles()
        logger.info("Using %d mock articles.", len(raw_articles))
    else:
        raw_articles = fetch_new_articles(verticals=verticals, force_refresh=force_refresh)

    if not raw_articles:
        logger.info("No new articles to ingest.")
        return {"fetched": 0, "processed": 0, "upserted": 0, "errors": 0}

    _cb(f"Processing {len(raw_articles)} articles…", 0.2)

    processed = []
    errors     = 0
    for i, article in enumerate(raw_articles):
        # Skip already-indexed articles (unless force refresh)
        if not force_refresh and id_exists(article["id"]):
            continue
        try:
            enriched = process_article(article)
            processed.append(enriched)
        except Exception as e:
            logger.warning("Processing failed for article '%s': %s", article.get("title", "?"), e)
            errors += 1

        if i % 5 == 0:
            pct = 0.2 + (i / len(raw_articles)) * 0.5
            _cb(f"Processed {i+1}/{len(raw_articles)} articles…", pct)

    _cb("Upserting vectors into LanceDB…", 0.75)

    upserted = 0
    if processed:
        try:
            upserted = upsert_articles(processed)
        except Exception as e:
            logger.error("Upsert failed: %s", e)
            errors += len(processed)

    _cb("Ingestion complete.", 1.0)

    result = {
        "fetched":   len(raw_articles),
        "processed": len(processed),
        "upserted":  upserted,
        "errors":    errors,
        "total_in_db": article_count(),
    }
    logger.info("Ingestion result: %s", result)
    return result


def start_background_scheduler():
    """
    Starts a background APScheduler job to poll ET RSS every N minutes.
    Call once at app startup. Safe to call multiple times (checks for existing scheduler).
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            func=lambda: ingest_articles(use_mock=False),
            trigger="interval",
            minutes=RSS_POLL_INTERVAL_MINUTES,
            id="et_rss_poller",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Background RSS poller started (every %d min).", RSS_POLL_INTERVAL_MINUTES)
        return scheduler
    except Exception as e:
        logger.warning("Background scheduler failed to start: %s", e)
        return None
