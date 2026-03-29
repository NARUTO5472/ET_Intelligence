"""
utils/ingestion_orchestrator.py  (UPGRADED)
Smart incremental ingestion pipeline.

Key upgrades:
  • "Fetch Live" only ingests TODAY's articles (older ones stay in DB forever)
  • Ingestion log tracks last fetch date to avoid redundant RSS polling
  • "Load Demo" always loads all 20 mock articles
  • Parallel article processing using ThreadPoolExecutor
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import RSS_POLL_INTERVAL_MINUTES, CACHE_DIR, INGEST_TODAY_ONLY
from ingestion.rss_fetcher import fetch_new_articles, fetch_todays_articles, get_mock_articles
from ingestion.article_processor import process_article
from vector_store.lancedb_manager import upsert_articles, id_exists, article_count

logger = logging.getLogger(__name__)

_INGESTION_LOG = CACHE_DIR / "ingestion_log.json"


# ── Ingestion log ─────────────────────────────────────────────────────────────

def _load_log() -> dict:
    if _INGESTION_LOG.exists():
        try:
            return json.loads(_INGESTION_LOG.read_text())
        except Exception:
            pass
    return {}


def _save_log(log: dict) -> None:
    try:
        _INGESTION_LOG.write_text(json.dumps(log, indent=2))
    except Exception as e:
        logger.warning("Could not save ingestion log: %s", e)


def already_fetched_today() -> bool:
    """Returns True if we already ran a live fetch today (avoids duplicate RSS polls)."""
    log       = _load_log()
    last_date = log.get("last_live_fetch_date", "")
    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return last_date == today


def mark_fetched_today() -> None:
    log = _load_log()
    log["last_live_fetch_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log["last_live_fetch_ts"]   = datetime.now(timezone.utc).isoformat()
    _save_log(log)


# ── Parallel article processing ───────────────────────────────────────────────

def _process_one(article: dict) -> dict | None:
    """Process a single article — runs in a thread-pool worker."""
    try:
        return process_article(article)
    except Exception as e:
        logger.warning(
            "Processing failed for '%s': %s",
            article.get("title", "?")[:60], e,
        )
        return None


def _process_articles_parallel(raw_articles: list[dict], force_refresh: bool = False) -> list[dict]:
    """
    Embed + NER + sentiment all articles in parallel.
    Uses up to 4 workers (RAM-safe for 16 GB).
    Skips articles already in the database (unless force_refresh=True).
    """
    to_process = [
        a for a in raw_articles
        if force_refresh or not id_exists(a["id"])
    ]
    if not to_process:
        logger.info("All articles already indexed — nothing to process.")
        return []

    logger.info("Processing %d articles in parallel…", len(to_process))
    processed = []
    # 4 workers: embedding model is thread-safe; GLiNER may not be,
    # so we cap at 4 to avoid contention.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_process_one, art): art for art in to_process}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                processed.append(result)

    return processed


# ── Main ingestion API ────────────────────────────────────────────────────────

def ingest_articles(
    use_mock: bool = False,
    verticals: Optional[list[str]] = None,
    force_refresh: bool = False,
    today_only: bool = True,
    progress_callback=None,
) -> dict:
    """
    Full ingestion cycle with smart date filtering.

    Args:
        use_mock:          Use 20 mock articles (offline/demo mode).
        verticals:         Limit to specific ET verticals (None = all).
        force_refresh:     Re-process even if article already in DB.
        today_only:        If True (default), only fetch articles published TODAY.
                           Older articles already in LanceDB are retained unchanged.
        progress_callback: Optional callable(step: str, pct: float) for UI updates.
    """
    def _cb(step, pct):
        if progress_callback:
            progress_callback(step, pct)

    _cb("Initialising ingestion…", 0.02)

    # ── Source selection ──────────────────────────────────────────────────────
    if use_mock:
        _cb("Loading 20 demo articles…", 0.05)
        raw_articles = get_mock_articles()
        logger.info("Using %d mock articles.", len(raw_articles))

    elif today_only and INGEST_TODAY_ONLY:
        _cb("Fetching today's ET RSS articles…", 0.05)
        raw_articles = fetch_todays_articles(verticals=verticals)
        logger.info("Today-only fetch: %d new articles.", len(raw_articles))
        if raw_articles:
            mark_fetched_today()

    else:
        _cb("Fetching all available ET RSS articles…", 0.05)
        raw_articles = fetch_new_articles(verticals=verticals, force_refresh=force_refresh)
        logger.info("Full fetch: %d articles.", len(raw_articles))

    if not raw_articles:
        logger.info("No new articles to ingest.")
        return {"fetched": 0, "processed": 0, "upserted": 0, "errors": 0,
                "total_in_db": article_count()}

    _cb(f"Processing {len(raw_articles)} articles (embed + NER + sentiment)…", 0.20)

    processed = _process_articles_parallel(raw_articles, force_refresh=force_refresh)
    errors    = len(raw_articles) - len(processed)

    _cb("Upserting vectors into LanceDB…", 0.80)

    upserted = 0
    if processed:
        try:
            upserted = upsert_articles(processed)
        except Exception as e:
            logger.error("Upsert failed: %s", e)
            errors += len(processed)

    _cb("Ingestion complete.", 1.0)

    result = {
        "fetched":     len(raw_articles),
        "processed":   len(processed),
        "upserted":    upserted,
        "errors":      errors,
        "total_in_db": article_count(),
    }
    logger.info("Ingestion result: %s", result)
    return result


def start_background_scheduler():
    """
    Starts a background APScheduler job to poll ET RSS every N minutes.
    Only fetches today's articles on each run.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            func=lambda: ingest_articles(use_mock=False, today_only=True),
            trigger="interval",
            minutes=RSS_POLL_INTERVAL_MINUTES,
            id="et_rss_poller",
            replace_existing=True,
        )
        scheduler.start()
        logger.info(
            "Background RSS poller started (every %d min, today-only mode).",
            RSS_POLL_INTERVAL_MINUTES,
        )
        return scheduler
    except Exception as e:
        logger.warning("Background scheduler failed to start: %s", e)
        return None
