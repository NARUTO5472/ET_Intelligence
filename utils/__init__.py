from .helpers import format_relative_time, vertical_color, sentiment_to_emoji

def ingest_articles(*args, **kwargs):
    from .ingestion_orchestrator import ingest_articles as _f
    return _f(*args, **kwargs)

def start_background_scheduler():
    from .ingestion_orchestrator import start_background_scheduler as _f
    return _f()

