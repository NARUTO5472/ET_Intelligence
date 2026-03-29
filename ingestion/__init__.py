# Lazy imports — individual modules are imported directly to avoid
# pulling in feedparser/sentence_transformers at package import time.
def get_mock_articles():
    from .rss_fetcher import get_mock_articles as _f
    return _f()

def fetch_new_articles(*args, **kwargs):
    from .rss_fetcher import fetch_new_articles as _f
    return _f(*args, **kwargs)

def process_article(*args, **kwargs):
    from .article_processor import process_article as _f
    return _f(*args, **kwargs)

def embed_single(*args, **kwargs):
    from .article_processor import embed_single as _f
    return _f(*args, **kwargs)

def extract_entities(*args, **kwargs):
    from .article_processor import extract_entities as _f
    return _f(*args, **kwargs)

