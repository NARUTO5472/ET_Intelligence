"""utils/helpers.py — Shared formatting and display utilities."""

from datetime import datetime, timezone


def format_relative_time(iso_str: str) -> str:
    """Returns human-readable relative time: '2 hours ago', 'Yesterday', etc."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff_secs = (now - dt).total_seconds()
        if diff_secs < 60:
            return "Just now"
        elif diff_secs < 3600:
            return f"{int(diff_secs // 60)}m ago"
        elif diff_secs < 86400:
            return f"{int(diff_secs // 3600)}h ago"
        elif diff_secs < 172800:
            return "Yesterday"
        else:
            return dt.strftime("%d %b %Y")
    except Exception:
        return "Recently"


def vertical_color(vertical: str) -> str:
    """Returns a hex colour for a given ET vertical."""
    colors = {
        "Markets":      "#F59E0B",
        "Tech":         "#3B82F6",
        "Startups":     "#8B5CF6",
        "Economy":      "#10B981",
        "Finance":      "#F97316",
        "Mutual Funds": "#EC4899",
        "Politics":     "#EF4444",
        "Healthcare":   "#06B6D4",
        "Auto":         "#84CC16",
        "Energy":       "#F59E0B",
    }
    return colors.get(vertical, "#6B7280")


def sentiment_to_emoji(compound: float) -> str:
    if compound >= 0.05:
        return "📈"
    elif compound <= -0.05:
        return "📉"
    return "➡️"
