"""Utility functions for Medical Vision AI."""

import uuid
from datetime import datetime, timezone


def generate_analysis_id() -> str:
    """Generate a unique identifier for an analysis run."""
    return f"ana_{uuid.uuid4().hex[:12]}"


def get_utc_timestamp() -> str:
    """Return the current UTC timestamp formatted as ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def format_duration_ms(ms: float) -> str:
    """Format duration in milliseconds to human-readable string.

    Args:
        ms: Duration in milliseconds.

    Returns:
        Formatted string (e.g. '45.2 ms' or '1.24 s').
    """
    if ms < 1000.0:
        return f"{ms:.1f} ms"
    return f"{ms / 1000.0:.2f} s"


def format_score_percentage(score: float) -> str:
    """Format numerical score into percentage string.

    Args:
        score: Value between 0.0 and 1.0.

    Returns:
        Formatted string (e.g. '85.4%').
    """
    clamped = max(0.0, min(1.0, score))
    return f"{clamped * 100:.1f}%"
