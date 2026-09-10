from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def context_timestamps(timestamp: Any, duration: Any = None, offset_s: float = 1.0) -> Dict[str, Optional[float]]:
    """Return before/incident/after times without inventing an incident timestamp."""
    try:
        incident = float(timestamp)
    except (TypeError, ValueError):
        return {"before": None, "incident": None, "after": None}

    offset = max(0.0, float(offset_s))
    before = max(0.0, incident - offset)
    after = incident + offset
    try:
        if duration is not None:
            after = min(after, max(0.0, float(duration)))
    except (TypeError, ValueError):
        pass
    return {"before": before, "incident": incident, "after": after}


def nearest_evidence(records: Iterable[Dict[str, Any]], target: Optional[float]) -> Optional[Dict[str, Any]]:
    """Select the closest timestamped evidence record, with stable first-item fallback."""
    records = list(records)
    if not records:
        return None
    timestamped = [record for record in records if record.get("timestamp") is not None]
    if target is None or not timestamped:
        return records[0]
    return min(timestamped, key=lambda record: abs(float(record["timestamp"]) - target))
