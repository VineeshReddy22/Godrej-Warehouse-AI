from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence


_EPSILON_MINUTES = 1e-6


def incident_rate(unique_incidents: int, duration_s: Any) -> float:
    """Return unique incidents per analyzed video minute, or zero for no duration/events."""
    try:
        incident_count = max(0, int(unique_incidents))
        duration_minutes = max(0.0, float(duration_s)) / 60.0
    except (TypeError, ValueError):
        return 0.0
    if incident_count == 0 or duration_minutes <= 0.0:
        return 0.0
    return incident_count / max(duration_minutes, _EPSILON_MINUTES)


def high_critical_share(incidents: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Return the HIGH/CRITICAL share as a percentage, or None when there are no incidents."""
    if not incidents:
        return None
    severe_count = sum(event.get("risk") in {"HIGH", "CRITICAL"} for event in incidents)
    return 100.0 * severe_count / len(incidents)


def evidence_coverage(
    incidents: Sequence[Dict[str, Any]],
    evidence_getter: Callable[[Dict[str, Any]], Iterable[Any]],
) -> Optional[float]:
    """Return the percentage of incidents with at least one usable evidence frame."""
    if not incidents:
        return None
    covered = sum(any(frame for frame in evidence_getter(event)) for event in incidents)
    return 100.0 * covered / len(incidents)


def scenario_coverage(
    incidents: Sequence[Dict[str, Any]],
    implemented_scenarios: Iterable[str],
) -> Dict[str, int]:
    """Return detected distinct scenarios and the configured implementation count."""
    implemented = set(implemented_scenarios)
    detected = {event.get("behaviour") for event in incidents if event.get("behaviour") in implemented}
    return {"detected": len(detected), "implemented": len(implemented)}


def merged_observation_rate(
    raw_count: Any,
    unique_count: Any,
    incidents: Sequence[Dict[str, Any]] = (),
) -> Optional[float]:
    """Return duplicate raw observations removed by deduplication as a percentage.

    When merged groups are present, each group's observations beyond its primary
    count as duplicates. Otherwise the existing raw-minus-unique metadata is used.
    """
    try:
        raw = max(0, int(raw_count))
        unique = max(0, int(unique_count))
    except (TypeError, ValueError):
        return None
    if raw == 0:
        return None

    grouped_duplicates = sum(
        max(0, len(event.get("merged_incidents") or []) - 1)
        for event in incidents
        if len(event.get("merged_incidents") or []) > 1
    )
    duplicate_count = grouped_duplicates if grouped_duplicates else max(0, raw - unique)
    duplicate_count = min(raw, duplicate_count)
    return 100.0 * duplicate_count / raw
