from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Set

from engine.config import RISK_ORDER

EPISODE_MAX_GAP_S = 1.5


def _timestamp(incident: Dict[str, Any]) -> Optional[float]:
    value = incident.get("timestamp")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _observations(incident: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged = incident.get("merged_incidents")
    if isinstance(merged, list) and merged:
        observations = [incident]
        incident_id = incident.get("id")
        for item in merged:
            if not isinstance(item, dict):
                continue
            if item is incident or (incident_id is not None and item.get("id") == incident_id):
                continue
            observations.append(item)
        return observations
    return [incident]


def _tracks(incident: Dict[str, Any]) -> Set[str]:
    tracks: Set[str] = set()
    for observation in _observations(incident):
        for field in ("track_id", "related_track_id"):
            value = observation.get(field)
            if value is not None and value != "":
                tracks.add(str(value))
    return tracks


def _subjects(incident: Dict[str, Any]) -> Set[str]:
    subjects = set()
    for observation in _observations(incident):
        value = observation.get("subject")
        if value:
            subjects.add(str(value).strip().lower())
    return subjects


def _frames(incident: Dict[str, Any]) -> List[Any]:
    result: List[Any] = []
    for observation in _observations(incident):
        values = observation.get("evidence_frames")
        if not isinstance(values, list) or not values:
            values = [observation.get("evidence_frame")]
        for frame in values:
            if frame and frame not in result:
                result.append(frame)
    return result


def _related(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_tracks = _tracks(left)
    right_tracks = _tracks(right)
    if left_tracks and right_tracks:
        if left_tracks & right_tracks:
            return True
    left_subjects = _subjects(left)
    right_subjects = _subjects(right)
    return not left_subjects or not right_subjects or bool(left_subjects & right_subjects)


def _close(left: Dict[str, Any], right: Dict[str, Any], max_gap_s: float) -> bool:
    left_time = _timestamp(left)
    right_time = _timestamp(right)
    if left_time is None or right_time is None:
        return left_time is None and right_time is None
    return abs(right_time - left_time) <= max_gap_s


def _episode_observations(group: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for incident in group:
        observations.extend(_observations(incident))
    return observations


def _first_value(observations: List[Dict[str, Any]], field: str) -> Any:
    for observation in observations:
        value = observation.get(field)
        if value is not None and value != "":
            return value
    return None


def build_event_episodes(
    incidents: Iterable[Dict[str, Any]],
    max_gap_s: float = EPISODE_MAX_GAP_S,
) -> List[Dict[str, Any]]:
    """Build deterministic, UI-only event episodes from unique AI incidents."""
    ordered = sorted(
        (deepcopy(item) for item in incidents if isinstance(item, dict)),
        key=lambda item: (_timestamp(item) is None, _timestamp(item) or 0.0, str(item.get("id", ""))),
    )
    groups: List[List[Dict[str, Any]]] = []
    for incident in ordered:
        if groups:
            current = groups[-1]
            previous = current[-1]
            same_behaviour = incident.get("behaviour") == previous.get("behaviour")
            if same_behaviour and _close(previous, incident, max_gap_s) and _related(previous, incident):
                current.append(incident)
                continue
        groups.append([incident])

    episodes: List[Dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        observations = _episode_observations(group)
        timestamps = [value for value in (_timestamp(item) for item in observations) if value is not None]
        risks = sorted(group, key=lambda item: RISK_ORDER.get(item.get("risk"), 0), reverse=True)
        strongest = risks[0] if risks else group[0]
        track_ids: List[Any] = []
        evidence_frames: List[Any] = []
        original_ids: List[Any] = []
        for incident in group:
            original_ids.append(incident.get("id"))
            for track in sorted(_tracks(incident)):
                if track not in {str(value) for value in track_ids}:
                    track_ids.append(int(track) if track.isdigit() else track)
            for frame in _frames(incident):
                if frame not in evidence_frames:
                    evidence_frames.append(frame)

        episode = {
            "episode_id": f"EP-{index:03d}",
            "id": f"EP-{index:03d}",
            "behaviour": strongest.get("behaviour"),
            "risk": strongest.get("risk"),
            "risk_score": max((float(item.get("risk_score", 0) or 0) for item in group), default=0),
            "confidence": max((float(item.get("confidence", 0) or 0) for item in group), default=0),
            "start_timestamp": min(timestamps) if timestamps else None,
            "end_timestamp": max(timestamps) if timestamps else None,
            "duration": (max(timestamps) - min(timestamps)) if timestamps else None,
            "observation_count": len(observations),
            "unique_incident_count": len(group),
            "evidence_frames": evidence_frames,
            "evidence_count": len(evidence_frames),
            "track_ids": track_ids,
            "original_incident_ids": original_ids,
            "explanation": _first_value(observations, "explanation"),
            "why_risky": _first_value(observations, "why_risky"),
            "recommended_action": _first_value(observations, "recommended_action"),
            "status": _first_value(observations, "status"),
            "damage_confirmed": next((item.get("damage_confirmed") for item in observations if item.get("damage_confirmed") is not None), None),
            "incidents": group,
        }
        episode["timestamp"] = episode["start_timestamp"]
        episode["evidence_frame"] = evidence_frames[0] if evidence_frames else None
        episode["track_id"] = track_ids[0] if track_ids else None
        episodes.append(episode)
    return episodes
