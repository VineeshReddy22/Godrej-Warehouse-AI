from __future__ import annotations
from typing import List, Dict, Any, Tuple

def _tracks_are_related(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    tid_a = a.get("track_id")
    tid_b = b.get("track_id")
    rtid_a = a.get("related_track_id")
    rtid_b = b.get("related_track_id")

    # Direct track match
    if tid_a is not None and tid_a == tid_b:
        return True
    
    # Cross match between primary track and related track
    if tid_a is not None and rtid_b is not None and tid_a == rtid_b:
        return True
    if rtid_a is not None and tid_b is not None and rtid_a == tid_b:
        return True

    # Shared related track
    if rtid_a is not None and rtid_b is not None and rtid_a == rtid_b:
        return True

    return False


def merge_nearby_incidents(incidents: List[Dict[str, Any]], window_s: float = 1.5) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Merges nearby incident observations of the same behavior with related tracks
    occurring within `window_s` seconds of each other.
    
    Preserves backward compatibility:
    - `evidence_frame`: Primary frame path from strongest observation
    - `evidence_frames`: List of all evidence frame paths for the merged group
    - `merged_incidents`: List of original raw incident observations merged into this event
    - Reassigns clean, unique sequential `id`s (1, 2, 3...)
    """
    if not incidents:
        return [], {"raw_incident_count": 0, "unique_incident_count": 0, "merged_incident_count": 0}

    # Sort incidents chronologically
    sorted_incidents = sorted(incidents, key=lambda x: x.get("timestamp", 0.0))
    raw_count = len(sorted_incidents)

    clusters: List[List[Dict[str, Any]]] = []

    for inc in sorted_incidents:
        merged_into_cluster = False
        beh = inc.get("behaviour")
        ts = inc.get("timestamp", 0.0)

        for cluster in clusters:
            # Check if inc matches any item in the cluster
            for member in cluster:
                if member.get("behaviour") == beh and abs(ts - member.get("timestamp", 0.0)) <= window_s:
                    if _tracks_are_related(member, inc):
                        cluster.append(inc)
                        merged_into_cluster = True
                        break
            if merged_into_cluster:
                break

        if not merged_into_cluster:
            clusters.append([inc])

    merged_output: List[Dict[str, Any]] = []

    for idx, cluster in enumerate(clusters, start=1):
        # Pick primary observation: highest risk_score, then highest confidence
        cluster.sort(key=lambda x: (x.get("risk_score", 0), x.get("confidence", 0.0)), reverse=True)
        primary = dict(cluster[0])

        # Gather all evidence frames
        evidence_frames: List[str] = []
        for member in cluster:
            ef = member.get("evidence_frame")
            if ef and ef not in evidence_frames:
                evidence_frames.append(ef)
            # Also include any sub-frames if member was already processed
            for sub_ef in member.get("evidence_frames", []):
                if sub_ef and sub_ef not in evidence_frames:
                    evidence_frames.append(sub_ef)

        # Primary evidence frame fallback
        primary_ef = primary.get("evidence_frame") or (evidence_frames[0] if evidence_frames else "")

        # Construct merged incident object
        primary["id"] = idx
        primary["evidence_frame"] = primary_ef
        primary["evidence_frames"] = evidence_frames
        primary["merged_incidents"] = [dict(m) for m in cluster]
        primary["observation_count"] = len(cluster)

        merged_output.append(primary)

    # Sort output by timestamp
    merged_output.sort(key=lambda x: x.get("timestamp", 0.0))
    # Re-assign sequential IDs after final timestamp sort
    for new_id, item in enumerate(merged_output, start=1):
        item["id"] = new_id

    unique_count = len(merged_output)
    merged_count = raw_count - unique_count

    stats = {
        "raw_incident_count": raw_count,
        "unique_incident_count": unique_count,
        "merged_incident_count": merged_count,
    }

    return merged_output, stats
