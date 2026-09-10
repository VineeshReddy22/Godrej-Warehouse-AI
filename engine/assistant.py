from __future__ import annotations
from collections import Counter
from typing import List, Dict, Any, Optional
from .config import SCENARIOS

class SupervisorAssistant:
    def answer(self, question: str, incidents: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        q = question.lower().strip()
        n = len(incidents)
        meta = meta or {}
        is_demo = meta.get("is_demo", False)
        prefix = "[DEMO MODE — SYNTHETIC INCIDENTS] " if is_demo else ""

        if not incidents:
            return {
                'title': f'{prefix}No observed risk events',
                'summary': 'The analysis produced no validated potential-risk events from the perception and temporal reasoning pipeline.',
                'recommendations': [],
                'evidence_count': 0,
                'is_demo': is_demo
            }

        # Handle specific behavior query lookup
        known_behaviours = list(SCENARIOS.keys())
        queried_beh = None
        for b in known_behaviours:
            if b.lower() in q or any(w in q for w in b.lower().split()):
                # check if there's a strong match
                if b.lower() in q:
                    queried_beh = b
                    break

        if queried_beh and not any(e.get('behaviour') == queried_beh for e in incidents):
            return {
                'title': f'{prefix}{queried_beh} — Not Detected',
                'summary': f'The behaviour "{queried_beh}" was not detected in the analyzed footage.',
                'recommendations': ['Continue standard monitoring according to warehouse safety guidelines.'],
                'evidence_count': 0,
                'is_demo': is_demo
            }

        if 'high' in q or 'critical' in q:
            chosen = [e for e in incidents if e.get('risk') in {'HIGH', 'CRITICAL'}]
            return self._summarize(chosen, f'{prefix}High & Critical Risk Events', meta)

        if 'common' in q or 'frequent' in q or 'top' in q:
            c = Counter(e.get('behaviour') for e in incidents if e.get('behaviour'))
            top = c.most_common(3)
            return {
                'title': f'{prefix}Most Frequent Observed Behaviours',
                'summary': ' | '.join(f'{k}: {v}' for k, v in top),
                'recommendations': [SCENARIOS[k][2] for k, _ in top if k in SCENARIOS],
                'evidence_count': len(incidents),
                'is_demo': is_demo
            }

        if 'dedup' in q or 'merge' in q or 'duplicate' in q or 'count' in q:
            raw_c = meta.get('raw_incident_count', len(incidents))
            uniq_c = meta.get('unique_incident_count', len(incidents))
            m_c = meta.get('merged_incident_count', 0)
            return {
                'title': f'{prefix}Incident Deduplication Summary',
                'summary': f'Analysis recorded {raw_c} raw observations which were deduplicated into {uniq_c} unique physical incident(s) ({m_c} duplicate observation(s) merged).',
                'recommendations': ['Review evidence frames for merged observations in the Audit tab.'],
                'evidence_count': len(incidents),
                'is_demo': is_demo
            }

        if 'why' in q or 'classified' in q or 'first' in q:
            e = incidents[0]
            avg_conf = f"{e.get('confidence', 0.0):.0%}"
            return {
                'title': f"{prefix}Why {e.get('behaviour')} Was Flagged",
                'summary': f"Event ID #{e.get('id', 1)} ({e.get('behaviour')}, Risk: {e.get('risk')} {e.get('risk_score')}/100, Confidence: {avg_conf}) at {e.get('timestamp', 0):.2f}s: {e.get('explanation')}",
                'recommendations': [e.get('recommended_action', 'Inspect evidence.')],
                'evidence_count': len(e.get('evidence_frames', [e.get('evidence_frame')])),
                'timestamp': e.get('timestamp'),
                'is_demo': is_demo
            }

        return self._summarize(incidents, f'{prefix}Supervisor Shift Brief', meta)

    def _summarize(self, events: List[Dict[str, Any]], title: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        is_demo = meta.get("is_demo", False)
        if not events:
            return {
                'title': title,
                'summary': 'No events matched the requested query filter.',
                'recommendations': [],
                'evidence_count': 0,
                'is_demo': is_demo
            }

        c = Counter(e.get('behaviour') for e in events if e.get('behaviour'))
        top = c.most_common(3)
        rec = []
        for k, _ in top:
            if k in SCENARIOS:
                rec.append(SCENARIOS[k][2])

        raw_c = meta.get('raw_incident_count', len(events))
        uniq_c = meta.get('unique_incident_count', len(events))
        duration = meta.get('duration_s', 0.0)

        summary_text = (
            f"{len(events)} evidence-backed unique risk event(s) (from {raw_c} raw observations across {duration}s footage). "
            f"Top patterns: " + ', '.join(f'{k} ({v})' for k, v in top)
        )

        return {
            'title': title,
            'summary': summary_text,
            'recommendations': rec,
            'evidence_count': sum(len(e.get('evidence_frames', [e.get('evidence_frame')])) for e in events),
            'is_demo': is_demo
        }
