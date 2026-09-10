from __future__ import annotations
import json
import time
from pathlib import Path
import cv2
import numpy as np

from .config import SCENARIOS
from .risk import classify

DEMO_SCENARIO_TEMPLATES = [
    {
        "behaviour": "Stepping on cartons",
        "base_risk": "CRITICAL",
        "timestamp": 4.12,
        "track_id": 11,
        "related_track_id": 20,
        "subject": "worker",
        "confidence": 0.88,
        "persistence_s": 1.2,
        "impact": 0.90,
        "explanation": "Observed worker foot region overlapping the top surface of a tracked carton.",
        "why_risky": "Stepping on cartons can cause structural damage to inner goods and presents a workplace tripping/fall hazard.",
        "recommended_action": SCENARIOS["Stepping on cartons"][2],
        "evidence_details": {"foot_point": [320.0, 410.0], "product_box": [280.0, 390.0, 360.0, 460.0]}
    },
    {
        "behaviour": "Throwing/Dropping",
        "base_risk": "CRITICAL",
        "timestamp": 7.95,
        "track_id": 14,
        "related_track_id": None,
        "subject": "package",
        "confidence": 0.92,
        "persistence_s": 0.5,
        "impact": 0.85,
        "explanation": "Package exhibited rapid downward vertical velocity (vy=185 px/s) consistent with dropping.",
        "why_risky": "Abrupt impact forces during dropping frequently result in high-severity product and packaging damage.",
        "recommended_action": SCENARIOS["Throwing/Dropping"][2],
        "evidence_details": {"vertical_velocity_px_s": 185.0, "speed_px_s": 195.0, "person_nearby": True}
    },
    {
        "behaviour": "Heavy item on light/fragile item",
        "base_risk": "HIGH",
        "timestamp": 12.40,
        "track_id": 18,
        "related_track_id": 22,
        "subject": "large crate",
        "confidence": 0.85,
        "persistence_s": 3.4,
        "impact": 0.70,
        "explanation": "A visually larger product (visual size ratio 2.1x) was placed directly above a smaller parcel.",
        "why_risky": "Heavy items placed atop fragile/light items risk crushing and structural collapse.",
        "recommended_action": SCENARIOS["Heavy item on light/fragile item"][2],
        "evidence_details": {"visual_size_ratio": 2.1, "horizontal_overlap": 0.68, "upper_area": 12000.0, "lower_area": 5700.0}
    },
    {
        "behaviour": "Dragging",
        "base_risk": "HIGH",
        "timestamp": 16.80,
        "track_id": 5,
        "related_track_id": 2,
        "subject": "carton",
        "confidence": 0.81,
        "persistence_s": 2.8,
        "impact": 0.60,
        "explanation": "Carton pulled across floor surface near operator at sustained horizontal velocity (vx=42 px/s).",
        "why_risky": "Floor dragging causes base abrasion, tearing, and potential structural damage.",
        "recommended_action": SCENARIOS["Dragging"][2],
        "evidence_details": {"speed_px_s": 45.0, "horizontal_velocity_px_s": 42.0, "person_nearby": True}
    },
    {
        "behaviour": "Unattended product",
        "base_risk": "MEDIUM",
        "timestamp": 22.10,
        "track_id": 29,
        "related_track_id": None,
        "subject": "parcel",
        "confidence": 0.84,
        "persistence_s": 6.5,
        "impact": 0.35,
        "explanation": "Parcel remained stationary for 6.5 seconds with no operator or equipment nearby.",
        "why_risky": "Unattended items in active aisles cause obstruction and risk unrecorded damage.",
        "recommended_action": SCENARIOS["Unattended product"][2],
        "evidence_details": {"stationary_duration_s": 6.5, "nearest_person": False}
    },
    {
        "behaviour": "Outside designated area",
        "base_risk": "MEDIUM",
        "timestamp": 26.50,
        "track_id": 31,
        "related_track_id": None,
        "subject": "wooden pallet",
        "confidence": 0.78,
        "persistence_s": 2.0,
        "impact": 0.30,
        "explanation": "Tracked pallet crossed outside configured operating handling zone.",
        "why_risky": "Staging outside zones risks collision with moving machinery or unauthorized handling.",
        "recommended_action": SCENARIOS["Outside designated area"][2],
        "evidence_details": {"center_x": 48.0, "center_y": 310.0, "configured_zone": [0.05, 0.10, 0.95, 0.95]}
    }
]


def generate_demo_incidents(output_dir: str = "outputs", save_evidence: bool = True):
    """
    Generates deterministic, realistic demo incidents that match the exact real AI schema.
    Clearly flags output as DEMO MODE — SYNTHETIC INCIDENTS.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    incidents = []
    for idx, tmpl in enumerate(DEMO_SCENARIO_TEMPLATES, start=1):
        rr = classify(
            base_risk=tmpl["base_risk"],
            confidence=tmpl["confidence"],
            persistence_s=tmpl["persistence_s"],
            impact=tmpl["impact"],
            recurrence=1
        )

        inc_id = idx
        beh_clean = tmpl["behaviour"].replace("/", "-").replace(" ", "_")
        t = tmpl["timestamp"]
        img_name = f"demo_event_{inc_id:04d}_{beh_clean}_{t:07.2f}s.jpg"
        img_path = str(evidence_dir / img_name)

        if save_evidence:
            # Generate deterministic synthetic image with watermark
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            canvas[:] = (20, 26, 12)  # Dark teal backdrop matching Godrej theme

            # Grid lines
            for x in range(0, 640, 40):
                cv2.line(canvas, (x, 0), (x, 480), (35, 45, 25), 1)
            for y in range(0, 480, 40):
                cv2.line(canvas, (0, y), (640, y), (35, 45, 25), 1)

            # Simulated bounding boxes
            cv2.rectangle(canvas, (180, 160), (460, 380), (168, 214, 66), 2)
            cv2.putText(canvas, f"DEMO TRACK #{tmpl['track_id']} [{tmpl['subject']}]", (185, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (168, 214, 66), 2)

            if tmpl["related_track_id"] is not None:
                cv2.rectangle(canvas, (240, 280), (400, 420), (220, 170, 0), 2)
                cv2.putText(canvas, f"DEMO TRACK #{tmpl['related_track_id']}", (245, 275),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 170, 0), 1)

            # Watermark header
            cv2.rectangle(canvas, (0, 0), (640, 42), (10, 16, 6), -1)
            cv2.putText(canvas, "DEMO MODE - SYNTHETIC EVIDENCE", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

            # Event details footer
            cv2.putText(canvas, f"{tmpl['behaviour']} | {rr.risk} ({rr.score}/100) | {t:.2f}s",
                        (15, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 240, 235), 1)

            cv2.imwrite(img_path, canvas)

        inc_record = {
            "id": inc_id,
            "behaviour": tmpl["behaviour"],
            "risk": rr.risk,
            "risk_score": rr.score,
            "confidence": round(tmpl["confidence"], 3),
            "timestamp": round(t, 2),
            "track_id": tmpl["track_id"],
            "related_track_id": tmpl["related_track_id"],
            "subject": tmpl["subject"],
            "evidence": tmpl["evidence_details"],
            "explanation": tmpl["explanation"],
            "why_risky": tmpl["why_risky"],
            "recommended_action": tmpl["recommended_action"],
            "status": "POTENTIAL_RISK",
            "damage_confirmed": False,
            "detection_source": "DEMO_MODE_SYNTHETIC",
            "analysis_latency_s": 0.05,
            "source_frame": int(t * 25),
            "evidence_frame": img_path if save_evidence else "",
            "evidence_frames": [img_path] if save_evidence else [],
            "merged_incidents": [],
            "observation_count": 1,
            "is_demo": True,
            "mode": "DEMO"
        }
        incidents.append(inc_record)

    from .dedup import merge_nearby_incidents
    incidents, stats = merge_nearby_incidents(incidents, window_s=1.5)

    meta = {
        "fps": 25.0,
        "frames": 750,
        "width": 640,
        "height": 480,
        "duration_s": 30.0,
        "detections_run": 250,
        "detection_stride": 3,
        "processing_s": 0.12,
        "realtime_factor": 250.0,
        "engine": "DEMO / FALLBACK ENGINE — SYNTHETIC INCIDENTS",
        "model": "demo_synthetic_model",
        "is_demo": True,
        "mode": "DEMO MODE — SYNTHETIC INCIDENTS",
        "raw_incident_count": stats["raw_incident_count"],
        "unique_incident_count": stats["unique_incident_count"],
        "merged_incident_count": stats["merged_incident_count"]
    }

    result = {
        "video": "demo_synthetic_video.mp4",
        "meta": meta,
        "incidents": incidents,
        "scenario_coverage": sorted(set(x["behaviour"] for x in incidents))
    }
    (out / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return incidents, meta
