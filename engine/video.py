from __future__ import annotations
import copy
import json
import os
import time
from pathlib import Path
import cv2
from ultralytics import YOLOWorld
from .config import PRODUCT_PROMPTS, PERSON_NAMES, EQUIPMENT_NAMES
from .tracker import GreedyTracker
from .detectors import BehaviourReasoner
from .dedup import merge_nearby_incidents


def _portable_path(value: str, output_dir: Path) -> str:
    try:
        relative = os.path.relpath(Path(value).resolve(), output_dir.resolve())
    except (OSError, ValueError):
        return value
    return Path(relative).as_posix()


def _portable_artifact(value, output_dir: Path):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {'video', 'evidence_frame'} and isinstance(item, str) and item:
                result[key] = _portable_path(item, output_dir)
            elif key == 'evidence_frames' and isinstance(item, list):
                result[key] = [_portable_path(path, output_dir) for path in item]
            else:
                result[key] = _portable_artifact(item, output_dir)
        return result
    if isinstance(value, list):
        return [_portable_artifact(item, output_dir) for item in value]
    return value

class VideoAnalyzer:
    def __init__(self, model_path='yolov8s-worldv2.pt', detection_stride=3, imgsz=512, confidence=.20, model=None):
        self.model_path = model_path
        self.detection_stride = max(1, int(detection_stride))
        self.imgsz = imgsz
        self.confidence = confidence
        self.model = model

    def _get_model(self):
        if self.model is not None:
            return self.model
        try:
            m = YOLOWorld(self.model_path)
            m.set_classes(PRODUCT_PROMPTS)
            return m
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOWorld model from '{self.model_path}': {e}")

    def analyze(self, video_path, output_dir='outputs', progress_cb=None, save_evidence=True):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        evidence = out / 'evidence'
        evidence.mkdir(exist_ok=True)

        model = self._get_model()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f'Unable to open video: {video_path}')

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        tracker = GreedyTracker(
            max_distance=max(width, height) * .12 if width else 150,
            max_missed=max(5, self.detection_stride * 3)
        )
        reasoner = BehaviourReasoner()
        raw_incidents = []
        frame_idx = 0
        detections_run = 0
        start = time.time()
        last_tracks = []
        last_event_at = {}

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t = (frame_idx - 1) / fps

            if frame_idx == 1 or (frame_idx - 1) % self.detection_stride == 0:
                results = model.predict(frame, imgsz=self.imgsz, conf=self.confidence, verbose=False)
                dets = []
                for r in results:
                    if r.boxes is None:
                        continue
                    boxes = r.boxes.xyxy.cpu().tolist()
                    cls = r.boxes.cls.int().cpu().tolist()
                    conf = r.boxes.conf.cpu().tolist()
                    names = r.names
                    for box, cid, score in zip(boxes, cls, conf):
                        label = str(names[int(cid)]).lower()
                        if label in PERSON_NAMES or label in EQUIPMENT_NAMES or label in PRODUCT_PROMPTS:
                            dets.append({'box': box, 'label': label, 'confidence': float(score)})

                last_tracks = tracker.update(dets, t)
                detections_run += 1
                events = reasoner.process(last_tracks, width, height, t)

                for e in events:
                    signature = (e.get('behaviour'), e.get('track_id'), e.get('related_track_id'))
                    cooldown = 10.0 if e.get('behaviour') == 'Unattended product' else 3.0
                    previous = last_event_at.get(signature)
                    if previous is not None and (t - previous) < cooldown:
                        continue
                    last_event_at[signature] = t

                    e['id'] = len(raw_incidents) + 1
                    e['analysis_latency_s'] = round(time.time() - start, 3)
                    e['source_frame'] = frame_idx

                    if save_evidence:
                        name = f"event_{e['id']:04d}_{e['behaviour'].replace('/', '-').replace(' ', '_')}_{t:07.2f}s.jpg"
                        p = evidence / name
                        annotated = frame.copy()
                        tid = e.get('track_id')
                        rt = e.get('related_track_id')
                        for tr in last_tracks:
                            if tr.track_id in {tid, rt}:
                                x1, y1, x2, y2 = map(int, tr.box)
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 170), 2)
                                cv2.putText(annotated, f"ID {tr.track_id} {tr.label}", (x1, max(18, y1 - 8)),
                                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 170), 2)
                        cv2.putText(annotated, f"{e['behaviour']} | {e['risk']} | {e['timestamp']:.2f}s", (20, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 240, 210), 2)
                        cv2.imwrite(str(p), annotated)
                        e['evidence_frame'] = str(p)

                    raw_incidents.append(e)

            if progress_cb and total:
                progress_cb(min(1.0, frame_idx / total))

        cap.release()
        duration = total / fps if total else frame_idx / fps

        # Perform lightweight post-processing deduplication
        unique_incidents, stats = merge_nearby_incidents(raw_incidents, window_s=1.5)

        meta = {
            'fps': fps,
            'frames': total or frame_idx,
            'width': width,
            'height': height,
            'duration_s': round(duration, 2),
            'detections_run': detections_run,
            'detection_stride': self.detection_stride,
            'processing_s': round(time.time() - start, 2),
            'realtime_factor': round(duration / max(0.001, time.time() - start), 2),
            'engine': 'YOLOWorld open-vocabulary perception + persistent greedy tracking + temporal geometry reasoning',
            'model': self.model_path,
            'is_demo': False,
            'mode': 'REAL AI ANALYSIS',
            'raw_incident_count': stats['raw_incident_count'],
            'unique_incident_count': stats['unique_incident_count'],
            'merged_incident_count': stats['merged_incident_count'],
        }

        result = {
            'video': str(video_path),
            'meta': meta,
            'incidents': unique_incidents,
            'raw_incidents': raw_incidents,
            'scenario_coverage': sorted(set(x['behaviour'] for x in unique_incidents))
        }
        artifact = _portable_artifact(copy.deepcopy(result), out)
        (out / 'analysis.json').write_text(json.dumps(artifact, indent=2), encoding='utf-8')
        return unique_incidents, meta
