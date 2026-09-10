from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple

@dataclass
class Track:
    track_id: int
    label: str
    confidence: float
    box: Tuple[float, float, float, float]
    last_time: float
    missed: int = 0
    history: List[dict] = field(default_factory=list)

    @property
    def center(self):
        x1,y1,x2,y2 = self.box
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    @property
    def width(self): return max(1.0, self.box[2]-self.box[0])
    @property
    def height(self): return max(1.0, self.box[3]-self.box[1])
    @property
    def area(self): return self.width*self.height
    @property
    def bottom(self): return self.box[3]

class GreedyTracker:
    def __init__(self, max_distance=150.0, max_missed=8):
        self.max_distance=max_distance
        self.max_missed=max_missed
        self.next_id=1
        self.tracks: Dict[int,Track]={}

    @staticmethod
    def _iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        aa=max(1,(a[2]-a[0])*(a[3]-a[1])); bb=max(1,(b[2]-b[0])*(b[3]-b[1]))
        return inter/(aa+bb-inter)

    @staticmethod
    def _dist(a,b):
        ax=(a[0]+a[2])/2; ay=(a[1]+a[3])/2
        bx=(b[0]+b[2])/2; by=(b[1]+b[3])/2
        return math.hypot(ax-bx, ay-by)

    def update(self, detections: List[dict], timestamp: float):
        unmatched_tracks=set(self.tracks)
        unmatched_dets=set(range(len(detections)))
        pairs=[]
        for tid,t in self.tracks.items():
            for di,d in enumerate(detections):
                dist=self._dist(t.box,d['box'])
                iou=self._iou(t.box,d['box'])
                score=dist - 180*iou
                if dist <= self.max_distance or iou >= 0.05:
                    pairs.append((score,tid,di))
        pairs.sort()
        for _,tid,di in pairs:
            if tid not in unmatched_tracks or di not in unmatched_dets: continue
            d=detections[di]; t=self.tracks[tid]
            t.box=tuple(d['box']); t.label=d['label']; t.confidence=float(d['confidence']); t.missed=0; t.last_time=timestamp
            self._append(t,timestamp)
            unmatched_tracks.remove(tid); unmatched_dets.remove(di)
        for tid in list(unmatched_tracks):
            self.tracks[tid].missed += 1
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]
        for di in unmatched_dets:
            d=detections[di]
            t=Track(self.next_id,d['label'],float(d['confidence']),tuple(d['box']),timestamp)
            self.next_id += 1
            self._append(t,timestamp)
            self.tracks[t.track_id]=t
        return list(self.tracks.values())

    def _append(self,t:Track,timestamp):
        cx,cy=t.center
        rec={'t':timestamp,'cx':cx,'cy':cy,'w':t.width,'h':t.height,'area':t.area,'bottom':t.bottom,'conf':t.confidence}
        if t.history:
            p=t.history[-1]; dt=max(1e-3,timestamp-p['t']); rec['vx']=(cx-p['cx'])/dt; rec['vy']=(cy-p['cy'])/dt; rec['speed']=math.hypot(rec['vx'],rec['vy'])
        else:
            rec['vx']=rec['vy']=rec['speed']=0.0
        t.history.append(rec)
        if len(t.history)>40: del t.history[:-40]
