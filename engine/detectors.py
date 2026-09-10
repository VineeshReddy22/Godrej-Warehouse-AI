from __future__ import annotations
from collections import defaultdict
import math
from .config import PERSON_NAMES, EQUIPMENT_NAMES, DEFAULT_ZONES, SCENARIOS

class BehaviourReasoner:
    def __init__(self):
        self.last_emit=defaultdict(float)
        self.sequence_active={}
        self.stationary_since={}
        self.orientation_baseline={}
        self.recurrence=defaultdict(int)

    def _emit(self, events, name, t, track_id, conf, impact, evidence, note=None):
        key=(track_id,name)
        if t-self.last_emit[key] < 2.5: return
        self.last_emit[key]=t; self.recurrence[key]+=1
        desc,base,action=SCENARIOS[name]
        from .risk import classify
        rr=classify(base,conf,persistence_s=float(evidence.get('persistence_s',0)),impact=impact,recurrence=self.recurrence[key])
        events.append({'id':None,'behaviour':name,'risk':rr.risk,'risk_score':rr.score,'confidence':round(conf,3),'timestamp':round(t,2),'track_id':track_id,'related_track_id':evidence.get('related_track_id'),'subject':evidence.get('subject','product'),'evidence':evidence,'explanation':note or desc,'why_risky':f"Observed evidence is consistent with {name.lower()}. The event is treated as a potential risk, not proof of damage.",'recommended_action':action,'status':'POTENTIAL_RISK','damage_confirmed':False,'detection_source':'AI_BEHAVIOUR_ENGINE'})

    @staticmethod
    def _is_person(t): return t.label.lower() in PERSON_NAMES
    @staticmethod
    def _is_equipment(t): return t.label.lower() in EQUIPMENT_NAMES
    @staticmethod
    def _is_product(t): return (not BehaviourReasoner._is_person(t)) and (not BehaviourReasoner._is_equipment(t))
    @staticmethod
    def _near(a,b,scale=0.35):
        ax,ay=a.center; bx,by=b.center
        return math.hypot(ax-bx,ay-by) <= scale*max(a.width,a.height,b.width,b.height)
    @staticmethod
    def _overlap(a,b):
        x=max(0,min(a.box[2],b.box[2])-max(a.box[0],b.box[0])); return x/max(1,min(a.width,b.width))

    def process(self, tracks, width, height, t):
        products=[x for x in tracks if self._is_product(x)]
        people=[x for x in tracks if self._is_person(x)]
        events=[]
        # 1) motion-based events
        for p in products:
            if len(p.history)<4: continue
            h=p.history
            cur=h[-1]; old=h[-4]; dt=max(.05,cur['t']-old['t']); vx=(cur['cx']-old['cx'])/dt; vy=(cur['cy']-old['cy'])/dt; speed=math.hypot(vx,vy)
            near_person=any(self._near(p,w,0.45) for w in people)
            floor=p.bottom/height
            if near_person and floor>0.58 and abs(vx)>35 and abs(vy)<abs(vx)*0.65 and len(h)>=5:
                self._emit(events,'Dragging',t,p.track_id,min(.96,.62+p.confidence*.25),min(1,speed/180),{'speed_px_s':round(speed,1),'horizontal_velocity_px_s':round(vx,1),'vertical_velocity_px_s':round(vy,1),'person_nearby':True,'persistence_s':round(max(0,t-old['t']),2),'subject':p.label})
            if len(h)>=5 and abs(vx)>40 and abs(vy)<abs(vx)*.55 and abs(p.width-p.history[-3]['w'])/max(1,p.history[-3]['w'])>.18:
                self._emit(events,'Rolling',t,p.track_id,min(.9,.55+p.confidence*.25),min(1,abs(vx)/220),{'horizontal_velocity_px_s':round(vx,1),'bbox_width_change':round(abs(p.width-p.history[-3]['w'])/max(1,p.history[-3]['w']),3),'persistence_s':round(max(0,t-old['t']),2),'subject':p.label})
            if vy>120 and cur['speed']>140 and old['cy'] < cur['cy']-20 and vy > abs(vx):
                self._emit(events,'Throwing/Dropping',t,p.track_id,min(.98,.68+p.confidence*.25),min(1,vy/350),{'vertical_velocity_px_s':round(vy,1),'speed_px_s':round(cur['speed'],1),'person_nearby':near_person,'subject':p.label})
            if len(h)>=4 and cur['speed']>180 and abs(cur['speed']-old.get('speed',0))>90:
                self._emit(events,'Rough handling / excessive force',t,p.track_id,min(.92,.58+p.confidence*.28),min(1,cur['speed']/300),{'speed_px_s':round(cur['speed'],1),'speed_change_px_s':round(abs(cur['speed']-old.get('speed',0)),1),'subject':p.label})
            # orientation proxy from temporal aspect ratio change; only report when a meaningful baseline exists
            ar=p.width/max(1,p.height)
            b=self.orientation_baseline.get(p.track_id)
            if b is None and len(h)>=10:
                vals=[x['w']/max(1,x['h']) for x in h[-10:]]; self.orientation_baseline[p.track_id]=sum(vals)/len(vals); b=self.orientation_baseline[p.track_id]
            if b and b<0.9 and ar>1.35 and len(h)>=10:
                self._emit(events,'Vertical product kept horizontally',t,p.track_id,.72,min(1,ar/2.2),{'baseline_aspect_ratio':round(b,3),'current_aspect_ratio':round(ar,3),'subject':p.label,'persistence_s':round(t-h[-10]['t'],2)},'The tracked product became substantially flatter than its earlier observed orientation and remained in that geometry.')
            # unattended (require track presence >= 5 frames before registering stationary duration)
            if len(h)>=5 and cur['speed']<8 and not any(self._near(p,w,0.55) for w in people):
                since=self.stationary_since.setdefault(p.track_id,t)
                if t-since>=5.0:
                    self._emit(events,'Unattended product',t,p.track_id,.82,.35,{'stationary_duration_s':round(t-since,1),'nearest_person':False,'subject':p.label,'persistence_s':round(t-since,1)})
            else:
                self.stationary_since.pop(p.track_id,None)
            # zone violation, based on configurable normalized zone
            z=DEFAULT_ZONES['designated_zone']; cx,cy=p.center
            inside=z.x1*width<=cx<=z.x2*width and z.y1*height<=cy<=z.y2*height
            if len(h)>=3 and not inside and floor>0.35:
                self._emit(events,'Outside designated area',t,p.track_id,.76,.25,{'center_x':round(cx,1),'center_y':round(cy,1),'configured_zone':[z.x1,z.y1,z.x2,z.y2],'subject':p.label})
        # 2) spatial stacking / heavy-on-light
        for a in products:
            for b in products:
                if a.track_id==b.track_id: continue
                ax1,ay1,ax2,ay2=a.box; bx1,by1,bx2,by2=b.box
                if ay2 <= by1 + .12*max(a.height,b.height):
                    area_ratio=a.area/max(1,b.area); ov=self._overlap(a,b); gap=max(0,by1-ay2)/max(1,b.height)
                    if area_ratio>=1.35 and ov>=.50 and gap<=.20 and len(a.history)>=4 and len(b.history)>=4:
                        self._emit(events,'Heavy item on light/fragile item',t,a.track_id,min(.95,.60+min(area_ratio/3,1)*.25),min(1,area_ratio/3),{'related_track_id':b.track_id,'upper_area':round(a.area,1),'lower_area':round(b.area,1),'visual_size_ratio':round(area_ratio,2),'horizontal_overlap':round(ov,2),'vertical_gap_ratio':round(gap,3),'subject':a.label,'persistence_s':round(t-min(a.history[-4]['t'],b.history[-4]['t']),2)})
                # unstable stacking proxy: high upper offset + weak overlap + close vertical relationship
                if abs(a.center[0]-b.center[0]) < max(a.width,b.width)*.9 and ay2<=by1+max(a.height,b.height)*.25:
                    if len(a.history)>=4 and len(b.history)>=4 and self._overlap(a,b)<.25 and a.bottom/height>.45:
                        self._emit(events,'Improper stacking',t,a.track_id,.68,.45,{'related_track_id':b.track_id,'horizontal_overlap':round(self._overlap(a,b),2),'subject':a.label})
        # 3) stepping: person's foot point inside product top/bounds
        for w in people:
            if len(w.history)<3: continue
            foot=(w.center[0],w.box[3])
            for p in products:
                if len(p.history)<3: continue
                x1,y1,x2,y2=p.box
                if x1-0.05*p.width<=foot[0]<=x2+0.05*p.width and y1-0.18*p.height<=foot[1]<=y2+0.12*p.height:
                    self._emit(events,'Stepping on cartons',t,w.track_id,.88,.9,{'related_track_id':p.track_id,'foot_point':[round(foot[0],1),round(foot[1],1)],'product_box':[round(v,1) for v in p.box],'subject':p.label})
        # 4) sequence: multiple products active in loading zone, newest arrives before previous settles
        z=DEFAULT_ZONES['loading_zone']; active=[]
        for p in products:
            cx,cy=p.center; inside=z.x1*width<=cx<=z.x2*width and z.y1*height<=cy<=z.y2*height
            if inside: active.append(p)
        if len(active)>=2:
            active.sort(key=lambda x:x.last_time)
            newest=active[-1]
            older=[x for x in active[:-1] if x.history and x.history[-1]['speed']>12]
            if older:
                self._emit(events,'Unsafe loading/unloading sequence',t,newest.track_id,.79,.55,{'related_track_id':older[-1].track_id,'active_products':len(active),'subject':newest.label})
        return events
