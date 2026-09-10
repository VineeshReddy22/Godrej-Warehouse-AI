from pathlib import Path
import json
import html
import re
import streamlit as st

from engine.video import VideoAnalyzer, _portable_artifact
from engine.assistant import SupervisorAssistant
from engine.config import SCENARIOS, risk_color
from engine.demo import generate_demo_incidents
from engine.kpi import evidence_coverage, high_critical_share, incident_rate, merged_observation_rate, scenario_coverage
from engine.replay import context_timestamps, nearest_evidence
from engine.episodes import build_event_episodes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
EVID = OUT / "evidence"
EVID.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Godrej Warehouse AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def get_cached_model(model_path="yolov8s-worldv2.pt"):
    """
    Loads YOLOWorld model once and caches it across Streamlit reruns.
    """
    try:
        from ultralytics import YOLOWorld
        from engine.config import PRODUCT_PROMPTS
        m = YOLOWorld(model_path)
        m.set_classes(PRODUCT_PROMPTS)
        return m
    except Exception as e:
        st.warning(f"Could not initialize cached YOLOWorld model ('{model_path}'): {e}")
        return None

# ---------------------------------------------------------------------
# VISUAL SYSTEM
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --bg:#061014; --panel:#0b171c; --panel2:#0c1a20;
  --line:#20353d; --text:#edf7f5; --muted:#8ea6ad;
  --accent:#42d6a8; --accent2:#75e4c3;
}
.stApp{
  background:
    radial-gradient(circle at 82% 8%, rgba(66,214,168,.08), transparent 25%),
    radial-gradient(circle at 8% 30%, rgba(42,113,105,.07), transparent 24%),
    var(--bg);
  color:var(--text);
}
.block-container{max-width:1540px;padding:1.1rem 2.2rem 2.5rem}
header[data-testid="stHeader"]{background:transparent}
section[data-testid="stSidebar"]{background:#071216;border-right:1px solid var(--line)}
h1,h2,h3,h4{font-family:"Space Grotesk",sans-serif!important;color:var(--text)!important}
p,div,span,button,input{font-family:"Inter",sans-serif}
.mono{font-family:"IBM Plex Mono",monospace}
.topbar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:4px 0 18px;margin-bottom:24px}
.brand{display:flex;align-items:center;gap:12px}
.brand-mark{width:38px;height:38px;border:1px solid var(--accent);display:flex;align-items:center;justify-content:center;color:var(--accent);font-weight:800;font-size:20px}
.brand-name{font-family:"Space Grotesk";font-weight:700;letter-spacing:.08em;font-size:1rem}
.brand-sub{color:var(--muted);font-size:.86rem;margin-top:3px}
.status{border:1px solid #2b5149;border-radius:999px;padding:8px 13px;color:var(--accent);font-family:"IBM Plex Mono";font-size:.82rem;letter-spacing:.04em}
.status-demo{border:1px solid #ff984d;border-radius:999px;padding:8px 13px;color:#ff984d;font-family:"IBM Plex Mono";font-size:.82rem;letter-spacing:.04em}
.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:30px;align-items:center;padding:28px 0 34px}
.eyebrow{font-family:"IBM Plex Mono";font-size:.82rem;letter-spacing:.12em;color:var(--accent);font-weight:600}
.hero h1{font-size:2.5rem;line-height:1.08;margin:.7rem 0 1rem;letter-spacing:-.02em}
.hero h1 span{color:var(--accent)}
.hero p{max-width:760px;color:var(--muted);font-size:1rem;line-height:1.7}
.meta-row{display:flex;gap:9px;flex-wrap:wrap;margin-top:22px}
.meta{border:1px solid var(--line);background:rgba(12,26,32,.7);padding:9px 12px;border-radius:999px;color:#b8c8cc;font-size:.86rem}
.orbit{min-height:260px;position:relative;display:flex;align-items:center;justify-content:center}
.orbit:before,.orbit:after{content:"";position:absolute;border:1px solid #1f3d40;border-radius:50%}
.orbit:before{width:230px;height:230px}
.orbit:after{width:150px;height:150px}
.core{width:94px;height:94px;border-radius:50%;border:1px solid var(--accent);box-shadow:0 0 45px rgba(66,214,168,.12);display:flex;align-items:center;justify-content:center;text-align:center;z-index:2;background:#08161a}
.core b{display:block;color:var(--accent);font-family:"Space Grotesk";font-size:1.35rem}
.core small{display:block;color:var(--muted);font-family:"IBM Plex Mono";font-size:.48rem;letter-spacing:.08em}
.node{position:absolute;border:1px solid #2b4d50;background:#08161a;padding:7px 10px;color:#b9ccce;font-family:"IBM Plex Mono";font-size:.58rem;letter-spacing:.06em}
.n1{top:8px;left:12%}.n2{right:5%;top:43%}.n3{left:13%;bottom:9%}
.section-head{display:flex;justify-content:space-between;gap:20px;align-items:end;margin:18px 0 12px}
.kicker{font-family:"IBM Plex Mono";font-size:.82rem;letter-spacing:.12em;color:var(--accent);font-weight:600}
.section-head h2{font-size:1.65rem;line-height:1.2;margin:.35rem 0 0}
.section-head p{max-width:650px;color:var(--muted);font-size:.96rem;line-height:1.55;margin:0}
.panel{background:linear-gradient(145deg,rgba(12,26,32,.97),rgba(8,18,22,.98));border:1px solid var(--line);border-radius:17px;padding:17px;box-shadow:0 12px 35px rgba(0,0,0,.13)}
.panel-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:13px}
.panel-kicker{font-family:"IBM Plex Mono";font-size:.8rem;color:var(--accent);letter-spacing:.1em}
.panel h3{font-size:1.2rem;line-height:1.25;margin:.3rem 0 0}
.count{border:1px solid var(--line);border-radius:999px;padding:6px 10px;color:#a8babe;font-size:.84rem;font-family:"IBM Plex Mono"}
.metric-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}
.metric{min-height:108px}
.executive-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.executive-metric{min-height:104px}
.executive-metric.index{border-color:#2f7868;background:linear-gradient(145deg,rgba(19,54,51,.98),rgba(8,23,25,.98))}
.executive-metric.index .metric-value{color:var(--accent)}
.metric-note{color:#9ab0b5;font-size:.86rem;line-height:1.5;margin-top:6px}
.kpi-section{margin:22px 0 8px}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kpi-card{min-height:116px}
.kpi-card.priority{grid-column:span 2}
.kpi-card .metric-value{font-size:1.55rem}
.kpi-help{color:#9ab0b5;font-size:.84rem;line-height:1.5;margin-top:7px}
.priority-list{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}
.priority-item{border:1px solid var(--line);border-radius:999px;padding:6px 9px;font-family:"IBM Plex Mono";font-size:.84rem}
.distribution-row{display:flex;align-items:center;gap:10px;margin:11px 0}
.distribution-label{width:90px;font-family:"IBM Plex Mono";font-size:.84rem;color:#b8c8cc}
.distribution-track{height:8px;flex:1;background:#13272d;border-radius:999px;overflow:hidden}
.distribution-fill{height:100%;border-radius:999px}
.distribution-count{width:38px;text-align:right;font-family:"IBM Plex Mono";font-size:.86rem;color:var(--text)}
.coverage-row{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #173039;padding:12px 0;color:#b8c8cc;font-size:.94rem}
.coverage-row:last-child{border-bottom:0}
.coverage-value{font-family:"IBM Plex Mono";color:var(--accent);font-weight:700}
.top-risk{border:1px solid var(--line);border-radius:12px;background:#09151a;padding:12px;margin:8px 0}
.top-risk-head{display:flex;justify-content:space-between;gap:10px;align-items:center}
.top-risk-meta{color:var(--muted);font-size:.88rem;line-height:1.45;margin-top:7px}
.system-status{border:1px solid #23434a;border-radius:14px;background:rgba(9,22,27,.86);padding:13px 15px;margin:14px 0 20px}
.system-status-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:10px}
.system-status-title{font-family:"IBM Plex Mono";font-size:.82rem;letter-spacing:.12em;color:var(--accent);font-weight:700}
.system-status-note{color:var(--muted);font-size:.86rem}
.system-status-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.system-status-item{border-left:2px solid #315258;padding:8px 9px;background:#08161a;min-height:62px}
.system-status-item.ready{border-left-color:#45d6a8}.system-status-item.warning{border-left-color:#ff984d}.system-status-item.unknown{border-left-color:#71878e}
.system-status-name{font-family:"IBM Plex Mono";font-size:.82rem;color:#b8c8cc;letter-spacing:.04em}
.system-status-state{font-weight:700;font-size:.9rem;margin-top:5px}.system-status-detail{color:#9ab0b5;font-size:.8rem;line-height:1.4;margin-top:4px}
.processing-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.explorer-toolbar{border:1px solid var(--line);border-radius:14px;background:#09151a;padding:14px;margin:12px 0 16px}
.explorer-summary{display:flex;align-items:center;gap:10px;flex-wrap:wrap;border-bottom:1px solid #173039;padding-bottom:10px}
.explorer-fields{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:12px}
.explorer-field{border-left:2px solid #21424a;padding:7px 10px;background:#09161a;min-height:48px}
.explorer-field-label{font-family:"IBM Plex Mono";font-size:.8rem;letter-spacing:.06em;color:var(--muted);text-transform:uppercase}
.explorer-field-value{font-size:.96rem;color:var(--text);line-height:1.5;margin-top:5px}
.explorer-evidence{border-top:1px solid #173039;margin-top:14px;padding-top:12px}
.explorer-evidence-title{font-family:"IBM Plex Mono";font-size:.82rem;letter-spacing:.08em;color:var(--accent);margin-bottom:10px}
.merged-observation{border:1px solid #1d3940;border-radius:9px;padding:9px;margin:7px 0;background:#081317}
.replay-panel{border:1px solid #2a5453;background:linear-gradient(145deg,rgba(10,30,32,.98),rgba(8,18,22,.98));border-radius:16px;padding:17px;margin:14px 0}
.replay-meta{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0 14px}
.timeline{display:flex;align-items:stretch;gap:0;overflow-x:auto;padding:8px 0 13px;margin-top:10px}
.timeline-step{min-width:118px;position:relative;text-align:center;padding:0 8px}
.timeline-step:after{content:"";position:absolute;top:30px;left:50%;right:-50%;height:1px;background:#315258;z-index:0}
.timeline-step:last-child:after{display:none}
.timeline-dot{position:relative;z-index:1;width:14px;height:14px;border-radius:50%;margin:9px auto 8px;border:2px solid #6d8e90;background:#081317}
.timeline-dot.incident{border-color:#ff984d;background:#ff984d;box-shadow:0 0 0 5px rgba(255,152,77,.12)}
.timeline-time{font-family:"IBM Plex Mono";font-size:.86rem;color:#d3e2df}
.timeline-label{font-family:"IBM Plex Mono";font-size:.8rem;color:var(--muted);letter-spacing:.06em;margin-top:4px}
.replay-copy{color:#b8c8cc;font-size:.98rem;line-height:1.6}
.comparison-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
.comparison-label{font-family:"IBM Plex Mono";font-size:.88rem;letter-spacing:.08em;color:var(--accent);font-weight:700;margin-bottom:8px}
.comparison-caption{color:var(--muted);font-size:.86rem;line-height:1.5;margin-top:8px}
.metric-label{font-family:"IBM Plex Mono";font-size:.8rem;color:var(--muted);letter-spacing:.08em}
.metric-value{font-family:"Space Grotesk";font-size:1.9rem;font-weight:700;margin-top:8px}
.metric-sub{color:#9ab0b5;font-size:.84rem;margin-top:5px}
.riskline{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.pill{font-family:"IBM Plex Mono";font-size:.86rem;font-weight:700;border-radius:999px;padding:6px 10px;border:1px solid currentColor}
.incident{border:1px solid var(--line);background:#09151a;border-radius:13px;padding:13px;margin:8px 0}
.incident:hover{border-color:#31555a}
.alert-card{border:1px solid #ff984d;background:linear-gradient(145deg,rgba(40,20,10,.85),rgba(20,10,5,.9));border-radius:15px;padding:16px;margin:16px 0}
.alert-title{font-family:"IBM Plex Mono";color:#ff984d;font-size:.84rem;letter-spacing:.12em;font-weight:700}
.small{color:var(--muted);font-size:.86rem;line-height:1.45}
.beh-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.beh{border:1px solid var(--line);border-radius:12px;padding:12px;background:#09151a}
.beh strong{font-size:1.08rem;line-height:1.3}.beh .bar{height:5px;background:#13272d;border-radius:999px;margin-top:10px;overflow:hidden}.beh .fill{height:100%;background:var(--accent)}
.guidance{border-left:2px solid var(--accent);background:#09161a;border-radius:0 11px 11px 0;padding:12px 14px;margin:8px 0}
.copilot{border:1px solid #29534d;background:linear-gradient(135deg,#0c1b20,#081519);border-radius:16px;padding:17px}
.copilot-title{font-family:"IBM Plex Mono";color:var(--accent);font-size:.82rem;letter-spacing:.1em}
.copilot-answer{border:1px solid #23443f;background:#09181a;border-radius:12px;padding:14px;margin-top:12px}
.evidence-card{border:1px solid var(--line);border-radius:13px;padding:11px;background:#081317;margin-bottom:12px}
.evidence-card img{border-radius:9px}
.footer{border-top:1px solid var(--line);margin-top:30px;padding:20px 0;color:#8da3a8;font-size:.82rem;line-height:1.5}
div[data-testid="stFileUploader"]{border:1px dashed #31545a;border-radius:13px;padding:3px;background:#081418}
.stButton>button{border-radius:10px!important;border:1px solid #29474c!important;background:#0c1c21!important;color:#e7f4f1!important;font-weight:700!important;font-size:.94rem!important;min-height:2.55rem!important;padding:7px 14px!important}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important}
.stButton>button[kind="primary"]{background:var(--accent)!important;color:#03100c!important;border-color:var(--accent)!important}
.stTabs [data-baseweb="tab"]{font-family:"IBM Plex Mono";font-size:.9rem;padding:10px 14px}
.stSelectbox label,.stRadio label{font-size:.94rem!important;font-weight:600!important}
[data-baseweb="select"] *{font-size:.94rem!important}
.stCaption, [data-testid="stCaptionContainer"]{font-size:.86rem!important;line-height:1.45!important}
textarea,input{font-size:.94rem!important}
.stProgress>div>div{background:var(--accent)!important}
[data-testid="stMetricValue"]{color:var(--text)}
@media(max-width:1000px){.hero,.workspace{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:1000px){.executive-grid,.processing-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:700px){.explorer-fields{grid-template-columns:1fr}}
@media(max-width:700px){.comparison-grid{grid-template-columns:1fr}}
@media(max-width:1000px){.kpi-grid{grid-template-columns:repeat(2,1fr)}.kpi-card.priority{grid-column:span 2}}
@media(max-width:700px){.kpi-grid{grid-template-columns:1fr}.kpi-card.priority{grid-column:span 1}}
@media(max-width:1000px){.system-status-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:700px){.system-status-grid{grid-template-columns:1fr}}
</style>
""",
    unsafe_allow_html=True,
)

def esc(value):
    return html.escape(str(value))


def calculate_safety_index(incidents):
    """Return the deterministic AI-derived project metric for unique incidents."""
    if not incidents:
        return 100.0
    average_score = sum(float(event.get("risk_score", 0)) for event in incidents) / len(incidents)
    return round(max(0.0, min(100.0, 100.0 - average_score)), 1)


def optional_value(value):
    if value is None or value == "":
        return "Not available"
    return str(value)


def evidence_paths(incident):
    frames = incident.get("evidence_frames")
    if isinstance(frames, list) and frames:
        candidates = frames
    elif incident.get("evidence_frame"):
        candidates = [incident.get("evidence_frame")]
    else:
        candidates = []
    return list(dict.fromkeys(frame for frame in candidates if frame))


def incident_matches(incident, risk_filter, behaviour_filter, search_term):
    if risk_filter != "All" and incident.get("risk") != risk_filter:
        return False
    if behaviour_filter != "All" and incident.get("behaviour") != behaviour_filter:
        return False
    searchable = " ".join(
        optional_value(incident.get(field))
        for field in ("id", "behaviour", "explanation", "why_risky", "recommended_action")
    ).lower()
    return search_term.lower().strip() in searchable


def evidence_records(incident):
    frames = evidence_paths(incident)
    timestamps = {}
    observations = incident.get("incidents") or [incident, *(incident.get("merged_incidents") or [])]
    for observation in observations:
        observation_timestamp = observation.get("timestamp")
        if observation_timestamp is None:
            continue
        for frame in evidence_paths(observation):
            timestamps.setdefault(str(frame), float(observation_timestamp))

    records = []
    for order, frame in enumerate(frames):
        timestamp = timestamps.get(str(frame))
        if timestamp is None:
            match = re.search(r"_(\d+(?:\.\d+)?)s(?:\.[^.]+)?$", Path(str(frame)).name)
            if match:
                timestamp = float(match.group(1))
        records.append({"path": frame, "timestamp": timestamp, "order": order})
    return sorted(records, key=lambda record: (record["timestamp"] is None, record["timestamp"] or 0, record["order"]))


def format_timestamp(value):
    if value is None:
        return "Timestamp unavailable"
    return f"{float(value):.2f}s"


def evidence_workspace_items(event, duration):
    records = evidence_records(event)
    event_time = event.get("start_timestamp", event.get("timestamp"))
    context = context_timestamps(event_time, duration)
    incident_record = nearest_evidence(records, context["incident"])
    before_record = nearest_evidence(records, context["before"])
    after_record = nearest_evidence(records, context["after"])

    items = []
    for item_id, label, category, record, description, target in [
        ("before", "BEFORE", "BEFORE", before_record, "Context before the detected event", context["before"]),
        ("incident", "INCIDENT", "INCIDENT", incident_record, "Primary evidence for the detected behaviour", context["incident"]),
        ("after", "AFTER", "AFTER", after_record, "Post-event context; not proof of safety or damage", context["after"]),
    ]:
        item = {
            "item_id": item_id,
            "label": label,
            "category": category,
            "path": record.get("path") if record else None,
            "timestamp": record.get("timestamp") if record and record.get("timestamp") is not None else target,
            "description": description,
            "status": "Nearest available" if record and target is not None and record.get("timestamp") != target else "Primary context",
        }
        items.append(item)

    for index, record in enumerate(records, start=1):
        item_id = f"frame-{index}"
        if any(item["path"] == record.get("path") for item in items if item.get("path")):
            continue
        items.append({
            "item_id": item_id,
            "label": f"EVIDENCE FRAME {index}",
            "category": "OBSERVATION",
            "path": record.get("path"),
            "timestamp": record.get("timestamp"),
            "description": "Additional chronological evidence from this episode",
            "status": "Evidence library",
        })

    items.append({
        "item_id": "source-video",
        "label": "SOURCE VIDEO",
        "category": "VIDEO",
        "path": None,
        "timestamp": event_time,
        "description": "Original video replay positioned at the episode timestamp",
        "status": "Available" if st.session_state.get("video_bytes") else "No source video available",
    })
    return items


def get_system_health(is_demo=False):
    """Return presentation-only deployment health without initializing services."""
    checks = {}

    try:
        from engine.detectors import BehaviourReasoner
        from engine.dedup import merge_nearby_incidents
        from engine.risk import classify
        from engine.tracker import GreedyTracker
        checks["AI ENGINE"] = {"state": "READY", "detail": "Perception + tracking + risk pipeline available"}
    except Exception:
        checks["AI ENGINE"] = {"state": "NOT READY", "detail": "Analysis engine unavailable"}

    model_path = ROOT / "yolov8s-worldv2.pt"
    model_health = st.session_state.get("_model_health")
    if not model_path.exists():
        checks["MODEL"] = {"state": "NOT READY", "detail": "Configured model unavailable"}
    elif model_health == "READY":
        detail = "YOLOWorld loaded"
        if is_demo:
            detail = "YOLOWorld loaded; demo evidence is synthetic"
        checks["MODEL"] = {"state": "READY", "detail": detail}
    elif model_health == "NOT READY":
        checks["MODEL"] = {"state": "NOT READY", "detail": "Model could not be loaded"}
    else:
        detail = "Model loads when analysis starts"
        if is_demo:
            detail = "Not initialized; demo mode uses synthetic incidents"
        checks["MODEL"] = {"state": "NOT INITIALIZED", "detail": detail}

    try:
        from engine.video import VideoAnalyzer
        checks["VIDEO ANALYZER"] = {"state": "READY", "detail": "Video analysis pipeline available"}
    except Exception:
        checks["VIDEO ANALYZER"] = {"state": "NOT READY", "detail": "Video analysis unavailable"}

    try:
        if not callable(context_timestamps) or not callable(nearest_evidence):
            raise RuntimeError
        EVID.mkdir(exist_ok=True)
        checks["EVIDENCE SYSTEM"] = {"state": "READY", "detail": "Evidence and replay available"}
    except Exception:
        checks["EVIDENCE SYSTEM"] = {"state": "NOT READY", "detail": "Evidence and replay unavailable"}

    try:
        if not hasattr(st, "session_state"):
            raise RuntimeError
        checks["REVIEW SYSTEM"] = {"state": "READY", "detail": "Supervisor review persistence available"}
    except Exception:
        checks["REVIEW SYSTEM"] = {"state": "NOT READY", "detail": "Supervisor review unavailable"}

    return checks


def render_system_health(is_demo=False, analysis_loaded=False):
    checks = get_system_health(is_demo)
    state_class = {"READY": "ready", "NOT READY": "warning", "NOT INITIALIZED": "unknown"}
    state_marker = {"READY": "●", "NOT READY": "⚠", "NOT INITIALIZED": "○"}
    items = []
    for name in ["AI ENGINE", "MODEL", "VIDEO ANALYZER", "EVIDENCE SYSTEM", "REVIEW SYSTEM"]:
        check = checks.get(name, {"state": "UNKNOWN", "detail": "Status unavailable"})
        state = check.get("state", "UNKNOWN")
        css_class = state_class.get(state, "unknown")
        marker = state_marker.get(state, "○")
        items.append(
            f'<div class="system-status-item {css_class}"><div class="system-status-name">{marker} {name}</div>'
            f'<div class="system-status-state">{esc(state)}</div><div class="system-status-detail">{esc(check.get("detail", "Status unavailable"))}</div></div>'
        )
    analysis_status = "Analysis loaded" if analysis_loaded else "Ready for video"
    if is_demo and analysis_loaded:
        analysis_status = "Demo analysis loaded"
    st.markdown(
        f'<div class="system-status"><div class="system-status-head"><div class="system-status-title">SYSTEM STATUS</div><div class="system-status-note">Deployment readiness · Analysis status: {analysis_status}</div></div><div class="system-status-grid">{"".join(items)}</div></div>',
        unsafe_allow_html=True,
    )

def clear_analysis():
    for key in [
        "incidents", "meta", "copilot_answer", "copilot_question",
        "selected_incident", "video_bytes", "video_name", "video_path", "is_demo"
    ]:
        st.session_state.pop(key, None)

incidents = st.session_state.get("incidents", [])
meta = st.session_state.get("meta", {})
is_demo = meta.get("is_demo", st.session_state.get("is_demo", False))

status_html = (
    '<div class="status-demo">● DEMO MODE — SYNTHETIC INCIDENTS</div>'
    if is_demo else
    '<div class="status">● REAL AI PIPELINE · HUMAN REVIEW</div>'
)

# Topbar
st.markdown(
    f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-mark">G</div>
    <div>
      <div class="brand-name">GODREJ WAREHOUSE AI</div>
      <div class="brand-sub">Video Intelligence for Damage Prevention</div>
    </div>
  </div>
  {status_html}
</div>
""",
    unsafe_allow_html=True,
)

# Hero
st.markdown(
    """
<div class="hero">
  <div>
    <div class="eyebrow">VIDEO INTELLIGENCE • DAMAGE PREVENTION</div>
    <h1>Understand handling.<br><span>Prevent damage.</span></h1>
    <p>
      AI-powered warehouse video intelligence that identifies risky handling
      behaviour, explains potential risk and helps supervisors intervene
      before product damage occurs.
    </p>
    <div class="meta-row">
      <div class="meta">● Object detection</div>
      <div class="meta">● Persistent tracking</div>
      <div class="meta">● Behaviour reasoning</div>
      <div class="meta">● Deduplication</div>
      <div class="meta">● Risk intelligence</div>
    </div>
  </div>
  <div class="orbit">
    <div class="node n1">01 · PERCEIVE</div>
    <div class="node n2">02 · UNDERSTAND</div>
    <div class="node n3">03 · PREVENT</div>
    <div class="core"><div><b>AI</b><small>FIELD<br>INTELLIGENCE</small></div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Section header
st.markdown(
    """
<div class="section-head">
  <div><div class="kicker">VIDEO INTELLIGENCE</div><h2>Analyze warehouse footage</h2></div>
  <p>Upload recorded warehouse footage. The AI pipeline analyzes people,
  products, movement and temporal behaviour patterns.</p>
</div>
""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Warehouse footage",
    type=["mp4", "mov", "avi", "mkv", "m4v"],
    label_visibility="collapsed",
)

if uploaded:
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    path = OUT / ("input_video" + suffix)
    data = uploaded.getvalue()
    path.write_bytes(data)

    st.session_state["video_bytes"] = data
    st.session_state["video_name"] = uploaded.name
    st.session_state["video_path"] = str(path)

    col_video, col_side = st.columns([2.5, 1], gap="small")
    with col_video:
        st.markdown(
            f"""
<div class="panel">
  <div class="panel-head">
    <div><div class="panel-kicker">INCIDENT REPLAY</div><h3>{esc(uploaded.name)}</h3></div>
    <div class="count">{len(st.session_state.get("incidents", []))} unique events</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.video(data, format=f"video/{suffix.lstrip('.')}", start_time=0)
        st.caption("Evidence-linked playback · select an incident below to reopen the video near its timestamp.")

    with col_side:
        st.markdown(
            """
<div class="panel">
  <div class="panel-head">
    <div><div class="panel-kicker">VIDEO INPUT</div><h3>Warehouse footage</h3></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.success("Video loaded and playable.")
        st.caption(f"Source: {uploaded.name}")
        st.caption(f"Size: {len(data)/1024/1024:.1f} MB")

        with st.expander("AI analysis controls", expanded=False):
            stride = st.slider(
                "Detection stride", 1, 6, 3,
                help="Lower values improve temporal resolution but increase processing time.",
            )
            imgsz = st.select_slider("Inference size", [384, 512, 640], value=512)
            conf = st.slider("Perception confidence", 0.10, 0.50, 0.20, 0.05)

        run_live = st.button("▶  RUN FULL AI ANALYSIS", type="primary", use_container_width=True)
        run_demo = st.button("⚡  RUN DEMO MODE (SYNTHETIC)", use_container_width=True)

        if st.button("↻ Reset", use_container_width=True):
            clear_analysis()
            st.rerun()

    if run_live:
        progress = st.progress(0)
        status = st.empty()

        def cb(v):
            progress.progress(max(0.0, min(1.0, v)))
            status.caption(f"AI pipeline processing source video · {v*100:.0f}%")

        try:
            cached_model = get_cached_model("yolov8s-worldv2.pt")
            st.session_state["_model_health"] = "READY" if cached_model is not None else "NOT READY"
            analyzer = VideoAnalyzer(
                model_path="yolov8s-worldv2.pt",
                detection_stride=stride,
                imgsz=imgsz,
                confidence=conf,
                model=cached_model
            )
            incidents, meta = analyzer.analyze(path, OUT, cb, True)
            incidents = sorted(incidents, key=lambda x: x.get("timestamp", 0))
            st.session_state["incidents"] = incidents
            st.session_state["meta"] = meta
            st.session_state["is_demo"] = False
            st.session_state.pop("copilot_answer", None)
            progress.progress(1.0)
            status.success(
                f"Real AI Analysis complete · {meta.get('processing_s', 0)}s · "
                f"{meta.get('realtime_factor', 0)}× realtime"
            )
            st.rerun()
        except Exception as e:
            st.error(f"Real AI Analysis failed: {e}")
            st.info("You may explicitly run Demo Mode below to explore dashboard features.")

    if run_demo:
        demo_incidents, demo_meta = generate_demo_incidents(OUT, save_evidence=True)
        st.session_state["incidents"] = demo_incidents
        st.session_state["meta"] = demo_meta
        st.session_state["is_demo"] = True
        st.session_state.pop("copilot_answer", None)
        st.success("Loaded DEMO MODE — SYNTHETIC INCIDENTS")
        st.rerun()

# Current state
incidents = st.session_state.get("incidents", [])
meta = st.session_state.get("meta", {})
is_demo = meta.get("is_demo", st.session_state.get("is_demo", False))

if incidents or meta:
    unique_incidents = incidents
    event_episodes = build_event_episodes(unique_incidents)
    incidents = event_episodes
    raw_count = meta.get("raw_incident_count", len(unique_incidents))
    unique_count = meta.get("unique_incident_count", len(unique_incidents))
    merged_count = meta.get("merged_incident_count", 0)

    critical = sum(x.get("risk") == "CRITICAL" for x in incidents)
    high = sum(x.get("risk") == "HIGH" for x in incidents)
    medium = sum(x.get("risk") == "MEDIUM" for x in incidents)
    low = sum(x.get("risk") == "LOW" for x in incidents)

    avg_conf = sum(x.get("confidence", 0) for x in incidents) / max(1, len(incidents))
    avg_score = sum(x.get("risk_score", 0) for x in incidents) / max(1, len(incidents))

    behaviours = [x.get("behaviour") for x in incidents if x.get("behaviour")]
    counts = {}
    for b in behaviours:
        counts[b] = counts.get(b, 0) + 1
    covered = len(counts)
    duration = meta.get("duration_s", 0)

    mode_label = "DEMO MODE — SYNTHETIC INCIDENTS" if is_demo else "REAL AI ANALYSIS"
    safety_index = calculate_safety_index(unique_incidents)
    processing_time = float(meta.get("processing_s", 0) or 0)
    realtime_factor = float(meta.get("realtime_factor", 0) or 0)
    kpi_incident_rate = incident_rate(unique_count, duration)
    kpi_severe_share = high_critical_share(incidents)
    kpi_merge_rate = merged_observation_rate(raw_count, unique_count, incidents)
    kpi_scenario = scenario_coverage(incidents, SCENARIOS.keys())
    kpi_evidence = evidence_coverage(
        incidents,
        lambda event: (frame for frame in evidence_paths(event) if Path(frame).exists()),
    )
    kpi_mode_note = (
        "Calculated from synthetic demo incidents; not real warehouse performance."
        if is_demo else
        "Summarizes observed incidents in the analyzed video."
    )

    st.markdown(
        f"""
<div class="kpi-section">
    <div class="section-head"><div><div class="kicker">WAREHOUSE SAFETY AT A GLANCE</div><h2>Supervisor event episodes</h2></div><p>{kpi_mode_note} Episode counts group continuous behaviour for review.</p></div>
    <div class="kpi-grid">
        <div class="panel kpi-card"><div class="metric-label">EVENT EPISODES</div><div class="metric-value">{len(event_episodes)}</div><div class="kpi-help">Continuous behavioural episodes for supervisor review.</div></div>
        <div class="panel kpi-card"><div class="metric-label">CRITICAL EVENTS</div><div class="metric-value">{critical}</div><div class="kpi-help">Supervisor event episodes classified CRITICAL.</div></div>
        <div class="panel kpi-card"><div class="metric-label">HIGH-RISK EVENTS</div><div class="metric-value">{high}</div><div class="kpi-help">Supervisor event episodes classified HIGH.</div></div>
        <div class="panel kpi-card"><div class="metric-label">EVIDENCE COVERAGE</div><div class="metric-value">{"N/A" if kpi_evidence is None else f"{kpi_evidence:.1f}%"}</div><div class="kpi-help">Event episodes with at least one usable local evidence frame.</div></div>
        <div class="panel kpi-card"><div class="metric-label">BEHAVIOURS DETECTED</div><div class="metric-value">{kpi_scenario["detected"]}/{kpi_scenario["implemented"]}</div><div class="kpi-help">Not detected does not mean safe.</div></div>
        <div class="panel kpi-card"><div class="metric-label">ANALYSIS PERIOD</div><div class="metric-value">{duration:.2f}s</div><div class="kpi-help">Source video duration.</div></div>
    </div>
</div>
<div class="section-head">
    <div><div class="kicker">AI RISK SUMMARY · {mode_label}</div><h2>Supervisor event episodes</h2></div>
    <p>Risk counts below are calculated from event episodes. Backend incident metrics remain secondary.</p>
</div>
<div class="panel" style="display:flex;justify-content:space-around;gap:12px;flex-wrap:wrap">
    <span style="color:#ff4d6d"><b>● {critical}</b> Critical</span>
    <span style="color:#ff984d"><b>● {high}</b> High</span>
    <span style="color:#f4c95d"><b>● {medium}</b> Medium</span>
    <span style="color:#45d6a8"><b>● {low}</b> Low</span>
</div>
""",
        unsafe_allow_html=True,
    )

    render_system_health(is_demo=is_demo, analysis_loaded=True)

    # Latest severe risk alert banner
    severe_events = [x for x in incidents if x.get("risk") in {"CRITICAL", "HIGH"}]
    if severe_events:
        latest_severe = max(severe_events, key=lambda x: x.get("timestamp", 0))
        color = risk_color(latest_severe.get("risk"))
        alert_cols = st.columns([1.45, 1], gap="small")
        with alert_cols[0]:
            st.markdown(
                f"""
<div class="alert-card">
    <div class="alert-title">LATEST PRIORITY EVENT</div>
  <div style="margin-top:6px;font-size:1.1rem;font-weight:700">
    {esc(latest_severe.get('behaviour'))} · {esc(latest_severe.get('risk'))} · {esc(latest_severe.get('risk_score'))}/100
  </div>
    <div style="margin-top:7px;font-size:.94rem"><b>When:</b> {latest_severe.get('start_timestamp',0):.2f}s → {latest_severe.get('end_timestamp',0):.2f}s · <b>Duration:</b> {latest_severe.get('duration',0):.2f}s · <b>Confidence:</b> {latest_severe.get('confidence',0):.0%}</div>
    <div style="margin-top:9px;font-size:.96rem;line-height:1.5"><b>Observed:</b> {esc(latest_severe.get('explanation'))}</div>
    <div style="margin-top:7px;font-size:.94rem;line-height:1.5;color:#75e4c3"><b>Action:</b> {esc(latest_severe.get('recommended_action'))}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with alert_cols[1]:
            latest_frames = latest_severe.get("evidence_frames") or ([latest_severe.get("evidence_frame")] if latest_severe.get("evidence_frame") else [])
            latest_frame = next((frame for frame in latest_frames if frame and Path(frame).exists()), None)
            if latest_frame:
                st.image(latest_frame, caption=f"Primary evidence · {latest_severe.get('evidence_count', 0)} episode frame(s)", use_container_width=True)
            else:
                st.info("No evidence frame was saved for the latest severe incident.")
    else:
        st.info("No HIGH or CRITICAL incidents were produced by this analysis.")

    st.markdown('<div class="section-head"><div><div class="kicker">PRIORITY EVENTS</div><h2>Highest-priority event episodes</h2></div><p>Ranked from supervisor event episodes by severity, then risk score.</p></div>', unsafe_allow_html=True)
    risk_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    top_risks = sorted(incidents, key=lambda event: (risk_rank.get(event.get("risk"), 0), event.get("risk_score", 0), -event.get("timestamp", 0)), reverse=True)[:5]
    if top_risks:
        for rank, event in enumerate(top_risks, start=1):
            color = risk_color(event.get("risk"))
            st.markdown(
                f'<div class="top-risk"><div class="top-risk-head"><span><span class="pill" style="color:{color}">{esc(event.get("episode_id"))} · {esc(event.get("risk"))} · {esc(event.get("risk_score"))}/100</span> <strong>{esc(event.get("behaviour"))}</strong></span><span class="small">{event.get("start_timestamp") if event.get("start_timestamp") is not None else "—"} → {event.get("end_timestamp") if event.get("end_timestamp") is not None else "—"} · Conf {event.get("confidence", 0):.0%}</span></div><div class="top-risk-meta">{esc(event.get("explanation", ""))} · {event.get("observation_count", 0)} observations · {event.get("evidence_count", 0)} evidence frames</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No incidents to rank in this analysis.")

    st.markdown(
        """
<div class="section-head">
    <div><div class="kicker">BEHAVIOUR INTELLIGENCE</div><h2>Detected behaviours</h2></div>
    <p>Counts represent event episodes, not raw detector observations.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if counts:
        max_count = max(counts.values())
        beh_html = '<div class="beh-grid">'
        for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            width = int(100 * count / max_count)
            beh_html += (
                f'<div class="beh"><strong>{esc(name)}</strong>'
                f'<span class="small"> · {count} event(s)</span>'
                f'<div class="bar"><div class="fill" style="width:{width}%"></div></div></div>'
            )
        beh_html += "</div>"
        st.markdown(beh_html, unsafe_allow_html=True)

    tabs = st.tabs(["Incident Timeline", "Evidence Review", "AI Copilot", "Audit"])
    ordered = sorted(incidents, key=lambda x: x.get("timestamp", 0))

    with tabs[0]:
        st.markdown(
            '<div class="section-head"><div><div class="kicker">INCIDENT TIMELINE</div>'
            '<h2>Event Explorer</h2></div><p>Filter and review event episodes without rerunning video analysis.</p></div>',
            unsafe_allow_html=True,
        )

        filter_cols = st.columns([1, 1.7, 1.1, 2], gap="small")
        with filter_cols[0]:
            risk_filter = st.selectbox("Risk", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"], key="incident_risk_filter")
        with filter_cols[1]:
            behaviour_filter = st.selectbox("Behaviour", ["All", *SCENARIOS.keys()], key="incident_behaviour_filter")
        with filter_cols[2]:
            sort_order = st.selectbox("Sort", ["Highest risk", "Latest first", "Earliest first"], key="incident_sort_order")
        with filter_cols[3]:
            search_term = st.text_input("Search incidents", placeholder="ID, behaviour, explanation or action", key="incident_search")

        filtered = [
            event for event in incidents
            if incident_matches(event, risk_filter, behaviour_filter, search_term)
        ]
        if sort_order == "Highest risk":
            filtered.sort(key=lambda event: (event.get("risk_score", 0), event.get("timestamp", 0)), reverse=True)
        elif sort_order == "Latest first":
            filtered.sort(key=lambda event: event.get("timestamp", 0), reverse=True)
        else:
            filtered.sort(key=lambda event: event.get("timestamp", 0))

        st.markdown(
            f'<div class="explorer-toolbar"><div class="explorer-summary"><strong>{len(filtered)} event episode(s)</strong><span class="small">of {len(incidents)} supervisor episode(s)</span><span class="small">· Search covers episode ID, behaviour, explanation and recommended action.</span></div></div>',
            unsafe_allow_html=True,
        )

        selected_event = None
        if filtered:
            options = [
                f"{e.get('episode_id', f'EP-{i + 1:03d}')} · {e.get('risk','LOW')} · {e.get('behaviour','Event')}"
                for i, e in enumerate(filtered)
            ]
            selected = st.selectbox("Select event episode to review", options, index=0)
            selected_event = filtered[options.index(selected)]
            st.session_state["selected_episode_id"] = selected_event.get("episode_id")

            # Detailed evidence is rendered once in the Evidence Review tab.
            legacy_detail_removed = True
            if legacy_detail_removed:
                st.info("Open the Evidence Review tab for the selected episode's review board.")
            _legacy_detail_markup = """

            selected_timestamp = selected_event.get("timestamp")
            selected_timestamp_label = f"{float(selected_timestamp):.2f}s" if selected_timestamp is not None else "Timestamp not available"
            selected_records = evidence_records(selected_event)
            selected_known_records = [record for record in selected_records if record["timestamp"] is not None]
            selected_points = []
            if selected_timestamp is not None:
                selected_points.append((float(selected_timestamp), "INCIDENT", True))
            for record in selected_known_records:
                point = (record["timestamp"], "EVIDENCE", abs(record["timestamp"] - float(selected_timestamp)) < 0.001 if selected_timestamp is not None else False)
                if not any(existing[0] == point[0] and existing[1] == point[1] for existing in selected_points):
                    selected_points.append(point)
            selected_points.sort(key=lambda point: point[0])

            st.markdown('<div class="replay-panel">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel-kicker">EVENT {esc(selected_event.get("episode_id"))} · {esc(mode_label)}</div><h3>{esc(optional_value(selected_event.get("behaviour")))}</h3>'
                f'<div class="replay-meta"><span class="pill" style="color:{risk_color(selected_event.get("risk"))}">{esc(optional_value(selected_event.get("risk")))}</span><span class="small">Risk {esc(optional_value(selected_event.get("risk_score")))}/100</span><span class="small">Confidence {optional_value(selected_event.get("confidence")) if selected_event.get("confidence") is None else f"{selected_event.get('confidence'):.1%}"}</span><span class="small">{selected_timestamp_label}</span><span class="small">Status {esc(optional_value(selected_event.get("status")))}</span><span class="small">Damage confirmed: {"Yes" if selected_event.get("damage_confirmed") is True else "No" if selected_event.get("damage_confirmed") is False else "Not available"}</span></div>',
                unsafe_allow_html=True,
            )
            if selected_points:
                timeline_html = '<div class="timeline">'
                for point_timestamp, label, is_incident in selected_points:
                    timeline_html += f'<div class="timeline-step"><div class="timeline-time">{point_timestamp:.2f}s</div><div class="timeline-dot{" incident" if is_incident else ""}"></div><div class="timeline-label">{"INCIDENT" if is_incident else label}</div></div>'
                timeline_html += '</div>'
                st.markdown(timeline_html, unsafe_allow_html=True)
                st.caption("The highlighted marker is the actual incident timestamp. Other markers come from merged observation or evidence metadata.")
            else:
                st.info("Incident timestamp not available; evidence can still be reviewed below.")

            comparison = context_timestamps(selected_timestamp, meta.get("duration_s"))
            incident_record = nearest_evidence(selected_records, comparison["incident"])
            before_record = nearest_evidence(selected_records, comparison["before"])
            after_record = nearest_evidence(selected_records, comparison["after"])
            source_video_available = bool(st.session_state.get("video_bytes"))

            st.markdown('<div class="comparison-grid">', unsafe_allow_html=True)
            comparison_cards = [
                ("BEFORE — normal/preceding state", "Context before detected event", comparison["before"], before_record, False),
                ("INCIDENT — detected unsafe behaviour", "Primary evidence for the flagged event", comparison["incident"], incident_record, True),
                ("AFTER — post-event context", "Post-event context; not proof of damage", comparison["after"], after_record, False),
            ]
            comparison_cols = st.columns(3, gap="small")
            for card_col, (label, caption, target_time, evidence_record, is_incident_card) in zip(comparison_cols, comparison_cards):
                with card_col:
                    st.markdown(f'<div class="comparison-label">{label}</div>', unsafe_allow_html=True)
                    if target_time is not None:
                        st.caption(f"{target_time:.2f}s")
                    evidence_available = bool(evidence_record and evidence_record.get("path") and Path(evidence_record["path"]).exists())
                    if is_incident_card and evidence_available:
                        evidence_caption = "Primary evidence frame"
                        if evidence_record.get("timestamp") is not None:
                            evidence_caption += f" · {evidence_record['timestamp']:.2f}s"
                        st.image(evidence_record["path"], caption=evidence_caption, use_container_width=True)
                    elif source_video_available and target_time is not None:
                        st.video(
                            st.session_state["video_bytes"],
                            format="video/mp4",
                            start_time=max(0, int(target_time)),
                        )
                    elif evidence_available:
                        evidence_caption = "Nearest available evidence frame"
                        if evidence_record.get("timestamp") is not None:
                            evidence_caption += f" · {evidence_record['timestamp']:.2f}s"
                        st.image(evidence_record["path"], caption=evidence_caption, use_container_width=True)
                    elif evidence_record:
                        st.info("Evidence frame not available locally.")
                    else:
                        st.info("Context unavailable.")
                    st.markdown(f'<div class="comparison-caption">{caption}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.caption("Before and after views provide temporal context around the AI-detected event. They do not by themselves prove safety or physical damage.")

            review_key = f"episode_review_{selected_event.get('episode_id')}"
            review_status = st.selectbox(
                "Supervisor review",
                ["Needs Review", "Confirmed", "False Positive"],
                index=["Needs Review", "Confirmed", "False Positive"].index(st.session_state.get(review_key, "Needs Review")),
                key=f"review_widget_{selected_event.get('episode_id')}",
            )
            st.session_state[review_key] = review_status

            replay_cols = st.columns([1.2, 1], gap="small")
            with replay_cols[0]:
                if source_video_available and selected_timestamp is not None:
                    window_start = max(0, float(selected_timestamp) - 1.0)
                    st.caption(f"Source replay window: {window_start:.2f}s to {comparison['after']:.2f}s.")
                else:
                    st.info("Original video replay is unavailable. Evidence-frame comparison remains available.")
            with replay_cols[1]:
                damage_confirmed = selected_event.get("damage_confirmed")
                if damage_confirmed is False:
                    semantic_note = "Potential risk detected — physical damage not confirmed."
                elif damage_confirmed is True:
                    semantic_note = "Physical damage is marked confirmed in the incident record."
                else:
                    semantic_note = "Damage confirmation is not available in the incident record."
                st.markdown(
                    f'<div class="replay-copy"><b>WHY DETECTED</b><br>{esc(optional_value(selected_event.get("explanation")))}<br><br><b>WHY RISKY</b><br>{esc(optional_value(selected_event.get("why_risky")))}<br><br><b>RECOMMENDED ACTION</b><br>{esc(optional_value(selected_event.get("recommended_action")))}<br><br><b>SEMANTICS</b><br>{esc(semantic_note)}</div>',
                    unsafe_allow_html=True,
                )

            existing_selected_frames = [record for record in selected_records if record["path"] and Path(record["path"]).exists()]
            st.markdown(
                f'<div class="explorer-evidence"><div class="explorer-evidence-title">EVIDENCE REPLAY · {len(selected_records)} FRAME(S) · CHRONOLOGICAL ORDER</div></div>',
                unsafe_allow_html=True,
            )
            if existing_selected_frames:
                evidence_cols = st.columns(min(3, len(existing_selected_frames)))
                for frame_index, record in enumerate(existing_selected_frames):
                    record_timestamp = record["timestamp"]
                    frame_time = f"{record_timestamp:.2f}s" if record_timestamp is not None else "Timestamp not available"
                    evidence_cols[frame_index % len(evidence_cols)].image(record["path"], caption=f"Frame {frame_index + 1} · {frame_time}", use_container_width=True)
            elif selected_records:
                st.caption("Evidence paths exist in the incident record, but no corresponding local frame is available.")
            else:
                st.caption("Evidence frame not available.")
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Technical observations", expanded=False):
                st.caption(f"{selected_event.get('observation_count', 0)} observations grouped into this event")
                observation_times = []
                for observation in selected_event.get("incidents", []):
                    for nested in observation.get("merged_incidents") or [observation]:
                        timestamp = nested.get("timestamp")
                        if timestamp is not None:
                            observation_times.append(float(timestamp))
                if observation_times:
                    st.write(" · ".join(f"{timestamp:.2f}s" for timestamp in sorted(observation_times)))
                st.caption("Original incident IDs: " + ", ".join(f"#{value}" for value in selected_event.get("original_incident_ids", [])))

            if is_demo:
                st.caption("SYNTHETIC DEMO EVIDENCE · This analysis is explicitly labeled DEMO MODE — SYNTHETIC INCIDENTS.")

            """

        if not filtered:
            st.info("No incidents match the selected filters or search term.")

        for e in []:
            risk = "LOW"
            color = risk_color(risk)
            risk_score = 0
            timestamp_label = "Not available"
            with st.expander("", expanded=False):
                st.markdown(
                    f'<div class="explorer-summary"><span class="pill" style="color:{color}">{esc(risk)}</span><strong>{esc(e.get("behaviour", "Unknown behaviour"))}</strong><span class="small">{timestamp_label} · Risk {esc(risk_score)}/100 · Confidence {optional_value(e.get("confidence")) if e.get("confidence") is None else f"{e.get('confidence'):.1%}"}</span></div>',
                    unsafe_allow_html=True,
                )

                damage_confirmed = e.get("damage_confirmed")
                damage_label = "Yes" if damage_confirmed is True else "No" if damage_confirmed is False else "Not available"
                fields = [
                    ("Incident ID", e.get("id")),
                    ("Confidence", optional_value(e.get("confidence")) if e.get("confidence") is None else f"{e.get('confidence'):.1%}"),
                    ("Track ID", e.get("track_id")),
                    ("Related track ID", e.get("related_track_id")),
                    ("Status", e.get("status")),
                    ("Damage confirmed", damage_label),
                    ("Detection source", e.get("detection_source")),
                    ("Timestamp", timestamp_label),
                    ("Merged observation count", e.get("observation_count", len(e.get("merged_incidents") or []))),
                    ("Evidence frame count", len(evidence_paths(e))),
                ]
                field_html = '<div class="explorer-fields">'
                for label, value in fields:
                    field_html += f'<div class="explorer-field"><div class="explorer-field-label">{label}</div><div class="explorer-field-value">{esc(optional_value(value))}</div></div>'
                field_html += '</div>'
                st.markdown(field_html, unsafe_allow_html=True)

                text_fields = [
                    ("Why detected", e.get("explanation")),
                    ("Why risky", e.get("why_risky")),
                    ("Recommended action", e.get("recommended_action")),
                ]
                text_html = '<div class="explorer-fields">'
                for label, value in text_fields:
                    text_html += f'<div class="explorer-field"><div class="explorer-field-label">{label}</div><div class="explorer-field-value">{esc(optional_value(value))}</div></div>'
                text_html += '</div>'
                st.markdown(text_html, unsafe_allow_html=True)

                frames = evidence_paths(e)
                existing_frames = [frame for frame in frames if Path(frame).exists()]
                st.markdown(
                    f'<div class="explorer-evidence"><div class="explorer-evidence-title">EVIDENCE · {len(frames)} AVAILABLE FRAME(S)</div></div>',
                    unsafe_allow_html=True,
                )
                if existing_frames:
                    image_cols = st.columns(min(3, len(existing_frames)))
                    for frame_index, frame in enumerate(existing_frames):
                        image_cols[frame_index % len(image_cols)].image(frame, caption=f"Frame {frame_index + 1}", use_container_width=True)
                elif frames:
                    st.caption("Evidence paths exist in the incident record, but no corresponding local frame is available.")
                else:
                    st.caption("No evidence frame is available for this incident.")

                merged_observations = e.get("merged_incidents") or []
                if len(merged_observations) > 1:
                    st.markdown(f'<div class="explorer-evidence-title">MERGED OBSERVATIONS · {len(merged_observations)}</div>', unsafe_allow_html=True)
                    for observation in merged_observations:
                        observation_time = observation.get("timestamp")
                        observation_time = f"{float(observation_time):.2f}s" if observation_time is not None else "Not available"
                        st.markdown(
                            f'<div class="merged-observation"><b>{esc(optional_value(observation.get("behaviour")))}</b> · {esc(optional_value(observation.get("risk")))} · {esc(optional_value(observation.get("risk_score")))}/100 · {observation_time} · Track #{esc(optional_value(observation.get("track_id")))}</div>',
                            unsafe_allow_html=True,
                        )

                technical = {
                    "evidence": e.get("evidence", {}),
                    "observation_count": e.get("observation_count"),
                    "analysis_latency_s": e.get("analysis_latency_s"),
                    "source_frame": e.get("source_frame"),
                    "is_demo": e.get("is_demo"),
                    "mode": e.get("mode"),
                }
                st.markdown('<div class="explorer-evidence-title">TECHNICAL DETAILS</div>', unsafe_allow_html=True)
                st.json(technical)

    with st.expander("Technical analysis", expanded=False):
        st.caption("Engineering metrics for technical review. Existing KPI definitions are unchanged.")
        technical_metrics = [
            ("RAW OBSERVATIONS", raw_count),
            ("UNIQUE AI INCIDENTS", unique_count),
            ("MERGED OBSERVATIONS", merged_count),
            ("INCIDENT RATE", f"{kpi_incident_rate:.2f}/min"),
            ("FRAMES", meta.get("frames", 0)),
            ("FPS", f"{float(meta.get('fps', 0) or 0):.2f}"),
            ("PROCESSING TIME", f"{processing_time:.2f}s"),
            ("REALTIME FACTOR", f"{realtime_factor:.2f}x"),
            ("AI-DERIVED SAFETY INDEX", f"{safety_index:.1f}/100"),
        ]
        technical_html = '<div class="processing-grid">'
        for label, value in technical_metrics:
            technical_html += f'<div class="panel executive-metric"><div class="metric-label">{label}</div><div class="metric-value">{esc(value)}</div></div>'
        technical_html += '</div>'
        st.markdown(technical_html, unsafe_allow_html=True)
        st.caption("AI-derived Safety Index: max(0, 100 minus mean risk score across unique AI incidents). This is not an industry-standard safety rating.")

    with tabs[1]:
        st.markdown(
            '<div class="section-head"><div><div class="kicker">EVIDENCE REVIEW</div>'
            '<h2>Evidence Review</h2></div><p>Highest-priority event episodes requiring supervisor attention.</p></div>',
            unsafe_allow_html=True,
        )
        risk_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        priority_events = sorted(
            incidents,
            key=lambda event: (
                risk_rank.get(event.get("risk"), 0),
                float(event.get("risk_score", 0) or 0),
                -(float(event.get("start_timestamp")) if event.get("start_timestamp") is not None else float("inf")),
            ),
            reverse=True,
        )[:3]
        if not priority_events:
            st.info("No priority event episodes are available for evidence review.")
        else:
            def priority_label(event):
                marker = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(event.get("risk"), "○")
                timestamp = event.get("start_timestamp", event.get("timestamp"))
                timestamp_label = format_timestamp(timestamp).replace("s", "")
                return f"{marker} {event.get('risk', 'UNKNOWN')} · {event.get('behaviour', 'Event')} · {timestamp_label} · {event.get('episode_id', 'EP-???')}"

            priority_options = [priority_label(event) for event in priority_events]
            selected_priority = st.selectbox("Select Priority Event Episode", priority_options, index=0, key="priority_event_episode")
            selected_event = priority_events[priority_options.index(selected_priority)]
            selected_records = evidence_records(selected_event)
            selected_timestamp = selected_event.get("start_timestamp", selected_event.get("timestamp"))
            comparison = context_timestamps(selected_timestamp, meta.get("duration_s"))
            incident_record = nearest_evidence(selected_records, comparison["incident"])
            before_record = nearest_evidence(selected_records, comparison["before"])
            after_record = nearest_evidence(selected_records, comparison["after"])

            start = selected_event.get("start_timestamp")
            end = selected_event.get("end_timestamp")
            time_range = format_timestamp(start) if start == end else f"{format_timestamp(start)} → {format_timestamp(end)}"
            st.markdown(
                f'<div class="panel"><div class="panel-head"><div><div class="panel-kicker">{esc(selected_event.get("episode_id"))}</div><h3>{esc(selected_event.get("behaviour"))}</h3></div>'
                f'<div class="riskline"><span class="pill" style="color:{risk_color(selected_event.get("risk"))}">{esc(selected_event.get("risk"))}</span><span class="small">Score {esc(selected_event.get("risk_score"))} · Confidence {selected_event.get("confidence", 0):.0%}</span></div></div>'
                f'<div class="replay-meta"><span class="small">{time_range}</span><span class="small">{selected_event.get("observation_count", 0)} observations</span><span class="small">{selected_event.get("evidence_count", 0)} evidence frames</span></div></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-head"><div><div class="kicker">SOURCE VIDEO</div><h2>Review the analyzed footage</h2></div><p>Jump to the selected episode timestamp. The source video is reused from the current analysis session.</p></div>', unsafe_allow_html=True)
            source_video = st.session_state.get("video_bytes")
            source_video_path = st.session_state.get("video_path")
            if source_video is None and source_video_path and Path(source_video_path).exists():
                source_video = source_video_path
            if source_video is not None:
                source_name = st.session_state.get("video_name") or source_video_path or "analyzed source video"
                source_suffix = Path(str(source_name)).suffix.lower().lstrip(".") or "mp4"
                st.video(source_video, format=f"video/{source_suffix}", start_time=max(0, int(selected_timestamp or 0)))
                st.caption(f"Incident timestamp: {format_timestamp(selected_timestamp)} · Episode range: {time_range} · Use the player controls to review the surrounding context.")
                if is_demo:
                    st.warning("DEMO MODE — SYNTHETIC EVIDENCE. The displayed analysis is synthetic.")
            else:
                st.warning("Source video unavailable. Available evidence frames are shown below.")

            st.markdown('<div class="section-head"><div><div class="kicker">INCIDENT CONTEXT</div><h2>Before → Incident → After</h2></div><p>Evidence frames provide context around the selected priority episode.</p></div>', unsafe_allow_html=True)
            evidence_columns = st.columns(3, gap="small")
            for column, label, record, target_time, description in zip(
                evidence_columns,
                ["BEFORE", "INCIDENT", "AFTER"],
                [before_record, incident_record, after_record],
                [comparison["before"], comparison["incident"], comparison["after"]],
                ["Context before the event", "Primary evidence of the detected behaviour", "Post-event context; not proof of safety or damage"],
            ):
                with column:
                    st.markdown(f'<div class="comparison-label">{label}</div>', unsafe_allow_html=True)
                    record_path = record.get("path") if record else None
                    evidence_available = bool(record_path and Path(record_path).exists())
                    frame_time = record.get("timestamp") if record and record.get("timestamp") is not None else target_time
                    if evidence_available:
                        st.image(record_path, caption=f"{format_timestamp(frame_time)} · {description}", use_container_width=True)
                    elif st.session_state.get("video_bytes") and target_time is not None:
                        st.video(st.session_state["video_bytes"], format="video/mp4", start_time=max(0, int(target_time)))
                        st.caption(f"{format_timestamp(target_time)} · Source video context")
                    else:
                        st.info(f"No {label.lower()} frame available")
                    if record and target_time is not None and record.get("timestamp") != target_time:
                        st.caption("Nearest available evidence")

            st.markdown(
                f'<div class="explorer-fields">'
                f'<div class="explorer-field"><div class="explorer-field-label">Behaviour</div><div class="explorer-field-value">{esc(selected_event.get("behaviour"))}</div></div>'
                f'<div class="explorer-field"><div class="explorer-field-label">Risk level</div><div class="explorer-field-value">{esc(selected_event.get("risk"))}</div></div>'
                f'<div class="explorer-field"><div class="explorer-field-label">Risk score</div><div class="explorer-field-value">{esc(selected_event.get("risk_score"))}/100</div></div>'
                f'<div class="explorer-field"><div class="explorer-field-label">AI confidence</div><div class="explorer-field-value">{selected_event.get("confidence", 0):.0%}</div></div>'
                f'<div class="explorer-field"><div class="explorer-field-label">Timestamp</div><div class="explorer-field-value">{esc(time_range)}</div></div>'
                f'<div class="explorer-field"><div class="explorer-field-label">Track ID</div><div class="explorer-field-value">{esc(selected_event.get("track_id", "Not available"))}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            explanation_cols = st.columns(2, gap="small")
            for column, heading, field in zip(explanation_cols, ["WHAT AI OBSERVED", "WHY IT MATTERS"], ["explanation", "why_risky"]):
                with column:
                    st.markdown(f'<div class="guidance"><b>{heading}</b><br>{esc(optional_value(selected_event.get(field)))}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="guidance"><b>RECOMMENDED ACTION</b><br>{esc(optional_value(selected_event.get("recommended_action")))}</div>', unsafe_allow_html=True)

            if is_demo:
                st.warning("DEMO MODE — SYNTHETIC EVIDENCE")

            st.markdown("### SUPERVISOR REVIEW")
            review_key = f"episode_review_{selected_event.get('episode_id')}"
            review_options = ["Confirm Risk", "False Positive", "Needs Review"]
            stored_review = st.session_state.get(review_key, "Needs Review")
            display_review = "Confirm Risk" if stored_review == "Confirmed" else stored_review
            review_status = st.radio("Supervisor review", review_options, index=review_options.index(display_review), horizontal=True, key=f"priority_review_{selected_event.get('episode_id')}")
            st.session_state[review_key] = "Confirmed" if review_status == "Confirm Risk" else review_status
            ai_status = "Potential risk" if selected_event.get("status") == "POTENTIAL_RISK" else optional_value(selected_event.get("status"))
            agreement = "Pending" if review_status == "Needs Review" else "Agreement" if review_status == "Confirm Risk" else "Disagreement"
            st.caption(f"AI assessment: {ai_status} · Human assessment: {review_status} · {agreement}. Review feedback does not retrain the model.")

    with tabs[2]:
        st.markdown(
            f"""
<div class="copilot">
  <div class="copilot-title">◆ GEG AI COPILOT ({mode_label})</div>
  <h3>Supervisor Intelligence</h3>
  <div class="small">Ask questions grounded strictly in the incidents and evidence observed in this analysis.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        prompts = [
            "Give me the shift brief",
            "What are the most common risky behaviours?",
            "Show me the high and critical risks",
            "Why was the first incident flagged?",
        ]
        pc = st.columns(4)
        for i, prompt in enumerate(prompts):
            if pc[i].button(prompt, key=f"prompt_{i}", use_container_width=True):
                st.session_state["copilot_question"] = prompt

        q = st.text_input(
            "Ask the Copilot",
            value=st.session_state.get("copilot_question", "Give me the shift brief"),
        )
        if st.button("Ask Copilot", type="primary", use_container_width=False):
            st.session_state["copilot_answer"] = SupervisorAssistant().answer(q, ordered, meta)

        answer = st.session_state.get("copilot_answer")
        if answer:
            st.markdown(
                f'<div class="copilot-answer"><b>{esc(answer.get("title","Supervisor brief"))}</b>'
                f'<br><br>{esc(answer.get("summary",""))}</div>',
                unsafe_allow_html=True,
            )
            recs = answer.get("recommendations", [])
            if recs:
                st.markdown("**Recommended intervention**")
                for r in recs:
                    st.markdown(f'<div class="guidance">{esc(r)}</div>', unsafe_allow_html=True)
            if answer.get("timestamp") is not None:
                st.caption(f"Evidence timestamp: {answer['timestamp']:.2f}s")

    with tabs[3]:
        st.markdown(
            '<div class="section-head"><div><div class="kicker">AUDITABILITY</div>'
            '<h2>Analysis audit</h2></div><p>Machine-readable output for technical review, reproducibility and judge inspection.</p></div>',
            unsafe_allow_html=True,
        )
        audit = _portable_artifact({
            "meta": meta,
            "scenario_coverage": sorted(counts),
            "event_episodes": event_episodes,
            "incidents": unique_incidents,
        }, OUT)
        st.json(audit)
        st.download_button(
            "↓ Download incident audit JSON",
            json.dumps(audit, indent=2),
            file_name="godrej_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    render_system_health(is_demo=is_demo, analysis_loaded=False)
    st.markdown(
        """
<div class="panel" style="margin-top:18px">
  <div class="panel-kicker">READY FOR ANALYSIS</div>
  <h3>Turn warehouse footage into prevention intelligence.</h3>
  <p class="note">
    Upload recorded footage above. The system will detect open-vocabulary entities,
    maintain persistent track IDs, reason over temporal motion and spatial relationships,
    deduplicate nearby observations, assign bounded risk, and preserve timestamped evidence.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="footer">
  <b>GODREJ WAREHOUSE AI</b> · Video Intelligence for Safer Material Handling<br>
  Observed behaviour → Potential risk → Human intervention → Damage prevention<br><br>
  Responsible AI · Potential risk ≠ confirmed damage. Significant incidents require human review.
  The system is intended for process improvement and damage prevention, not automated punitive decisions.
</div>
""",
    unsafe_allow_html=True,
)
