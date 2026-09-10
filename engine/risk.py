from dataclasses import dataclass
from .config import RISK_ORDER

@dataclass
class RiskResult:
    risk: str
    score: int
    rationale: str


def classify(base_risk: str, confidence: float, persistence_s: float = 0.0, impact: float = 0.0, recurrence: int = 1) -> RiskResult:
    base = RISK_ORDER.get(base_risk, 2)
    score = base * 18 + int(max(0.0, min(1.0, confidence)) * 32)
    score += min(12, int(max(0.0, persistence_s) * 2.0))
    score += min(12, max(0, recurrence - 1) * 3)
    score += int(max(0.0, min(1.0, impact)) * 14)
    score = max(0, min(100, score))
    if score >= 76:
        risk = "CRITICAL"
    elif score >= 51:
        risk = "HIGH"
    elif score >= 26:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return RiskResult(risk, score, f"Base severity {base_risk}; visual/temporal confidence {confidence:.0%}; persistence {persistence_s:.1f}s; recurrence {recurrence}; score {score}/100.")
