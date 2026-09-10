# Challenge mapping

The challenge requires video ingestion, object detection/tracking, behaviour identification, risk classification, incident visualization, AI-generated explanation/recommendation, and at least 10 predefined behaviours/scenarios. It also emphasizes evidence-backed prevention and responsible AI.

This prototype maps those requirements to:

- Video ingestion: Streamlit upload + OpenCV
- Object perception: YOLOWorld open-vocabulary prompts
- Tracking: persistent greedy track IDs with temporal history
- Behaviour identification: 10+ temporal/spatial behaviour classes
- Risk: LOW/MEDIUM/HIGH/CRITICAL score with rationale
- Incident visualization: timestamped timeline + evidence images
- AI recommendation: evidence-grounded supervisor assistant
- Prevention framing: POTENTIAL_RISK and damage_confirmed=false
- Responsible AI: human review, no automated punitive decisions, no unsupported weight/damage claims
