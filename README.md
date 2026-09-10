# GEG Field Intelligence — Top-3 Submission Build

## Run

### Windows
Double-click `run.bat`.

### Linux/macOS
```bash
./run.sh
```

The first run creates a virtual environment, installs the dependencies and starts the local Streamlit dashboard. The YOLOWorld model is downloaded/cached automatically by Ultralytics on first analysis.

## Use

1. Upload any MP4/MOV/AVI/MKV/M4V warehouse/loading/unloading video.
2. Keep Detection stride at `3` for the default speed/temporal balance.
3. Click **Run full AI analysis**.
4. Review timestamped incidents, persistent track IDs, risk scores, explanations, interventions and evidence frames.
5. Ask the evidence-grounded supervisor assistant questions about the detected events.
6. Download `godrej_analysis.json` as the audit artifact.

## Core pipeline

`Video -> YOLOWorld perception -> persistent tracking -> temporal motion/spatial reasoning -> behaviour -> risk -> evidence -> supervisor intervention`

No filename-based behaviour rules or challenge-video timestamp rules are used.

## Supported behaviour taxonomy

The engine supports more than the required 10 scenario classes, including dragging, rolling, throwing/dropping, improper stacking, stepping on cartons, heavy-on-light configuration, orientation change, outside designated area, unattended product, unsafe loading/unloading sequence and rough handling.

## Important technical honesty

This is a strong prototype architecture, not a claim of production-grade accuracy across every warehouse and camera. Generic open-vocabulary models cannot reliably infer physical product weight or every semantic action from arbitrary RGB footage. The engine therefore reports visual evidence and potential risk instead of claiming facts that the video cannot establish.

For a production deployment, the next step is warehouse-specific labelled temporal data and validation against precision, recall, false-positive rate and latency.
