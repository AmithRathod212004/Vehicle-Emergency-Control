# Vehicle-Emergency-Control

Vehicle Emergency Control is an AI-powered traffic management system that detects ambulances using a YOLOv8 model and applies emergency-priority signal logic at a 4-way intersection (North, South, East, West).

## Features

- YOLOv8 ambulance detection model integration
- Streamlit dashboard for local deployment
- Image, video, and webcam input support
- 4-way traffic signal visualization
- Ambulance detection confidence display
- Plotly analytics charts
- Detection summary table
- Emergency-priority signal control logic

## Project Structure

```text
Vehicle-Emergency-Control/
├── app.py
├── traffic_logic.py
├── tests/
│   └── test_traffic_logic.py
├── requirements.txt
└── README.md
```

## Setup

1. **Clone and enter project**
   ```bash
   git clone https://github.com/AmithRathod212004/Vehicle-Emergency-Control.git
   cd Vehicle-Emergency-Control
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Provide YOLOv8 ambulance model**
   - Place your trained model at `models/ambulance_yolov8.pt`, or
   - Use the dashboard sidebar to point to any local `.pt` model file.

## Run the Dashboard

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in terminal and:
- Upload an image/video, or
- Use webcam capture,
- Review detections, confidence, traffic signal state, and analytics.

## Dataset and Training Note

This repository focuses on local inference and emergency-signal simulation. You can train YOLOv8 ambulance models using Roboflow datasets and then plug the trained `.pt` model into this app.

## Tests

Run focused logic tests:

```bash
pytest -q
```
