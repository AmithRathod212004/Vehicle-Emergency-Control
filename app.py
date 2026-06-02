from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from traffic_logic import LANES, get_signal_state


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def infer_lane(frame_width: int, x_center: float) -> str:
    lane_width = frame_width / len(LANES)
    index = int(min(len(LANES) - 1, max(0, x_center // lane_width)))
    return LANES[index]


def detect_frame(model, frame: np.ndarray, conf_threshold: float) -> Tuple[np.ndarray, List[dict]]:
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    annotated = frame.copy()
    detections: List[dict] = []

    for result in results:
        names = getattr(result, "names", {})
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            class_name = str(names.get(cls_id, cls_id))
            x_center = (x1 + x2) / 2
            lane = infer_lane(frame.shape[1], x_center)
            is_ambulance = "ambulance" in class_name.lower()

            label = f"{class_name} {conf:.2f} ({lane})"
            color = (0, 255, 0) if is_ambulance else (0, 165, 255)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, label, (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "lane": lane,
                    "is_ambulance": is_ambulance,
                }
            )

    return annotated, detections


def render_signal_state(signal_state: dict):
    st.subheader("4-Way Traffic Signal")
    cols = st.columns(4)
    for col, lane in zip(cols, LANES):
        color = signal_state[lane]
        dot = "🟢" if color == "green" else "🔴"
        col.metric(lane, dot)


def render_analytics(detections: List[dict]):
    st.subheader("Detection Summary")
    if not detections:
        st.info("No detections available yet.")
        return

    df = pd.DataFrame(detections)
    st.dataframe(df, use_container_width=True)

    class_counts = df["class"].value_counts().reset_index()
    class_counts.columns = ["class", "count"]
    lane_counts = df["lane"].value_counts().reindex(LANES, fill_value=0).reset_index()
    lane_counts.columns = ["lane", "count"]

    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(class_counts, x="class", y="count", title="Detections by Class"), use_container_width=True)
    col2.plotly_chart(px.pie(lane_counts, values="count", names="lane", title="Detections by Lane"), use_container_width=True)


def main():
    st.set_page_config(page_title="Vehicle Emergency Control", layout="wide")
    st.title("Vehicle Emergency Control")
    st.caption("AI-powered emergency-priority traffic management demo using YOLOv8 + OpenCV + Streamlit")

    with st.sidebar:
        st.header("Model & Input")
        model_path = st.text_input("YOLOv8 model path", value="models/ambulance_yolov8.pt")
        conf_threshold = st.slider("Confidence threshold", min_value=0.1, max_value=0.95, value=0.35, step=0.05)
        source_type = st.radio("Input source", ["Image", "Video", "Webcam"], index=0)

    if not Path(model_path).exists():
        st.warning(f"Model not found at {model_path}. Provide a trained YOLOv8 model file to run detection.")
        return

    try:
        model = load_model(model_path)
    except Exception as exc:
        st.error(f"Failed to load YOLO model: {exc}")
        return

    all_detections: List[dict] = []

    if source_type == "Image":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated, detections = detect_frame(model, frame, conf_threshold)
            all_detections.extend(detections)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection output", use_container_width=True)

    elif source_type == "Video":
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

            cap = cv2.VideoCapture(video_path)
            frame_placeholder = st.empty()
            frame_index = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % 3 != 0:
                    continue
                annotated, detections = detect_frame(model, frame, conf_threshold)
                all_detections.extend(detections)
                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            cap.release()

    else:  # Webcam
        camera_image = st.camera_input("Capture webcam frame")
        if camera_image is not None:
            file_bytes = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated, detections = detect_frame(model, frame, conf_threshold)
            all_detections.extend(detections)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Webcam detection output", use_container_width=True)

    signal_state = get_signal_state(all_detections)
    render_signal_state(signal_state)

    ambulance_detections = [d for d in all_detections if d["is_ambulance"]]
    top_conf = max((d["confidence"] for d in ambulance_detections), default=0.0)
    st.metric("Highest ambulance confidence", f"{top_conf:.2f}")

    render_analytics(all_detections)


if __name__ == "__main__":
    main()
