import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import tempfile
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import defaultdict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

st.set_page_config(page_title="Ambulance Detector", layout="wide")

# Custom CSS for creative UI
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Title styling */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #f0f0f0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    /* Card-like containers */
    div[data-testid="stBlock"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #4299e1;
    }
    
    /* Radio buttons */
    div[role="radiogroup"] label {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    div[role="radiogroup"] label:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        color: #2d3748;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
    }
    
    /* Headers */
    h2, h3 {
        color: #2d3748 !important;
        font-weight: 700 !important;
    }
    
    /* File uploader */
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #a0aec0;
        border-radius: 12px;
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚑 Smart Ambulance Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Traffic Signal Control • Real-Time Multi-Lane Monitoring</p>", unsafe_allow_html=True)

# Sidebar: model selection
st.sidebar.header("Model")
model_path = st.sidebar.text_input(
    "Path to YOLO model (best.pt)",
    value=str(Path(__file__).parent / "best.pt"),
    help="Provide your trained weights. After training, copy best.pt here."
)
# Set confidence threshold to 40% for better ambulance detection
conf_threshold = 0.4
# Only show ambulance boxes (no other vehicles)
show_all = False
st.sidebar.markdown("---")

# Force model to use the trained weights
if not Path(model_path).exists():
    st.sidebar.error(f"❌ Model file not found: {model_path}")
    st.sidebar.warning("Model detection disabled until valid weights are provided")
    model = None
else:
    st.sidebar.info(f"Using trained model: {Path(model_path).name} ({Path(model_path).stat().st_size / 1e6:.1f}MB)")

# Load model
model = None
model_type = "pretrained"
if YOLO_AVAILABLE and Path(model_path).exists():
    try:
        model = YOLO(model_path)
        model_type = "custom" if Path(model_path).name == "best.pt" else "pretrained"
        st.sidebar.success(f"✅ Model loaded: {Path(model_path).name}")
        st.sidebar.info(f"Classes in model: {list(model.names.values())}")
    except (AttributeError, RuntimeError) as load_err:
        st.sidebar.warning(f"Model architecture issue: {str(load_err)[:100]}...")
        try:
            # Try loading with device='cpu'
            model = YOLO(model_path, task='detect')
            model.to('cpu')
            st.sidebar.success(f"✅ Model loaded (CPU mode): {Path(model_path).name}")
            st.sidebar.info(f"Classes: {list(model.names.values())}")
        except Exception as retry_err:
            st.sidebar.error(f"❌ Model load failed: {str(retry_err)[:150]}")
            model = None
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {str(e)[:150]}")
        model = None
elif not YOLO_AVAILABLE:
    st.sidebar.error("Ultralytics not installed. Install from requirements.txt.")


# Utility: draw boxes from YOLO results
# Ambulance detection: focus on ambulance classes
AMBULANCE_CLASSES = {"ambulance", "ambulance_off", "ambulance_on", "Ambulance"}

def annotate(frame_bgr, results):
    """Draw bounding boxes ONLY for ambulances. Strict class ID + confidence check."""
    annotated = frame_bgr.copy()
    if results and results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = results.names.get(cls, "").lower().strip()
            
            # STRICT FILTER: 
            # - Class ID must be 0 (ambulance class)
            # - Class name must be "ambulance"
            # - Confidence must be > 40%
            if cls == 0 and class_name == "ambulance" and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 255)  # Cyan box for ambulance
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                # Label with confidence percentage
                label = f"AMBULANCE {conf*100:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            # ALL OTHER DETECTIONS: Completely ignored, no box drawn
    
    return annotated


def detect_frame(frame_bgr):
    """Run YOLO on a single frame and return annotated frame and ambulance flag.
    Only detects AMBULANCE class with >50% confidence.
    """
    if not model:
        return frame_bgr, False, 0.0
    try:
        res = model(frame_bgr, conf=conf_threshold, verbose=False)
        if not res:
            return frame_bgr, False, 0.0
    except Exception as e:
        st.error(f"Detection error: {str(e)[:100]}")
        return frame_bgr, False, 0.0
    
    r0 = res[0]
    annotated = annotate(frame_bgr, r0)

    ambulance_present = False
    top_conf = 0.0
    
    # Check all detections
    if r0.boxes is not None:
        for box in r0.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = r0.names.get(cls, "").lower().strip()
            
            # STRICT: Only accept class 0 (AMBULANCE) with >40% confidence
            # Ignore all other classes
            if cls == 0 and class_name == "ambulance" and conf > 0.4:
                if conf > top_conf:
                    top_conf = conf
                    ambulance_present = True
    
    return annotated, ambulance_present, top_conf


def signal_state(detections):
    """Return dict of signal colors based on detections per direction."""
    state = {d: "red" for d in ["north", "south", "east", "west"]}
    # If multiple detected, we prioritize the highest confidence
    best_dir = None
    best_conf = 0.0
    for direction, info in detections.items():
        if info["ambulance"] and info["confidence"] >= best_conf:
            best_conf = info["confidence"]
            best_dir = direction
    if best_dir:
        state[best_dir] = "green"
    return state


def render_signal(state):
    """Render a realistic traffic signal grid for N/S/E/W with visual lights."""
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #2d3748; margin-bottom: 2rem;'>🚦 Live Traffic Signal Dashboard</h2>", unsafe_allow_html=True)
    
    # Create animated traffic signal layout
    st.markdown("""
    <style>
        .signal-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        .signal-row {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin: 10px 0;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        .active-green {
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create 2x2 grid layout for intersection
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # North signal (top)
        render_traffic_light("NORTH", state['north'])
    
    st.write("")
    
    row_cols = st.columns([1, 1, 1])
    with row_cols[0]:
        # West signal (left)
        render_traffic_light("WEST", state['west'])
    
    with row_cols[1]:
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
            <h1 style='color: white; font-size: 3rem; margin: 0;'>🚦</h1>
            <h2 style='color: white; margin: 10px 0 0 0;'>INTERSECTION</h2>
            <p style='color: rgba(255,255,255,0.8); margin-top: 5px; font-size: 0.9rem;'>4-Way Control</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row_cols[2]:
        # East signal (right)
        render_traffic_light("EAST", state['east'])
    
    st.write("")
    
    with col2:
        # South signal (bottom)
        render_traffic_light("SOUTH", state['south'])
    
    st.markdown("---")


def render_traffic_light(direction, current_state):
    """Render a single traffic light pole with red, yellow, green lights."""
    red_on = "🔴" if current_state == "red" else "⚫"
    yellow_on = "🟡" if current_state == "yellow" else "⚫"
    green_on = "🟢" if current_state == "green" else "⚫"
    
    glow_class = "active-green" if current_state == "green" else ""
    bg_gradient = "linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%)" if current_state == "red" else "linear-gradient(135deg, #065f46 0%, #047857 100%)"
    
    st.markdown(f"""
    <div style='text-align: center; padding: 15px; background: {bg_gradient}; 
                border-radius: 20px; border: 4px solid {"#dc2626" if current_state == "red" else "#10b981"}; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.3); transition: all 0.3s ease;'>
        <div style='font-weight: bold; color: white; margin-bottom: 12px; font-size: 1.1rem; 
                    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);'>{direction}</div>
        <div class='{glow_class}' style='font-size: 45px; line-height: 1.3;'>
            {red_on}<br/>
            {yellow_on}<br/>
            {green_on}
        </div>
        <div style='font-size: 13px; color: {"#10b981" if current_state == "green" else "#ef4444"}; 
                    margin-top: 10px; font-weight: 800; text-transform: uppercase;
                    background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 10px;'>
            {"🚑 CLEAR PATH" if current_state == "green" else "⛔ STOP"}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_single_feed_signal(ambulance_present: bool):
    """Render a simple N/S/E/W signal where green appears when an ambulance is present."""
    if ambulance_present:
        st.success("🚑 AMBULANCE DETECTED - All lanes GREEN for emergency clearance")
        state = {d: "green" for d in ["north", "south", "east", "west"]}
    else:
        st.info("🚦 Normal traffic flow - All lanes RED (no ambulance)")
        state = {d: "red" for d in ["north", "south", "east", "west"]}
    render_signal(state)


def run_normal_traffic_cycle(directions, cycle_rounds: int = 4, cycle_duration: int = 10):
    """Rotate signals in normal mode when no ambulance is present."""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0; 
                border-left: 6px solid #f59e0b; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);'>
        <h3 style='color: #78350f; margin: 0 0 0.5rem 0;'>🔄 NORMAL TRAFFIC CYCLE MODE</h3>
        <p style='color: #92400e; margin: 0; font-size: 1.1rem; font-weight: 600;'>
            Automatic signal rotation every 10 seconds: North → East → South → West
        </p>
    </div>
    """, unsafe_allow_html=True)

    signal_placeholder = st.empty()
    status_placeholder = st.empty()
    cycle_order = ["north", "east", "south", "west"]

    for cycle_round in range(cycle_rounds):
        for idx, active_direction in enumerate(cycle_order):
            state = {d: "red" for d in directions}
            state[active_direction] = "green"

            status_placeholder.markdown(f"""
            <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                        padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;
                        border: 3px solid #10b981;'>
                <h3 style='color: #065f46; margin: 0;'>
                    🟢 {active_direction.upper()} LANE - GREEN SIGNAL
                </h3>
                <p style='color: #047857; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                    Cycle {cycle_round + 1}/{cycle_rounds} • Signal {idx + 1}/{len(cycle_order)}
                </p>
            </div>
            """, unsafe_allow_html=True)

            with signal_placeholder.container():
                render_signal(state)

            time.sleep(cycle_duration)

    st.success(f"✅ Normal traffic cycle completed ({cycle_rounds} full rotation{'s' if cycle_rounds > 1 else ''})")

# Single tab for 4-lane intersection
tab_intersection = st.tabs(["🛣️ 4-Lane Intersection"])[0]

with tab_intersection:
    st.markdown("<h2 style='text-align: center; color: #2d3748; margin-bottom: 1rem;'>🛣️ Multi-Lane Intersection Control</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); 
                padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem; border-left: 5px solid #3b82f6;'>
        <p style='margin: 0; color: #1e40af; font-weight: 500;'>
            🎯 Configure each lane with Image, Video, or Live Camera input. 
            The system will detect ambulances and automatically control traffic signals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    directions = ["north", "south", "east", "west"]
    
    # Input type selection for each direction
    st.markdown("<h3 style='color: #374151; margin-top: 1.5rem;'>📍 Lane Configuration</h3>", unsafe_allow_html=True)
    input_types = {}
    for i, dir_name in enumerate(directions):
        with cols[i]:
            st.markdown(f"<div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 10px; border-radius: 10px; text-align: center; font-weight: 700; color: #78350f; margin-bottom: 10px;'>{dir_name.upper()} LANE</div>", unsafe_allow_html=True)
            input_types[dir_name] = st.radio(
                f"{dir_name} input",
                ["None", "Image", "Video", "Webcam"],
                key=f"input_type_{dir_name}",
                label_visibility="collapsed"
            )
    
    st.markdown("---")
    st.markdown("<h3 style='color: #374151;'>📤 Upload Media / Camera Setup</h3>", unsafe_allow_html=True)
    
    uploads = {}
    cam_indices = {}
    
    cols2 = st.columns(4)
    for i, dir_name in enumerate(directions):
        with cols2[i]:
            if input_types[dir_name] == "Image":
                uploads[dir_name] = st.file_uploader(
                    f"{dir_name.capitalize()} image",
                    type=["jpg", "jpeg", "png", "bmp", "webp"],
                    key=f"upload_img_{dir_name}"
                )
            elif input_types[dir_name] == "Video":
                uploads[dir_name] = st.file_uploader(
                    f"{dir_name.capitalize()} video",
                    type=["mp4", "mov", "avi", "mkv", "webm"],
                    key=f"upload_vid_{dir_name}"
                )
            elif input_types[dir_name] == "Webcam":
                cam_indices[dir_name] = st.number_input(
                    f"{dir_name.capitalize()} cam index",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key=f"cam_{dir_name}",
                    help=f"Camera index for {dir_name} lane (usually 0 for default webcam)"
                )

    run_intersection = st.button("🚦 START DETECTION & SIGNAL CONTROL", type="primary")

    if run_intersection:
        if not model:
            st.error("Load a model in the sidebar first.")
        else:
            # Check if all lanes are set to "None" - activate normal traffic cycle
            all_none = all(input_types[d] == "None" for d in directions)
            
            if all_none:
                run_normal_traffic_cycle(directions, cycle_rounds=4, cycle_duration=10)
            else:
                # Original detection mode
                placeholders = {d: st.empty() for d in directions}
                detections = {d: {"ambulance": False, "confidence": 0.0} for d in directions}

                image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

                for direction in directions:
                    input_type = input_types[direction]
                    
                    if input_type == "None":
                        placeholders[direction].info("No input configured")
                        continue
                    
                    elif input_type == "Image":
                        clip = uploads.get(direction)
                        if not clip:
                            placeholders[direction].warning("No image uploaded")
                            continue
                        
                        data = clip.read()
                        arr = np.frombuffer(data, np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is None:
                            placeholders[direction].error("Unable to read image")
                            continue
                        annotated, has_amb, conf_val = detect_frame(frame)
                        detections[direction] = {"ambulance": has_amb, "confidence": conf_val}
                        placeholders[direction].image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", caption=direction.upper())
                    
                    elif input_type == "Video":
                        clip = uploads.get(direction)
                        if not clip:
                            placeholders[direction].warning("No video uploaded")
                            continue
                        
                        suffix = Path(clip.name).suffix.lower()
                        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        temp.write(clip.read())
                        temp.flush()

                        cap = cv2.VideoCapture(temp.name)
                        if not cap.isOpened():
                            placeholders[direction].error("Unable to open video")
                            continue

                        best_frame = None
                        best_conf = 0.0
                        frame_count = 0
                        sample_rate = 5

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_count += 1
                            if frame_count % sample_rate != 0:
                                continue

                            annotated, has_amb, conf_val = detect_frame(frame)
                            if has_amb and conf_val > best_conf:
                                best_conf = conf_val
                                best_frame = annotated.copy()

                        cap.release()
                        detections[direction] = {"ambulance": best_conf > 0, "confidence": best_conf}

                        if best_frame is not None:
                            placeholders[direction].image(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=direction.upper())
                        else:
                            placeholders[direction].info("No ambulance detected in video")
                    
                    elif input_type == "Webcam":
                        cam_idx = cam_indices.get(direction, 0)
                        cap = cv2.VideoCapture(int(cam_idx))
                        if not cap.isOpened():
                            placeholders[direction].error(f"Cannot access camera {cam_idx}")
                            continue
                        
                        # Capture and process 10 frames
                        best_frame = None
                        best_conf = 0.0
                        for _ in range(10):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            annotated, has_amb, conf_val = detect_frame(frame)
                            if has_amb and conf_val > best_conf:
                                best_conf = conf_val
                                best_frame = annotated.copy()
                        
                        cap.release()
                        detections[direction] = {"ambulance": best_conf > 0, "confidence": best_conf}
                        
                        if best_frame is not None:
                            placeholders[direction].image(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=direction.upper())
                        else:
                            placeholders[direction].info("No ambulance detected from webcam")

                state = signal_state(detections)
                
                # Show detection summary
                detected_lanes = [d for d, info in detections.items() if info["ambulance"]]
                if detected_lanes:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                                padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0; 
                                border-left: 6px solid #10b981; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);'>
                        <h3 style='color: #065f46; margin: 0 0 0.5rem 0;'>🚑 AMBULANCE DETECTED</h3>
                        <p style='color: #047857; margin: 0; font-size: 1.1rem; font-weight: 600;'>
                            Active Lanes: {', '.join([d.upper() for d in detected_lanes])}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    render_signal(state)
                else:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #e0f2fe 0%, #bfdbfe 100%); 
                                padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0; 
                                border-left: 6px solid #3b82f6;'>
                        <h3 style='color: #1e40af; margin: 0 0 0.5rem 0;'>🚦 Normal Traffic Flow</h3>
                        <p style='color: #1e3a8a; margin: 0; font-size: 1.1rem;'>
                            No ambulance detected in any lane
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    # When no ambulance is found on any configured feed, fall back to normal cycle
                    run_normal_traffic_cycle(directions, cycle_rounds=1, cycle_duration=10)
                
                # Analytics Section
                st.markdown("<hr style='margin: 2rem 0; border: 2px solid #e5e7eb;'>", unsafe_allow_html=True)
                st.markdown("<h2 style='text-align: center; color: #2d3748; margin-bottom: 1.5rem;'>📊 System Analytics</h2>", unsafe_allow_html=True)
                
                # Prepare analytics data
                analytics_data = []
                for direction in ["north", "south", "east", "west"]:
                    info = detections[direction]
                    analytics_data.append({
                        "Lane": direction.upper(),
                        "Ambulance Detected": "Yes" if info["ambulance"] else "No",
                        "Confidence": info["confidence"],
                        "Signal Status": state[direction].upper()
                    })
                
                # Create two columns for graphs
                graph_col1, graph_col2 = st.columns(2)
                
                with graph_col1:
                    st.markdown("""
                    <div style='background: rgba(255,255,255,0.95); padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <h3 style='color: #2d3748; margin-bottom: 1rem;'>🎯 Confidence Levels by Lane</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar chart
                    df_analytics = pd.DataFrame(analytics_data)
                    fig_confidence = go.Figure()
                    
                    colors = []
                    for _, row in df_analytics.iterrows():
                        if row["Ambulance Detected"] == "Yes":
                            colors.append('#10b981')  # Green for ambulance detected
                        else:
                            colors.append('#94a3b8')  # Gray for no detection
                    
                    fig_confidence.add_trace(go.Bar(
                        x=df_analytics["Lane"],
                        y=df_analytics["Confidence"] * 100,
                        marker_color=colors,
                        text=[f"{c*100:.1f}%" for c in df_analytics["Confidence"]],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
                    ))
                    
                    fig_confidence.update_layout(
                        yaxis_title="Confidence (%)",
                        xaxis_title="Lane Direction",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=350,
                        showlegend=False,
                        font=dict(size=12),
                        yaxis=dict(range=[0, 100], gridcolor='#e5e7eb'),
                        xaxis=dict(gridcolor='#e5e7eb')
                    )
                    
                    fig_confidence.add_hline(y=50, line_dash="dash", line_color="#ef4444", 
                                            annotation_text="50% Threshold", 
                                            annotation_position="right")
                    
                    st.plotly_chart(fig_confidence, use_container_width=True)
                
                with graph_col2:
                    st.markdown("""
                    <div style='background: rgba(255,255,255,0.95); padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <h3 style='color: #2d3748; margin-bottom: 1rem;'>🚦 Signal Status Distribution</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Signal status pie chart
                    signal_counts = df_analytics["Signal Status"].value_counts()
                    
                    fig_signals = go.Figure(data=[go.Pie(
                        labels=signal_counts.index,
                        values=signal_counts.values,
                        marker=dict(colors=['#10b981', '#ef4444']),  # Green and Red
                        hole=0.4,
                        textinfo='label+percent',
                        hovertemplate='<b>%{label}</b><br>Lanes: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )])
                    
                    fig_signals.update_layout(
                        showlegend=True,
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig_signals, use_container_width=True)
                
                # Detection summary table
                st.markdown("""
                <div style='background: rgba(255,255,255,0.95); padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-top: 1.5rem;'>
                    <h3 style='color: #2d3748; margin-bottom: 1rem;'>📋 Detection Summary Table</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Style the dataframe
                df_display = df_analytics.copy()
                df_display["Confidence"] = df_display["Confidence"].apply(lambda x: f"{x*100:.2f}%")
                
                # Color code the table
                def highlight_ambulance(row):
                    if row["Ambulance Detected"] == "Yes":
                        return ['background-color: #d1fae5; color: #065f46'] * len(row)
                    else:
                        return ['background-color: #f1f5f9; color: #475569'] * len(row)
                
                styled_df = df_display.style.apply(highlight_ambulance, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Real-time metrics
                st.markdown("<hr style='margin: 2rem 0; border: 2px solid #e5e7eb;'>", unsafe_allow_html=True)
                
                metric_cols = st.columns(4)
                
                total_detections = sum(1 for d in detections.values() if d["ambulance"])
                avg_confidence = np.mean([d["confidence"] for d in detections.values() if d["ambulance"]]) if total_detections > 0 else 0
                green_lights = sum(1 for s in state.values() if s == "green")
                red_lights = sum(1 for s in state.values() if s == "red")
                
                with metric_cols[0]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                                padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);'>
                        <h4 style='color: #065f46; margin: 0; font-size: 0.9rem;'>AMBULANCES DETECTED</h4>
                        <p style='color: #047857; font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>{total_detections}</p>
                        <p style='color: #059669; margin: 0; font-size: 0.8rem;'>out of 4 lanes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[1]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                                padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);'>
                        <h4 style='color: #1e40af; margin: 0; font-size: 0.9rem;'>AVG CONFIDENCE</h4>
                        <p style='color: #1e3a8a; font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>{avg_confidence*100:.1f}%</p>
                        <p style='color: #2563eb; margin: 0; font-size: 0.8rem;'>detection accuracy</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[2]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #d1fae5 0%, #6ee7b7 100%); 
                                padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);'>
                        <h4 style='color: #065f46; margin: 0; font-size: 0.9rem;'>GREEN SIGNALS</h4>
                        <p style='color: #047857; font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>{green_lights}</p>
                        <p style='color: #059669; margin: 0; font-size: 0.8rem;'>clear path</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[3]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                                padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);'>
                        <h4 style='color: #991b1b; margin: 0; font-size: 0.9rem;'>RED SIGNALS</h4>
                        <p style='color: #b91c1c; font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>{red_lights}</p>
                        <p style='color: #dc2626; margin: 0; font-size: 0.8rem;'>stop</p>
                    </div>
                    """, unsafe_allow_html=True)

