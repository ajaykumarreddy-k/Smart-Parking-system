# ──────────────────────────────────────────────
# IPSD — Intelligent Parking Space Detection
# Streamlit GUI — Image / Video / Webcam
#
# Setup:
#   uv add streamlit ultralytics opencv-python-headless pillow
#
# Run:
#   uv run streamlit run main_gui.py
# ──────────────────────────────────────────────
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "models/parking_yolo.pt"
CONF_THRESH = 0.4

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="IPSD · Parking Detection",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME — dark control-room console ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:        #0B0F14;
    --panel:     #11161D;
    --border:    #1E2630;
    --text:      #E6EAEE;
    --muted:     #7C8896;
    --vacant:    #2ECC71;
    --occupied:  #E74C3C;
    --accent:    #4FD1C5;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp { background-color: var(--bg); color: var(--text); }

[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px solid var(--border);
}

h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }

.ipsd-hero {
    display: flex; align-items: baseline; gap: 14px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 18px; margin-bottom: 6px;
}
.ipsd-hero .mark {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; color: var(--accent);
    background: rgba(79, 209, 197, 0.08);
    border: 1px solid rgba(79, 209, 197, 0.3);
    padding: 4px 10px; border-radius: 4px;
}
.ipsd-hero h1 { font-size: 28px; font-weight: 700; margin: 0; }
.ipsd-sub { color: var(--muted); font-size: 14px; margin-top: -8px; margin-bottom: 24px; }

.metric-row { display: flex; gap: 16px; margin: 18px 0 28px 0; }
.metric-card {
    flex: 1; background: var(--panel);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px;
}
.metric-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 34px; font-weight: 700; line-height: 1;
}
.metric-card.vacant .value   { color: var(--vacant); }
.metric-card.occupied .value { color: var(--occupied); }
.metric-card.total .value    { color: var(--text); }

.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    padding: 5px 12px; border-radius: 20px;
    background: rgba(46, 204, 113, 0.1); color: var(--vacant);
    border: 1px solid rgba(46, 204, 113, 0.3);
}
.status-pill.idle {
    background: rgba(124, 136, 150, 0.1); color: var(--muted);
    border: 1px solid rgba(124, 136, 150, 0.3);
}
.status-pill.err {
    background: rgba(231, 76, 60, 0.1); color: var(--occupied);
    border: 1px solid rgba(231, 76, 60, 0.3);
}

div.stButton > button {
    background: var(--panel); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px;
    font-weight: 500; padding: 10px 16px;
}
div.stButton > button:hover {
    border-color: var(--accent); color: var(--accent);
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADING (cached so it only loads once) ──
@st.cache_resource
def load_model(path: str):
    if not Path(path).exists():
        return None
    return YOLO(path)


def annotate(frame: np.ndarray, model: YOLO):
    """Run YOLO inference, draw boxes, return annotated frame + counts.
    Handles 3-class models (spaces / space-empty / space-occupied) by
    ignoring the generic 'spaces' class if present.
    """
    results = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]
    occupied = empty = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name == "spaces":
            # generic/ambiguous class — skip, don't count as empty or occupied
            color = (255, 200, 0)  # amber, just for visual debug
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            continue

        is_occupied = "occupied" in cls_name
        color = (231, 76, 60) if is_occupied else (46, 204, 113)  # RGB
        if is_occupied:
            occupied += 1
        else:
            empty += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    total = occupied + empty
    return frame, empty, occupied, total


def render_metrics(empty: int, occupied: int, total: int, placeholder):
    occ_pct = f"{(occupied/total*100):.0f}%" if total else "—"
    placeholder.markdown(f"""
    <div class="metric-row">
        <div class="metric-card vacant">
            <div class="label">Vacant</div>
            <div class="value">{empty}</div>
        </div>
        <div class="metric-card occupied">
            <div class="label">Occupied</div>
            <div class="value">{occupied}</div>
        </div>
        <div class="metric-card total">
            <div class="label">Total Spaces</div>
            <div class="value">{total}</div>
        </div>
        <div class="metric-card total">
            <div class="label">Occupancy</div>
            <div class="value">{occ_pct}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── HERO ──
st.markdown("""
<div class="ipsd-hero">
    <span class="mark">IPSD</span>
    <h1>Intelligent Parking Space Detection</h1>
</div>
<div class="ipsd-sub">YOLOv8 · trained on PKLot · real-time occupancy from image, video, or live camera</div>
""", unsafe_allow_html=True)

# ── MODEL STATUS ──
model = load_model(MODEL_PATH)

status_box = st.empty()
if model is None:
    status_box.markdown(
        f'<span class="status-pill err">● model not found at {MODEL_PATH}</span>',
        unsafe_allow_html=True)
    st.warning(
        f"Place your downloaded **parking_yolo.pt** at `{MODEL_PATH}` "
        "and refresh this page.")
    st.stop()
else:
    status_box.markdown(
        '<span class="status-pill">● model loaded</span>', unsafe_allow_html=True)

# ── SIDEBAR — input controls ──
with st.sidebar:
    st.markdown("### Input Source")
    mode = st.radio(
        "Choose source", ["Image", "Video", "Webcam"],
        label_visibility="collapsed")

    st.markdown("### Detection Settings")
    conf = st.slider("Confidence threshold", 0.1, 0.9, CONF_THRESH, 0.05)
    CONF_THRESH = conf

    st.markdown("---")
    st.markdown(
        f'<span style="font-family:JetBrains Mono;font-size:12px;color:#7C8896;">'
        f'classes: {", ".join(model.names.values())}</span>',
        unsafe_allow_html=True)

metrics_placeholder = st.empty()
render_metrics(0, 0, 0, metrics_placeholder)
frame_placeholder = st.empty()

# ── IMAGE MODE ──
if mode == "Image":
    uploaded = st.file_uploader(
        "Upload a parking lot image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        annotated, empty, occupied, total = annotate(frame, model)
        render_metrics(empty, occupied, total, metrics_placeholder)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_rgb, use_container_width=True)
    else:
        frame_placeholder.info("Upload an image to detect parking space occupancy.")

# ── VIDEO MODE ──
elif mode == "Video":
    uploaded = st.file_uploader(
        "Upload a parking lot video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None:
        # Unique filename per upload avoids stale/partial file collisions
        tmp_path = Path(f"temp_{uploaded.file_id}.mp4").resolve() \
            if hasattr(uploaded, "file_id") else Path("temp_uploaded_video.mp4").resolve()

        if tmp_path.exists():
            tmp_path.unlink()  # always start from a clean file

        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
            f.flush()
            import os
            os.fsync(f.fileno())  # force OS to actually write to disk before reading

        st.caption(f"Saved: {tmp_path}  ({tmp_path.stat().st_size} bytes)")

        run = st.button("▶ Start Detection")
        stop_flag = st.checkbox("Stop after current run")

        if run:
            cap = cv2.VideoCapture(str(tmp_path), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                st.error(
                    "Could not open the uploaded video with FFMPEG backend. "
                    "Trying default backend..."
                )
                cap = cv2.VideoCapture(str(tmp_path))

            if not cap.isOpened():
                st.error(
                    f"OpenCV cannot decode this file (path: {tmp_path}, "
                    f"size: {tmp_path.stat().st_size} bytes). "
                    "This usually means missing codecs in opencv-python-headless, "
                    "or the file extension doesn't match its real codec. "
                    "Try re-exporting the video as H.264 .mp4, or run: "
                    "`uv add opencv-python` (non-headless build includes more codecs)."
                )
            else:
                frame_count = 0
                detection_count = 0
                stop_btn_placeholder = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                    annotated, empty, occupied, total = annotate(frame, model)
                    detection_count += total
                    render_metrics(empty, occupied, total, metrics_placeholder)

                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_rgb, use_container_width=True)

                    if stop_flag:
                        break
                    time.sleep(0.02)
                cap.release()
                st.caption(
                    f"Processed {frame_count} frames · "
                    f"{detection_count} total detections across all frames"
                )
    else:
        frame_placeholder.info("Upload a video to begin frame-by-frame detection.")

# ── WEBCAM MODE ──
elif mode == "Webcam":
    st.caption("Uses your default system camera (device 0).")
    run = st.checkbox("Start Webcam", key="webcam_running")

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam.")
        else:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("No frame received from webcam.")
                    break

                annotated, empty, occupied, total = annotate(frame, model)
                render_metrics(empty, occupied, total, metrics_placeholder)

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, use_container_width=True)

                time.sleep(0.03)
            cap.release()
    else:
        frame_placeholder.info("Enable the checkbox to start the live feed.")
