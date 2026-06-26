"""Smart Parking - Streamlit interface.

Run with:  uv run streamlit run app.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from parking.detector import VehicleDetector
from parking.slots import Slot, load_slots, save_slots, compute_occupancy
from parking.draw import draw_overlay

SLOTS_PATH = Path("data/slots.json")
MODEL_OPTIONS = ["yolo11n.pt", "yolov8n.pt", "yolo11s.pt", "yolov8s.pt"]

st.set_page_config(page_title="Smart Parking", layout="wide")
st.title("🅿️ Smart Parking — YOLO Occupancy Detector")


@st.cache_resource(show_spinner="Loading YOLO model...")
def get_detector(model_path: str, conf: float) -> VehicleDetector:
    return VehicleDetector(model_path=model_path, conf=conf)


with st.sidebar:
    st.header("Settings")
    model_path = st.selectbox("Model", MODEL_OPTIONS, index=0)
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.35, 0.05)
    overlap_thresh = st.slider("Slot occupied overlap %", 0.05, 0.9, 0.15, 0.05)
    show_boxes = st.checkbox("Show raw detection boxes", value=True)
    st.caption("First run downloads the YOLO weights (~6 MB for the 'n' models).")

tab_define, tab_detect = st.tabs(["1️⃣ Define Slots", "2️⃣ Detect Occupancy"])

# ---------------------------------------------------------------- Define ---
with tab_define:
    st.write(
        "Upload a clear reference photo of the empty/partial lot, draw a polygon "
        "per parking slot (click each corner, double-click to close), then save."
    )
    ref_file = st.file_uploader("Reference image", type=["jpg", "jpeg", "png"], key="ref")

    if ref_file:
        ref_img = Image.open(ref_file).convert("RGB")
        max_w = 900
        scale = min(1.0, max_w / ref_img.width)
        disp_w, disp_h = int(ref_img.width * scale), int(ref_img.height * scale)

        canvas_result = st_canvas(
            background_image=ref_img.resize((disp_w, disp_h)),
            height=disp_h,
            width=disp_w,
            drawing_mode="polygon",
            stroke_width=2,
            stroke_color="#00FF00",
            fill_color="rgba(0,255,0,0.25)",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            drawn = []
            for sid, obj in enumerate(canvas_result.json_data.get("objects", [])):
                if obj.get("type") != "path":
                    continue
                left, top = obj.get("left", 0), obj.get("top", 0)
                pts = []
                for seg in obj.get("path", []):
                    if len(seg) >= 3:
                        x, y = seg[-2] + left, seg[-1] + top
                        pts.append([round(x / scale, 1), round(y / scale, 1)])
                if len(pts) >= 3:
                    drawn.append(Slot(id=sid, points=pts, label=f"S{sid}"))

            st.write(f"Polygons drawn: **{len(drawn)}**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save slots", disabled=not drawn):
                    SLOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    save_slots(SLOTS_PATH, drawn)
                    st.success(f"Saved {len(drawn)} slots to {SLOTS_PATH}")
            with col2:
                st.download_button(
                    "⬇️ Download slots.json",
                    data=json.dumps([s.__dict__ for s in drawn], indent=2),
                    file_name="slots.json",
                    mime="application/json",
                    disabled=not drawn,
                )

    st.divider()
    st.caption("Or edit the raw JSON directly (coordinates are in original image pixels):")
    existing = load_slots(SLOTS_PATH)
    raw = st.text_area(
        "slots.json",
        value=json.dumps([s.__dict__ for s in existing], indent=2) if existing else "[]",
        height=180,
    )
    if st.button("💾 Save edited JSON"):
        try:
            data = json.loads(raw)
            SLOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            save_slots(SLOTS_PATH, [Slot(**s) for s in data])
            st.success("Saved.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

# ---------------------------------------------------------------- Detect ---
with tab_detect:
    slots = load_slots(SLOTS_PATH)
    if not slots:
        st.warning("No slots defined yet — go to the 'Define Slots' tab first.")
    else:
        st.caption(f"Loaded {len(slots)} slots from {SLOTS_PATH}")

    src_type = st.radio("Source", ["Image", "Video"], horizontal=True)

    if src_type == "Image":
        img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="det_img")
        if img_file and slots:
            img = Image.open(img_file).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            detector = get_detector(model_path, conf)
            boxes = detector.detect(frame)
            occupancy = compute_occupancy(slots, boxes, overlap_thresh)
            result = draw_overlay(frame, slots, occupancy, boxes, show_boxes)

            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
            free = sum(1 for v in occupancy.values() if not v)
            st.metric("Free slots", f"{free} / {len(slots)}")

    else:  # Video
        vid_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"], key="det_vid")
        if vid_file and slots:
            tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_in.write(vid_file.read())
            tmp_in.flush()

            cap = cv2.VideoCapture(tmp_in.name)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = st.slider("Preview frame", 0, max(n_frames - 1, 0), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()

            detector = get_detector(model_path, conf)
            if ok:
                boxes = detector.detect(frame)
                occupancy = compute_occupancy(slots, boxes, overlap_thresh)
                result = draw_overlay(frame, slots, occupancy, boxes, show_boxes)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
                free = sum(1 for v in occupancy.values() if not v)
                st.metric("Free slots", f"{free} / {len(slots)}")

            if st.button("▶️ Process full video"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                writer = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                progress = st.progress(0)
                i = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    boxes = detector.detect(frame)
                    occupancy = compute_occupancy(slots, boxes, overlap_thresh)
                    writer.write(draw_overlay(frame, slots, occupancy, boxes, show_boxes))
                    i += 1
                    if n_frames:
                        progress.progress(min(i / n_frames, 1.0))
                writer.release()

                st.success("Done.")
                st.video(tmp_out.name)
                with open(tmp_out.name, "rb") as f:
                    st.download_button("⬇️ Download annotated video", f, file_name="annotated.mp4")

            cap.release()
