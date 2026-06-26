"""Smart Parking - Streamlit interface (auto mode, no manual annotation).

Run with:  uv run streamlit run app.py
"""
from __future__ import annotations

import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from parking.detector import VehicleDetector
from parking.draw import draw_boxes, draw_banner

MODEL_OPTIONS = ["yolo11n.pt", "yolov8n.pt", "yolo11s.pt", "yolov8s.pt"]

st.set_page_config(page_title="Smart Parking", layout="wide")
st.title("🅿️ Smart Parking — Auto Occupied/Vacant")
st.caption("Upload a lot photo → set capacity once → instant count. No manual slot drawing.")


@st.cache_resource(show_spinner="Loading YOLO model...")
def get_detector(model_path: str, conf: float) -> VehicleDetector:
    return VehicleDetector(model_path=model_path, conf=conf)


def crop_box(w: int, h: int, left: int, right: int, top: int, bottom: int):
    x1, x2 = int(w * left / 100), int(w * right / 100)
    y1, y2 = int(h * top / 100), int(h * bottom / 100)
    return x1, y1, x2, y2


def run_on_frame(frame: np.ndarray, detector: VehicleDetector, roi, total: int):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    boxes = detector.detect(crop)
    boxes_full = [(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, c, name) for (bx1, by1, bx2, by2, c, name) in boxes]

    occupied = len(boxes_full)
    out = draw_boxes(frame, boxes_full)
    full_frame_roi = (0, 0, frame.shape[1], frame.shape[0])
    out = draw_banner(out, occupied, total, roi if roi != full_frame_roi else None)
    return out, occupied, boxes_full


def estimate_capacity(roi, boxes_full) -> int | None:
    """Area-based heuristic: total_capacity ≈ roi_area / avg_detected_car_area."""
    if not boxes_full:
        return None
    x1, y1, x2, y2 = roi
    roi_area = max((x2 - x1) * (y2 - y1), 1)
    avg_area = sum((bx2 - bx1) * (by2 - by1) for (bx1, by1, bx2, by2, c, name) in boxes_full) / len(boxes_full)
    if avg_area <= 0:
        return None
    return max(len(boxes_full), round(roi_area / avg_area))


if "total_slots" not in st.session_state:
    st.session_state["total_slots"] = 10

with st.sidebar:
    st.header("Settings")
    model_path = st.selectbox("Model", MODEL_OPTIONS, index=0)
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.35, 0.05)
    total_slots = st.number_input("Total parking capacity", min_value=1, step=1, key="total_slots")

    with st.expander("Crop region (exclude road / background)"):
        left = st.slider("Left %", 0, 99, 0)
        right = st.slider("Right %", 1, 100, 100)
        top = st.slider("Top %", 0, 99, 0)
        bottom = st.slider("Bottom %", 1, 100, 100)

    debug_mode = st.checkbox("Debug: show raw detections (no filter)")
    st.caption("First run downloads YOLO weights (~6 MB for the 'n' models).")

detector = get_detector(model_path, conf)
tab_img, tab_vid = st.tabs(["🖼️ Image", "🎬 Video"])

with tab_img:
    img_file = st.file_uploader("Upload parking lot photo", type=["jpg", "jpeg", "png"], key="img")
    if img_file:
        img = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        roi = crop_box(w, h, left, right, top, bottom)

        result, occupied, boxes_full = run_on_frame(frame, detector, roi, total_slots)
        vacant = max(total_slots - occupied, 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total_slots)
        c2.metric("Occupied", occupied)
        c3.metric("Vacant", vacant)

        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

        est = estimate_capacity(roi, boxes_full)
        if est and est != total_slots:
            cest1, cest2 = st.columns([3, 1])
            cest1.info(f"📐 From detected car size vs. ROI area, this lot likely fits ~**{est}** cars.")
            if cest2.button("Use this as Total"):
                st.session_state["total_slots"] = est
                st.rerun()

        if occupied == 0:
            st.warning(
                "0 detected. If you can clearly see cars in the photo, this is almost always a "
                "**domain-gap** issue: stock COCO-trained YOLO rarely recognizes cars from a "
                "straight-down (nadir) view — it was trained on street-level photos. "
                "Enable 'Debug: show raw detections' in the sidebar to confirm."
            )

        if debug_mode:
            st.divider()
            st.caption("Raw detections — no class filter, conf ≥ 0.05 (debug):")
            raw = detector.detect_debug(frame[roi[1]:roi[3], roi[0]:roi[2]])
            if raw:
                st.dataframe(
                    {"class": [r[0] for r in raw[:20]], "confidence": [round(r[1], 3) for r in raw[:20]]},
                    use_container_width=True,
                )
            else:
                st.error(
                    "Nothing detected at all, even unfiltered. This points to the model weights "
                    "themselves (corrupted/incomplete download), not domain gap. Delete the cached "
                    "yolo*.pt file and let it re-download, then check its file size is a few MB, not 0 KB."
                )

with tab_vid:
    vid_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"], key="vid")
    if vid_file:
        tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_in.write(vid_file.read())
        tmp_in.flush()

        cap = cv2.VideoCapture(tmp_in.name)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        roi = crop_box(w, h, left, right, top, bottom)

        frame_idx = st.slider("Preview frame", 0, max(n_frames - 1, 0), 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            result, occupied, _ = run_on_frame(frame, detector, roi, total_slots)
            vacant = max(total_slots - occupied, 0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", total_slots)
            c2.metric("Occupied", occupied)
            c3.metric("Vacant", vacant)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("▶️ Process full video"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            writer = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            progress = st.progress(0)
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result, _, _ = run_on_frame(frame, detector, roi, total_slots)
                writer.write(result)
                i += 1
                if n_frames:
                    progress.progress(min(i / n_frames, 1.0))
            writer.release()

            st.success("Done.")
            st.video(tmp_out.name)
            with open(tmp_out.name, "rb") as f:
                st.download_button("⬇️ Download annotated video", f, file_name="annotated.mp4")

        cap.release()
