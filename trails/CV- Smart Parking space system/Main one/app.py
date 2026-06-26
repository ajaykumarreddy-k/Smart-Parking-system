"""Streamlit dashboard: Qwen2-VL-2B parking occupancy detection.

Two modes:
  Quick Count    - upload image, instant occupied/vacant count, no setup.
  Per-Slot (Adv) - define exact slot boxes, get a green/red overlay per slot.
"""
import os
import json
import tempfile

import streamlit as st
from PIL import Image

from slot_config import load_slots, save_slots
from frame_extractor import extract_frames, get_first_frame
from vlm_classifier import ParkingSlotClassifier
from detector import run_detection, compute_metrics
from overlay import draw_overlay

st.set_page_config(page_title="Smart Parking Detection", layout="wide")
st.title("🚗 Smart Parking Space Detection — Qwen2-VL-2B")

REF_FRAME_PATH = "reference_frame.jpg"


@st.cache_resource(show_spinner="Loading Qwen2-VL-2B model (first run only)...")
def get_classifier():
    return ParkingSlotClassifier()


tab_quick, tab_advanced = st.tabs(["⚡ Quick Count", "🎯 Per-Slot (Advanced)"])

# =====================================================================
# QUICK COUNT — just upload, instant occupied/vacant count, no setup
# =====================================================================
with tab_quick:
    st.caption("Upload an image and get occupied/vacant counts immediately — "
               "no slot setup required.")
    quick_file = st.file_uploader(
        "Upload parking lot image", type=["jpg", "jpeg", "png"], key="quick_upload"
    )

    if quick_file is not None:
        quick_image = Image.open(quick_file).convert("RGB")
        st.image(quick_image, caption="Uploaded image", use_container_width=True)

        if st.button("▶️ Analyze", key="quick_run"):
            classifier = get_classifier()
            with st.spinner("Analyzing parking lot..."):
                result = classifier.count_occupancy(quick_image)

            if result["total"] is None:
                st.warning("Model didn't return a parseable count. Raw response below.")
                st.code(result["raw"])
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Spaces", result["total"])
                c2.metric("Occupied", result["occupied"])
                c3.metric("Vacant", result["vacant"])
                c4.metric("Occupancy %", f"{result['occupancy_pct']}%")
    else:
        st.info("Upload an image above to begin.")

# =====================================================================
# PER-SLOT (ADVANCED) — exact box-by-box detection with overlay
# =====================================================================
with tab_advanced:
    st.caption("Define exact slot boxes for a precise per-slot overlay "
               "(also supports video).")

    mode = st.radio("Input type", ["Image", "Video"], horizontal=True, key="adv_mode")
    uploaded = st.file_uploader(
        "Upload parking image/video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        key="adv_upload",
    )

    ref_image = None
    video_path = None

    if uploaded is not None:
        if mode == "Image":
            ref_image = Image.open(uploaded).convert("RGB")
        else:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name
            ref_image = get_first_frame(video_path)

        if ref_image is not None:
            ref_image.save(REF_FRAME_PATH)

    with st.sidebar:
        st.header("1. Parking Slot Coordinates")

        if "slots" not in st.session_state:
            st.session_state.slots = load_slots()

        if ref_image is not None:
            st.success(f"Reference frame saved → `{REF_FRAME_PATH}`")
            st.code(f"python slot_picker.py {REF_FRAME_PATH} slots.json", language="bash")
            st.caption(
                "Run that in a terminal (drag boxes, **s** to save, **q** to quit), "
                "then upload the resulting slots.json below."
            )
        else:
            st.caption("Upload an image/video in the Advanced tab first.")

        slot_file = st.file_uploader("Upload slots.json", type=["json"], key="slot_json_upload")
        if slot_file is not None:
            st.session_state.slots = json.load(slot_file)

        slots_text = st.text_area(
            "Or edit slot JSON directly",
            value=json.dumps(st.session_state.slots, indent=2),
            height=200,
        )
        try:
            st.session_state.slots = json.loads(slots_text)
        except json.JSONDecodeError:
            st.warning("Invalid JSON — using last valid version.")

        if st.button("💾 Save as slots.json"):
            save_slots(st.session_state.slots)
            st.success("Saved.")

        st.header("2. Video Settings")
        interval = st.slider("Frame extraction interval (sec)", 1, 10, 3)

    slots = st.session_state.get("slots", {})

    def render_results(image: Image.Image, results: dict, caption: str):
        metrics = compute_metrics(results)
        overlay_img = draw_overlay(image, results)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(overlay_img, caption=caption, use_container_width=True)
        with col2:
            st.metric("Total Slots", metrics["total"])
            st.metric("Occupied", metrics["occupied"])
            st.metric("Available", metrics["available"])
            st.metric("Occupancy %", f"{metrics['occupancy_pct']}%")
            st.json(results)

    if uploaded is None:
        st.info("Upload a parking image or video above to begin.")
    elif not slots:
        st.info("Define parking slots in the sidebar (see steps above), then come back.")
    elif st.button("▶️ Run Detection", key="adv_run"):
        classifier = get_classifier()

        if mode == "Image":
            with st.spinner("Classifying slots..."):
                results = run_detection(ref_image, slots, classifier)
            render_results(ref_image, results, "Occupancy Overlay")
        else:
            with st.spinner("Extracting frames..."):
                frame_paths = extract_frames(video_path, interval_seconds=interval)
            st.info(f"Extracted {len(frame_paths)} frames.")
            
            if frame_paths:
                idx = (
                    st.slider("Frame", 0, len(frame_paths) - 1, 0)
                    if len(frame_paths) > 1
                    else 0
                )
                frame_img = Image.open(frame_paths[idx]).convert("RGB")
                with st.spinner("Classifying slots..."):
                    results = run_detection(frame_img, slots, classifier)
                render_results(frame_img, results, f"Frame {idx} Occupancy Overlay")