# ──────────────────────────────────────────────────────────────────────────────
# IPSD — Intelligent Parking Space Detection
# VLM Edition: Qwen2-VL-2B-Video-Instruct
#
# Run:   uv run streamlit run main_gui.py
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from vlm_classifier import ParkingLotCounter


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPSD · VLM Parking Detection",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:       #080C10;
    --panel:    #0F1419;
    --panel2:   #141C24;
    --border:   #1E2A36;
    --text:     #D8E1EB;
    --muted:    #5C6E82;
    --vacant:   #27C27B;
    --occupied: #E05252;
    --accent:   #38BDF8;
    --accent2:  #818CF8;
    --warn:     #F0A84A;
}

*, html, body { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }

/* ── Hero ── */
.hero {
    display: flex; align-items: center; gap: 16px;
    padding: 28px 0 16px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}
.hero-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent);
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.25);
    padding: 5px 12px; border-radius: 5px;
}
.hero h1 {
    font-size: 30px; font-weight: 700;
    letter-spacing: -0.03em; margin: 0;
}
.hero-sub {
    color: var(--muted); font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.01em;
    margin-top: 4px; margin-bottom: 24px;
}

/* ── Status pill ── */
.pill {
    display: inline-flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 5px 13px; border-radius: 20px;
}
.pill-ok   { background: rgba(39,194,123,0.08); color: var(--vacant);   border: 1px solid rgba(39,194,123,0.25); }
.pill-warn { background: rgba(240,168,74,0.08); color: var(--warn);     border: 1px solid rgba(240,168,74,0.25); }
.pill-err  { background: rgba(224,82,82,0.08);  color: var(--occupied); border: 1px solid rgba(224,82,82,0.25); }
.pill-idle { background: rgba(92,110,130,0.08); color: var(--muted);    border: 1px solid rgba(92,110,130,0.2); }

/* ── Metrics ── */
.metric-row { display: flex; gap: 14px; margin: 22px 0 30px 0; }
.mc {
    flex: 1; background: var(--panel);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 20px 22px;
    transition: border-color 0.25s;
}
.mc:hover { border-color: var(--border); }
.mc-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 10px;
}
.mc-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 40px; font-weight: 700; line-height: 1;
}
.mc-bar {
    margin-top: 10px; height: 3px; border-radius: 2px;
    background: var(--panel2);
    overflow: hidden;
}
.mc-bar-fill { height: 100%; border-radius: 2px; }

.mc.vacant   .mc-val { color: var(--vacant); }
.mc.occupied .mc-val { color: var(--occupied); }
.mc.total    .mc-val { color: var(--text); }
.mc.pct      .mc-val { color: var(--accent); }

/* ── Model reasoning box ── */
.reasoning-box {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--muted);
    line-height: 1.7;
    margin-top: 10px;
    white-space: pre-wrap;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--panel) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 24px !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: var(--panel); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px;
    font-weight: 500; padding: 10px 20px;
    transition: all 0.2s;
}
div.stButton > button:hover {
    border-color: var(--accent); color: var(--accent);
    box-shadow: 0 0 12px rgba(56,189,248,0.15);
}
div.stButton > button:active { transform: scale(0.98); }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Qwen2-VL-2B — one moment…")
def get_counter():
    return ParkingLotCounter()


# ── RENDER HELPERS ─────────────────────────────────────────────────────────────
def metric_html(label: str, value: str, css_class: str,
                bar_pct: float = 0.0, bar_color: str = "#38BDF8") -> str:
    fill = f'<div class="mc-bar-fill" style="width:{bar_pct:.0f}%;background:{bar_color};"></div>'
    return f"""
    <div class="mc {css_class}">
        <div class="mc-lbl">{label}</div>
        <div class="mc-val">{value}</div>
        <div class="mc-bar">{fill}</div>
    </div>"""


def render_metrics(total, occupied, vacant, pct, placeholder):
    if total is None:
        total_str    = "—"
        occupied_str = "—"
        vacant_str   = "—"
        pct_str      = "—"
        occ_bar      = 0.0
        vac_bar      = 0.0
    else:
        total_str    = str(total)
        occupied_str = str(occupied) if occupied is not None else "—"
        vacant_str   = str(vacant)   if vacant   is not None else "—"
        pct_str      = f"{pct}%" if pct is not None else "—"
        occ_bar      = (occupied / total * 100) if total else 0
        vac_bar      = (vacant   / total * 100) if total else 0

    html = f"""
    <div class="metric-row">
        {metric_html("🟢 Vacant Spaces",     vacant_str,   "vacant",   vac_bar, "#27C27B")}
        {metric_html("🔴 Occupied Spaces",   occupied_str, "occupied", occ_bar, "#E05252")}
        {metric_html("⬜ Total Spaces",       total_str,    "total",    100,    "#1E2A36")}
        {metric_html("📊 Occupancy Rate",    pct_str,      "pct",      occ_bar, "#38BDF8")}
    </div>"""
    placeholder.markdown(html, unsafe_allow_html=True)


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-badge">IPSD · VLM</span>
    <h1>Intelligent Parking Space Detection</h1>
</div>
<div class="hero-sub">Qwen2-VL-2B-Video-Instruct · chain-of-thought reasoning · accurate occupancy counting</div>
""", unsafe_allow_html=True)

# Model status
status = st.empty()
status.markdown('<span class="pill pill-idle">◌ model loading…</span>',
                unsafe_allow_html=True)

counter = get_counter()

actual_device = counter.device
if actual_device == "cpu":
    status.markdown(
        '<span class="pill pill-warn">⚠ Qwen2-VL-2B on CPU '
        '· inference ~10–30 s</span>',
        unsafe_allow_html=True)
else:
    status.markdown(
        f'<span class="pill pill-ok">● Qwen2-VL-2B ready · {actual_device.upper()}</span>',
        unsafe_allow_html=True)

# ── METRICS (initial empty) ───────────────────────────────────────────────────
metrics_ph = st.empty()
render_metrics(None, None, None, None, metrics_ph)

st.divider()

# ── INPUT ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown("#### Upload Parking Lot Image")
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=uploaded.name, width="stretch")

with col_right:
    st.markdown("#### Analysis")

    if not uploaded:
        st.markdown(
            '<span class="pill pill-idle">↑ Upload an image to begin</span>',
            unsafe_allow_html=True)

    else:
        run_btn = st.button("▶ Analyze Parking Lot", use_container_width=True)

        if run_btn:
            gpu_note = st.empty()
            if torch.cuda.is_available():
                free_mb = torch.cuda.memory_reserved(0) / 1e6
                total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
                if total_mb < 5000:
                    gpu_note.warning(
                        f"⚠️ Small GPU detected ({total_mb/1024:.1f} GiB). "
                        "If CUDA runs out of memory the model will automatically "
                        "retry on CPU — inference may take 30–60 seconds.")

            with st.spinner("Qwen2-VL-2B is analyzing the image…"):
                import torch as _torch
                result = counter.count(image)
            gpu_note.empty()


            total    = result["total"]
            occupied = result["occupied"]
            vacant   = result["vacant"]
            pct      = result["occupancy_pct"]
            raw      = result["raw"]

            render_metrics(total, occupied, vacant, pct, metrics_ph)

            if total is None:
                st.warning(
                    "The model couldn't parse a structured count. "
                    "See the raw response below — the image may be unclear "
                    "or lack visible parking space markings.")
            else:
                if pct is not None and pct >= 80:
                    st.error(f"🔴 Lot is **{pct}% full** — nearly no spaces available.")
                elif pct is not None and pct >= 50:
                    st.warning(f"🟡 Lot is **{pct}% full** — limited availability.")
                else:
                    st.success(f"🟢 Lot is **{pct}% full** — plenty of spaces available.")

            # Model reasoning
            with st.expander("🧠 Model Reasoning", expanded=(total is None)):
                st.markdown(
                    f'<div class="reasoning-box">{raw}</div>',
                    unsafe_allow_html=True)

# ── SIDEBAR — tips ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
**Model**: Qwen2-VL-2B-Video-Instruct

**How it works**:
1. Upload an overhead/aerial photo of a parking lot
2. The model uses chain-of-thought reasoning to:
   - Identify all painted parking spaces
   - Check each space for a vehicle
   - Report Occupied / Vacant / Total

**Best results with**:
- Clear overhead/bird's-eye view
- Good lighting
- Visible lane markings
""")
    st.markdown("---")
    st.markdown("""
**Tips for accuracy**:
- Use high-resolution images
- Avoid extreme angles
- Make sure parking lines are visible
""")
