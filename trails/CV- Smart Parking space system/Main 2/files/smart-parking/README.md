# Smart Parking (Ultralytics YOLO)

Lightweight alternative to VLMs for parking occupancy: YOLO detects vehicles,
polygons mark each parking slot, overlap between the two decides free/occupied.

## Setup
```bash
uv sync
uv run streamlit run app.py
```
First run downloads YOLO weights automatically (yolo11n.pt is smallest/fastest).

## Workflow
1. **Define Slots** tab — upload a reference photo, draw one polygon per slot, save.
   `data/slots.json` stores `[{id, points:[[x,y],...], label}, ...]` in original image pixel coords.
2. **Detect Occupancy** tab — upload an image or video; vehicle boxes overlapping
   a slot beyond the threshold mark it red (occupied) / green (free).

## Why YOLO over a VLM here
- Fixed vehicle classes (car/bus/truck/motorcycle) → no need for open-vocab reasoning.
- `yolo11n.pt`/`yolov8n.pt` run in real time on CPU; VLMs are 100x+ heavier for this.
- Geometry (slot polygon ∩ box) does the "is it parked here" logic — the model
  only needs to detect, not reason about the scene.

## Tuning
- `conf` slider: raise if false-positive detections appear.
- Slot overlap % slider: lower it if cars partially overlap slot lines and don't trigger occupied.
- Swap to `*s.pt` models in the sidebar for better accuracy at a speed cost.

## Files
```
app.py              Streamlit UI
parking/detector.py YOLO wrapper (vehicle classes only)
parking/slots.py    Slot dataclass, JSON load/save, occupancy logic
parking/draw.py     Overlay rendering
data/slots.json     Saved slot polygons
```
