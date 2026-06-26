# Smart Parking (Ultralytics YOLO)

No manual slot drawing. Set total capacity once, upload a photo/video, get
occupied/vacant instantly: `vacant = total_capacity - vehicles_detected`.

## Setup
```bash
uv sync
uv run streamlit run app.py
```
First run downloads YOLO weights automatically (yolo11n.pt is smallest/fastest).

## Workflow
1. Sidebar: enter **Total parking capacity** (number of marked spaces in the lot) once.
2. Upload an image or video of the lot.
3. App detects vehicles (car/bus/truck/motorcycle) → occupied = count detected,
   vacant = capacity − occupied. Boxes + a live count banner are drawn on the frame.

Optional: expand **"Crop region"** in the sidebar to exclude road/background via
simple %-sliders, so passing traffic outside the lot isn't counted.

## Why YOLO over a VLM here
- Fixed vehicle classes → no need for open-vocab reasoning.
- `yolo11n.pt`/`yolov8n.pt` run in real time on CPU; VLMs are 100x+ heavier for this.
- Counting detections directly skips per-slot geometry — simplest path from
  image to occupied/vacant.

## Tuning
- `conf` slider: raise if false-positive detections appear.
- Crop sliders: tighten if vehicles outside the lot get counted.
- Swap to `*s.pt` models in the sidebar for better accuracy at a speed cost.

## Limitations / next step
Counting is lot-wide, not per-space — it tells you *how many* are free, not
*which* space. If you need "Spot #4 is free", that requires marking spot
locations once (a config step, not avoidable for per-spot precision).

## Files
```
app.py              Streamlit UI
parking/detector.py YOLO wrapper (vehicle classes only)
parking/draw.py     Box + occupied/vacant banner overlay
```
