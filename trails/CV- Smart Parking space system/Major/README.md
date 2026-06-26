# Smart Parking Space Detection (Qwen2.5-VL)

Implements PRD v1.0: VLM-based occupancy detection, no custom training.

## Files
| File | Maps to PRD |
|---|---|
| `app.py` | Streamlit dashboard — upload, overlay, metrics (FR-01,02,07,08) |
| `slot_picker.py` | Manual slot annotation tool, drag boxes → `slots.json` (FR-04) |
| `slot_config.py` | Load/save/validate slot coordinates (FR-04) |
| `frame_extractor.py` | Video → frames every N sec, default 3s (FR-03) |
| `vlm_classifier.py` | Qwen2.5-VL-3B inference, Occupied/Empty (FR-06) |
| `detector.py` | Crop slots + run classifier + compute metrics (FR-05,07) |
| `overlay.py` | Green/red box overlay (FR-08) |
| `slots_example.json` | Sample 3-slot layout |

## Setup
```bash
# with uv (recommended, per PRD)
uv venv --python 3.12
uv pip install -r requirements.txt

# or plain pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU build of torch (CUDA) — install per https://pytorch.org/get-started/locally/
before the line above if `pip install torch` alone doesn't pick up CUDA.

## Run
```bash
uv pip install -r requirements.txt
uv run streamlit run app.py
```
Two tabs:
- **⚡ Quick Count** — upload an image, click Analyze, get occupied/vacant
  counts directly from the whole image. No slot setup. Less precise (the
  model estimates counts itself) but instant.
- **🎯 Per-Slot (Advanced)** — exact box-by-box results + overlay (also
  handles video):
  1. Upload image/video — saves `reference_frame.jpg`.
  2. Run the sidebar command: `python slot_picker.py reference_frame.jpg slots.json`
     (drag boxes, `s` to save, `q` to quit).
  3. Upload that `slots.json` in the sidebar.
  4. Click "Run Detection".

## Notes / things to verify before calling it done
- First run downloads `Qwen/Qwen2.5-VL-3B-Instruct` from Hugging Face (~7GB).
  If your 4GB-VRAM GPU OOMs, switch `model_id` in `vlm_classifier.py` to
  `"Qwen/Qwen2.5-VL-2B-Instruct"` (PRD's stated fallback).
- `classify()` does one VLM call per slot per frame — for many slots this is
  the main latency cost against the <5s/frame NFR. If too slow, batch crops
  in a single `processor(...)` call instead of looping (straightforward
  extension of `detector.run_detection`).
- Accuracy (≥80% target) depends entirely on prompt + crop tightness — tight,
  well-cropped slot boxes from `slot_picker.py` matter more than model choice.
- Not yet wired up (Phase 2+ in PRD, not required for v1): SAM2/Grounding DINO
  auto-discovery, multi-camera, cloud deploy.
