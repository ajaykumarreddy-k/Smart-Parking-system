# Smart Parking Space Detection (SmolVLM-256M)

Implements PRD v1.0: VLM-based occupancy detection, no custom training.

## Files
| File | Maps to PRD |
|---|---|
| `app.py` | Streamlit dashboard — upload, overlay, metrics (FR-01,02,07,08) |
| `slot_picker.py` | Manual slot annotation tool, drag boxes → `slots.json` (FR-04) |
| `slot_config.py` | Load/save/validate slot coordinates (FR-04) |
| `frame_extractor.py` | Video → frames every N sec, default 3s (FR-03) |
| `vlm_classifier.py` | SmolVLM-256M inference, Occupied/Empty (FR-06) |
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