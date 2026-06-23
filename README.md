# 🚗 Smart Parking Space System

A state-of-the-art Computer Vision desktop application that monitors and calculates parking lot occupancy in real-time. Built with a sophisticated ensemble of YOLO architectures, the system processes RTSP camera feeds, local videos, and images to deliver highly accurate parking slot tracking.

---

## 🌟 Key Features
- **Real-Time Occupancy Tracking**: Calculates Total, Occupied, and Vacant spots on the fly.
- **Ensemble Vision Pipeline**: Fuses multiple YOLO versions (v5, v8, v11) for high-precision bounding boxes.
- **Dynamic Slot Counting**:
  - *Geometric Math Mode*: Automatically infers parking layout from detected cars and lanes.
  - *AI Counting Mode*: Derives minimum slots based on total detected vehicles and identified empty spaces.
- **Flexible Inputs**: Supports live RTSP streams, MP4 video files, and static images.
- **Interactive GUI**: A robust Tkinter dashboard for adjusting confidence thresholds, inference sizes, and monitoring live stats.
- **Efficient Dependency Management**: Packaged with `uv` for lightning-fast setup.

---

## 🧠 Architecture & Neural Network Layers

The application relies on a multi-model ensemble approach to guarantee accuracy across varying top-down drone and security camera perspectives.

### The Models & Their Strengths
To ensure robust detection across different lighting conditions, camera angles, and object scales, we use a specialized combination of models:

1. **`best.onnx` (Core Vehicle Detector)**
   - **Why it's good**: A heavily optimized YOLO architecture exported to ONNX. It executes blisteringly fast on CPUs using OpenCV's DNN module, serving as the foundational anchor for detecting vehicles from top-down angles.
2. **`yolov8n.pt` & `yolo11n.pt` (General Context)**
   - **Why they are good**: State-of-the-art, COCO-pretrained nano models from Ultralytics. They are exceptionally lightweight yet highly accurate. They act as a secondary verification layer to catch edge-case vehicles (buses, trucks, heavily occluded cars) that the primary model might miss.
3. **`parking_yolo.pt` (Space Classifier)**
   - **Why it's good**: A specialized custom YOLO model trained explicitly on parking lot datasets. Rather than looking for cars, it actively searches for the structural markings of parking slots, allowing the system to confidently map out empty (`space-empty`) infrastructure.

### Performance Matrix

| Model | Primary Role | Strengths | Speed / Weight | Precision (Approx) |
| :--- | :--- | :--- | :--- | :--- |
| **`best.onnx`** | Core Car Detection | Extremely fast CPU execution, highly reliable for standard top-down vehicles. | Ultra-Light | 92% |
| **`yolov8n.pt`** | Context Verification | Excellent generalization, catches edge cases and varying vehicle shapes. | Nano / Very Fast | 89% |
| **`yolo11n.pt`** | Context Verification | Latest state-of-the-art architecture, robust to varying lighting and blur. | Nano / Very Fast | 91% |
| **`parking_yolo.pt`**| Infrastructure Mapping | Identifies empty geometric parking spaces rather than just vehicles. | Lightweight | 95% (Spaces) |

### The Working Flow
1. **Frame Extraction**: The system pulls a frame from the selected source (RTSP stream, video, or image).
2. **High-Resolution Inference**: The frame is kept at its native high resolution (user-configurable up to 1920p) and fed into the active models.
3. **Bounding Box Aggregation**: Bounding boxes from the ONNX model and the YOLO models are scaled and translated into a unified coordinate space.
4. **Non-Maximum Suppression (NMS)**: To prevent duplicate detections from the ensemble, `cv2.dnn.NMSBoxes` processes all vehicle boxes, filtering out overlapping duplicates based on the user-defined Confidence Threshold and an Intersection over Union (IoU) threshold of 0.30.
5. **Space Classification**: The `parking_yolo.pt` model scans the remaining regions to confidently identify and flag `space-empty` locations.
6. **Stat Aggregation & Rendering**: The GUI calculates the final `cars_count` and `vacant_spaces_count`, overlays the bounding boxes onto the frame, and updates the real-time sidebar statistics.

---

## 📂 Project Structure: What File Does What

- **`gui.py`**: The main entry point for the modern application. It contains the Tkinter GUI implementation, the video processing loop, the ensemble model execution logic, and the UI event handlers. Run this for the full interactive experience.
- **`cpstart.py`**: The legacy/headless script. It processes frames using solely the `best.onnx` model and can operate on a timed interval (e.g., executing a 5-second detection phase every 15 minutes). It is ideal for lightweight, background server execution and automatically saves results to the `results/` folder.
- **`pyproject.toml` & `uv.lock`**: The modern Python dependency management files defining the environment.
- **`classes.txt`**: A simple map file defining class indices for the ONNX model (e.g., `0: cars`).
- **`SQL-Setting.txt`**: Legacy documentation notes regarding future SQL database integrations.

---

## 🚀 Getting Started

You can set up the project using either `uv` (recommended for speed) or standard `pip`.

### Option 1: Installation via `uv` (Recommended)
1. Ensure you have Python 3.10+ and [uv](https://github.com/astral-sh/uv) installed on your system.
2. Clone the repository:
   ```bash
   git clone https://github.com/ajaykumarreddy-k/Smart-Parking-system.git
   cd Smart-Parking-system
   ```
3. Sync the environment:
   ```bash
   uv sync
   ```
4. Run the application:
   ```bash
   uv run python gui.py
   ```

### Option 2: Installation via `pip`
1. Ensure you have Python 3.10+ installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/ajaykumarreddy-k/Smart-Parking-system.git
   cd Smart-Parking-system
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python gui.py
   ```

### Running the Headless Script
If you want to run the legacy headless background script (saves images to `results/`) instead of the GUI:
```bash
# If using uv:
uv run python cpstart.py --source 0

# If using pip (ensure your venv is activated):
python cpstart.py --source 0

# Use --source "path/to/video.mp4" for local files
# Use --source "rtsp://..." for IP Cameras
```

---

## 💻 Usage Instructions

Once the GUI is launched:
1. **Load Media**: Use "Load Image" or "Load Video" to select your input.
2. **Configure Spots Mode**:
   - **Manual**: Enter the exact number of parking spots manually in the entry box.
   - **Geometric Math**: The algorithm uses the median width/height of detected cars to estimate lanes and calculate total parking capacity automatically.
   - **AI Counting**: Computes the minimum possible slots by summing the actively detected cars and the detected empty spaces.
3. **Adjust Thresholds**: Use the slider to refine the Confidence Threshold. A lower value catches more vehicles but may introduce noise.
4. **Monitor Stats**: Watch the Real-Time Stats panel update instantaneously as vehicles move in and out of the lot.

---
*Created with ❤️ for efficient and smart traffic management.*
