"""Vehicle detector built on Ultralytics YOLO."""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

# COCO class ids for vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(self, model_path: str = "yolo11n.pt", conf: float = 0.35, device: str | None = None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float, str]]:
        """Run inference on a BGR frame. Returns (x1, y1, x2, y2, conf, class_name)."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            classes=list(VEHICLE_CLASSES.keys()),
            device=self.device,
            verbose=False,
        )[0]

        boxes = []
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            boxes.append((x1, y1, x2, y2, conf, VEHICLE_CLASSES.get(cls_id, "vehicle")))
        return boxes

    def detect_debug(self, frame: np.ndarray, conf: float = 0.05):
        """No class filter, low conf — see exactly what the model perceives.
        Use this to tell apart 'weights are broken' vs 'domain gap' (e.g. top-down
        photos look very different to a model trained on street-level COCO images)."""
        results = self.model.predict(frame, conf=conf, device=self.device, verbose=False)[0]
        out = []
        for b in results.boxes:
            cls_id = int(b.cls[0])
            out.append((self.model.names.get(cls_id, str(cls_id)), float(b.conf[0])))
        return sorted(out, key=lambda t: -t[1])
