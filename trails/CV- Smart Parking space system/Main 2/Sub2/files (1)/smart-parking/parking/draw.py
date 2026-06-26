"""Overlay drawing helpers (cv2)."""
from __future__ import annotations

import cv2
import numpy as np


def draw_boxes(frame: np.ndarray, boxes) -> np.ndarray:
    out = frame.copy()
    for (x1, y1, x2, y2, conf, cls_name) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
        cv2.putText(out, cls_name, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    return out


def draw_banner(frame: np.ndarray, occupied: int, total: int, roi=None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)

    vacant = max(total - occupied, 0)
    full = occupied >= total
    color = (0, 0, 255) if full else (0, 200, 0)

    bw = min(360, w)
    cv2.rectangle(out, (0, 0), (bw, 60), (0, 0, 0), -1)
    cv2.putText(out, f"VACANT: {vacant} / {total}", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(out, f"Occupied: {occupied}", (14, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out
