"""Overlay drawing helpers (cv2)."""
from __future__ import annotations

import cv2
import numpy as np


def draw_overlay(frame: np.ndarray, slots, occupancy: dict, boxes=None, show_boxes: bool = True) -> np.ndarray:
    out = frame.copy()

    if show_boxes and boxes:
        for (x1, y1, x2, y2, conf, cls_name) in boxes:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)

    for slot in slots:
        pts = np.array(slot.points, dtype=np.int32)
        if len(pts) < 3:
            continue
        occ = occupancy.get(slot.id, False)
        color = (0, 0, 255) if occ else (0, 200, 0)  # BGR: red=occupied, green=free

        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        cv2.polylines(out, [pts], True, color, 2)

        cx, cy = pts.mean(axis=0).astype(int)
        label = slot.label or str(slot.id)
        cv2.putText(out, label, (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    free = sum(1 for v in occupancy.values() if not v)
    total = len(occupancy)
    cv2.rectangle(out, (0, 0), (220, 40), (0, 0, 0), -1)
    cv2.putText(out, f"Free: {free}/{total}", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out
