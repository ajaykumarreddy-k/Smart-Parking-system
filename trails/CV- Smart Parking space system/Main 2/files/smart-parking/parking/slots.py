"""Parking slot polygons and occupancy computation."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from shapely.geometry import Polygon, box as shapely_box


@dataclass
class Slot:
    id: int
    points: list = field(default_factory=list)  # [[x, y], ...] >= 3 points
    label: str = ""

    def polygon(self) -> Polygon:
        return Polygon(self.points)


def load_slots(path: str | Path) -> list[Slot]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [Slot(**s) for s in data]


def save_slots(path: str | Path, slots: list[Slot]) -> None:
    Path(path).write_text(json.dumps([asdict(s) for s in slots], indent=2))


def compute_occupancy(
    slots: list[Slot],
    boxes: list[tuple],
    overlap_thresh: float = 0.15,
) -> dict[int, bool]:
    """A slot is 'occupied' if a detected vehicle box covers more than
    `overlap_thresh` fraction of the slot polygon's area."""
    occupied: dict[int, bool] = {}
    for slot in slots:
        poly = slot.polygon()
        is_occ = False
        if poly.is_valid and poly.area > 0:
            for (x1, y1, x2, y2, conf, cls_name) in boxes:
                vbox = shapely_box(x1, y1, x2, y2)
                inter = poly.intersection(vbox).area
                if inter / poly.area > overlap_thresh:
                    is_occ = True
                    break
        occupied[slot.id] = is_occ
    return occupied
