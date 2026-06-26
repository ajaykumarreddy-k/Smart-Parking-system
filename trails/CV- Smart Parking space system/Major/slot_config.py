"""FR-04: Parking slot coordinate mapping (manual annotation, JSON-based)."""
import json
from pathlib import Path

DEFAULT_PATH = "slots.json"


def load_slots(path: str = DEFAULT_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return json.load(f)


def save_slots(slots: dict, path: str = DEFAULT_PATH) -> None:
    with open(path, "w") as f:
        json.dump(slots, f, indent=2)


def validate_slot(coords: list, img_w: int, img_h: int) -> bool:
    """coords = [x1, y1, x2, y2]"""
    if len(coords) != 4:
        return False
    x1, y1, x2, y2 = coords
    if x1 >= x2 or y1 >= y2:
        return False
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return False
    return True


def validate_all(slots: dict, img_w: int, img_h: int) -> dict:
    """Returns {slot_id: is_valid}."""
    return {sid: validate_slot(coords, img_w, img_h) for sid, coords in slots.items()}
