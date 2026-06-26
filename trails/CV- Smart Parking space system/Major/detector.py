"""FR-05: Slot cropping. FR-07: Aggregate occupancy metrics for the dashboard."""
from PIL import Image


def crop_slot(image: Image.Image, coords: list) -> Image.Image:
    x1, y1, x2, y2 = coords
    return image.crop((x1, y1, x2, y2))


def run_detection(image: Image.Image, slots: dict, classifier) -> dict:
    """Returns {slot_id: {coords, label, raw}}."""
    results = {}
    for slot_id, coords in slots.items():
        crop = crop_slot(image, coords)
        res = classifier.classify(crop)
        results[slot_id] = {"coords": coords, **res}
    return results


def compute_metrics(results: dict) -> dict:
    total = len(results)
    occupied = sum(1 for r in results.values() if r["label"] == "Occupied")
    available = total - occupied
    pct = round((occupied / total) * 100, 1) if total else 0.0
    return {
        "total": total,
        "occupied": occupied,
        "available": available,
        "occupancy_pct": pct,
    }
