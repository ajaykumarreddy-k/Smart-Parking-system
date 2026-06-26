"""FR-08: Visual overlay - green for available, red for occupied."""
from PIL import Image, ImageDraw

OCCUPIED_COLOR = (220, 30, 30)
AVAILABLE_COLOR = (30, 180, 30)


def draw_overlay(image: Image.Image, results: dict) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for slot_id, r in results.items():
        x1, y1, x2, y2 = r["coords"]
        color = OCCUPIED_COLOR if r["label"] == "Occupied" else AVAILABLE_COLOR
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, max(y1 - 14, 0)), f"{slot_id}: {r['label']}", fill=color)
    return img
