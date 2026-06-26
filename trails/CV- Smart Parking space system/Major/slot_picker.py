"""FR-04: Interactive slot annotation tool.

Usage:
    python slot_picker.py <image_path> [output_json]

Controls:
    Left-click-drag : draw a box for one slot
    s                : save slots to JSON
    z                : undo last slot
    q                : quit
"""
import sys
import json
import cv2

WINDOW = "Slot Picker  [drag=box, s=save, z=undo, q=quit]"


def pick_slots(image_path: str, out_path: str = "slots.json") -> dict:
    base = cv2.imread(image_path)
    if base is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    slots: dict[str, list[int]] = {}
    state = {"start": None, "counter": 1}

    def redraw():
        canvas = base.copy()
        for sid, (x1, y1, x2, y2) in slots.items():
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, sid, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["start"]:
            x1, y1 = state["start"]
            x1, x2 = sorted([x1, x])
            y1, y2 = sorted([y1, y])
            if x2 - x1 > 3 and y2 - y1 > 3:
                sid = f"slot_{state['counter']}"
                slots[sid] = [x1, y1, x2, y2]
                state["counter"] += 1
            state["start"] = None

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        cv2.imshow(WINDOW, redraw())
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            with open(out_path, "w") as f:
                json.dump(slots, f, indent=2)
            print(f"Saved {len(slots)} slots to {out_path}")
        elif key == ord("z") and slots:
            last_key = list(slots.keys())[-1]
            del slots[last_key]
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return slots


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slot_picker.py <image_path> [output_json]")
        sys.exit(1)
    img_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "slots.json"
    pick_slots(img_path, out)
