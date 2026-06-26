"""FR-03: Extract frames from uploaded videos at configurable intervals (default 3s)."""
import cv2
from pathlib import Path
from PIL import Image


def get_first_frame(video_path: str) -> Image.Image | None:
    """Read just the first frame, for use as a slot-drawing preview."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def extract_frames(video_path: str, interval_seconds: float = 3, out_dir: str = "frames") -> list[str]:
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps * interval_seconds))

    frames = []
    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            out_path = str(Path(out_dir) / f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            frames.append(out_path)
            saved += 1
        idx += 1
    cap.release()
    return frames
