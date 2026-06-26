# ──────────────────────────────────────────────
# IPSD — Local inference with downloaded parking_yolo.pt
# Works on .mp4 video, webcam, or image folder
# ──────────────────────────────────────────────
from ultralytics import YOLO
import cv2

MODEL_PATH   = "models/parking_yolo.pt"   # downloaded from Kaggle
VIDEO_SOURCE = "data/carPark.mp4"         # or 0 for webcam
CONF_THRESH  = 0.4

model = YOLO(MODEL_PATH)
print(f"Classes: {model.names}")   # e.g. {0: 'space-empty', 1: 'space-occupied'}

cap = cv2.VideoCapture(VIDEO_SOURCE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]

    occupied = empty = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        is_occupied = "occupied" in cls_name.lower()
        color = (0, 0, 200) if is_occupied else (0, 200, 0)
        if is_occupied: occupied += 1
        else: empty += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    total = occupied + empty
    cv2.putText(frame, f"Free: {empty}/{total}",
                (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 220, 0), 2)

    cv2.imshow("IPSD — YOLO Parking Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
