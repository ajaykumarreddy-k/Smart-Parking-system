import cv2
import numpy as np
import argparse
import time
import datetime
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Car Parking System YOLOv5')
parser.add_argument('--source', type=str, default='0', help='Source: 0 for webcam, or path to an image/video file')
args = parser.parse_args()

model = 'best.onnx'
img_w = 640
img_h = 640
classes_file = 'classes.txt'

def class_names():
    classes = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as file:
            for line in file:
                name = line.strip('\n')
                classes.append(name)
    return classes

width_frame = 640
net = cv2.dnn.readNetFromONNX(model)
classes = {0: "cars"}  # Default classes mapping

total_slots = 15

# Specify the directory path to save the image results
output_directory = 'results'
os.makedirs(output_directory, exist_ok=True)

def process_frame(frame):
    height = int(frame.shape[0] * (width_frame / frame.shape[1]))
    dim = (width_frame, height)
    img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    blob = cv2.dnn.blobFromImage(img, 1/255, (img_w, img_h), swapRB=True, mean=(0, 0, 0), crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    out = outputs[0]
    n_detections = out.shape[1]
    
    img_height, img_width = img.shape[:2]
    x_scale = img_width / img_w
    y_scale = img_height / img_h
    conf_threshold = 0.7
    score_threshold = 0.5
    nms_threshold = 0.5

    class_ids = []
    score = []
    boxes = []

    for i in range(n_detections):
        detect = out[0][i]
        confidence = detect[4]
        if confidence >= conf_threshold:
            class_score = detect[5:]
            class_id = np.argmax(class_score)
            if class_id == 0 and class_score[class_id] > score_threshold:
                score.append(float(confidence))
                class_ids.append(class_id)
                x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                left = int((x - w/2) * x_scale)
                top = int((y - h/2) * y_scale)
                box_width = int(w * x_scale)
                box_height = int(h * y_scale)
                box = np.array([left, top, box_width, box_height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, score, conf_threshold, nms_threshold)
    
    cars_count = len(indices) if len(indices) > 0 else 0
    empty_slots = total_slots - cars_count

    for i in indices:
        # Check if i is a scalar or an array (depends on OpenCV version)
        idx = i[0] if isinstance(i, (list, np.ndarray)) else i
        box = boxes[idx]
        left, top, box_width, box_height = box[0], box[1], box[2], box[3]
        class_id = class_ids[idx]

        if class_id == 0:
            cv2.rectangle(img, (left, top), (left + box_width, top + box_height), (0, 0, 255), 2)
            label = "{}".format(classes.get(class_id, "cars"))
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            dim_text, baseline = text_size[0], text_size[1]
            cv2.rectangle(img, (left, top - 20), (left + dim_text[0], top + dim_text[1] + baseline - 20), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, label, (left, top + dim_text[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # Add text overlay
    text_width_count, text_height_count = cv2.getTextSize(f"Cars Count: {cars_count}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_width_slots, text_height_slots = cv2.getTextSize(f"Empty Slots: {empty_slots}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

    text_x_count = img.shape[1] - text_width_count - 5
    text_y_count = 35
    text_x_slots = img.shape[1] - text_width_slots - 5
    text_y_slots = 57

    cv2.putText(img, f"Cars Count: {cars_count}", (text_x_count, text_y_count),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Empty Slots: {empty_slots}", (text_x_slots, text_y_slots),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return img

source = args.source
is_image = source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

if is_image:
    print(f"Processing image file: {source}")
    frame = cv2.imread(source)
    if frame is None:
        print(f"Failed to load image: {source}")
        exit(1)
        
    result_img = process_frame(frame)
    
    # Save the image results
    filename = f"result_{os.path.basename(source)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    output_path = os.path.join(output_directory, filename)
    cv2.imwrite(output_path, result_img)
    print(f"Saved result to {output_path}")
    
    cv2.imshow("Object Detection", result_img)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # Handle video / webcam stream
    if source == '0':
        source = 0
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        exit(1)
        
    print(f"Starting video stream from source: {source}")
    
    # Initialize so that detection happens immediately on the first frame
    last_detection_time = datetime.datetime.now() - datetime.timedelta(minutes=15)

    while True:
        current_time = datetime.datetime.now()
        time_difference = current_time - last_detection_time

        if time_difference.total_seconds() >= (15 * 60):
            detection_start_time = datetime.datetime.now()
            detection_end_time = detection_start_time + datetime.timedelta(seconds=5)
            last_detection_time = detection_end_time

            print("Running 5-second detection phase...")
            while datetime.datetime.now() <= detection_end_time:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                result_img = process_frame(frame)

                # Save the image results
                filename = f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                output_path = os.path.join(output_directory, filename)
                cv2.imwrite(output_path, result_img)

                # Display the captured frame
                cv2.imshow("Object Detection", result_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Display the captured frame
            cv2.imshow("Camera Feed", frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()