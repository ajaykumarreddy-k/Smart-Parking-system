import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from ultralytics import YOLO

class ParkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Parking System")
        self.root.geometry("1000x700")
        
        # YOLO model paths
        self.model_onnx_path = 'best.onnx'
        self.model_yolov8_path = 'yolov8n.pt'
        self.model_custom_path = 'parking_yolo.pt'
        
        # Load ONNX model
        if os.path.exists(self.model_onnx_path):
            self.model_onnx = cv2.dnn.readNetFromONNX(self.model_onnx_path)
            self.onnx_img_w, self.onnx_img_h = 640, 640
        else:
            self.model_onnx = None
            messagebox.showwarning("Warning", f"ONNX Model '{self.model_onnx_path}' not found! Ensemble will skip it.")

        # Load standard YOLOv8 model
        try:
            self.model_yolov8 = YOLO(self.model_yolov8_path)
        except Exception as e:
            self.model_yolov8 = None
            messagebox.showwarning("Warning", f"Failed to load YOLOv8 model: {e}")
            
        # Load standard YOLO11n model
        self.model_yolo11_path = 'yolo11n.pt'
        try:
            self.model_yolo11 = YOLO(self.model_yolo11_path)
        except Exception as e:
            self.model_yolo11 = None
            messagebox.showwarning("Warning", f"Failed to load YOLO11n model: {e}")
            
        # Load custom YOLO model
        if os.path.exists(self.model_custom_path):
            try:
                self.model_custom = YOLO(self.model_custom_path)
            except Exception as e:
                self.model_custom = None
                messagebox.showwarning("Warning", f"Failed to load custom YOLO model: {e}")
        else:
            self.model_custom = None
            messagebox.showwarning("Warning", f"Custom Model '{self.model_custom_path}' not found! Ensemble will skip it.")
        
        # App state
        self.cap = None
        self.is_video_playing = False
        self.after_id = None
        self.max_auto_spots = 0
        
        
        self.setup_ui()
        
    def setup_ui(self):
        # Configure grid
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        
        # --- Sidebar ---
        self.sidebar = tk.Frame(self.root, width=250, bg="#2C3E50", padx=15, pady=20)
        self.sidebar.grid(row=0, column=0, sticky="nswe")
        
        # Title
        title_lbl = tk.Label(self.sidebar, text="PARKING SYSTEM", bg="#2C3E50", fg="white", font=("Arial", 16, "bold"))
        title_lbl.pack(pady=(0, 20))
        
        # Controls
        controls_frame = tk.Frame(self.sidebar, bg="#2C3E50")
        controls_frame.pack(fill="x", pady=10)
        
        tk.Label(controls_frame, text="Total Parking Spots:", bg="#2C3E50", fg="white", font=("Arial", 11)).pack(anchor="w")
        self.total_spots_var = tk.StringVar(value="15")
        self.total_spots_entry = tk.Entry(controls_frame, textvariable=self.total_spots_var, font=("Arial", 12), width=10)
        self.total_spots_entry.pack(anchor="w", pady=(5, 5))
        
        tk.Label(controls_frame, text="Total Spots Mode:", bg="#2C3E50", fg="white", font=("Arial", 11)).pack(anchor="w", pady=(5, 0))
        self.mode_var = tk.StringVar(value="Manual")
        self.mode_menu = tk.OptionMenu(controls_frame, self.mode_var, "Manual", "Geometric Math", "AI Counting")
        self.mode_menu.config(bg="#34495E", fg="white", highlightthickness=0, width=15)
        self.mode_menu.pack(anchor="w", pady=(2, 10))
        
        tk.Label(controls_frame, text="Confidence Threshold:", bg="#2C3E50", fg="white", font=("Arial", 11)).pack(anchor="w")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.conf_slider = tk.Scale(controls_frame, from_=0.01, to=1.0, resolution=0.01, orient="horizontal", variable=self.conf_var, bg="#2C3E50", fg="white", highlightthickness=0)
        self.conf_slider.pack(fill="x", pady=(0, 10))

        tk.Label(controls_frame, text="Inference Size (imgsz):", bg="#2C3E50", fg="white", font=("Arial", 11)).pack(anchor="w")
        self.imgsz_var = tk.StringVar(value="1024")
        self.imgsz_menu = tk.OptionMenu(controls_frame, self.imgsz_var, "640", "1024", "1280", "1920")
        self.imgsz_menu.config(bg="#34495E", fg="white", highlightthickness=0)
        self.imgsz_menu.pack(fill="x", pady=(0, 15))
        
        # Buttons
        self.btn_load_img = tk.Button(self.sidebar, text="Load Image", command=self.load_image, bg="#3498DB", fg="white", font=("Arial", 11, "bold"), pady=5)
        self.btn_load_img.pack(fill="x", pady=5)
        
        self.btn_load_vid = tk.Button(self.sidebar, text="Load Video", command=self.load_video, bg="#3498DB", fg="white", font=("Arial", 11, "bold"), pady=5)
        self.btn_load_vid.pack(fill="x", pady=5)
        
        self.btn_stop_vid = tk.Button(self.sidebar, text="Stop Video", command=self.stop_video, bg="#E74C3C", fg="white", font=("Arial", 11, "bold"), pady=5, state=tk.DISABLED)
        self.btn_stop_vid.pack(fill="x", pady=5)
        
        # Stats Panel
        stats_frame = tk.Frame(self.sidebar, bg="#34495E", padx=10, pady=10)
        stats_frame.pack(fill="x", pady=30)
        
        tk.Label(stats_frame, text="Real-Time Stats", bg="#34495E", fg="#ECF0F1", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        self.lbl_total = tk.Label(stats_frame, text="Total Spots: 15", bg="#34495E", fg="#3498DB", font=("Arial", 11, "bold"))
        self.lbl_total.pack(anchor="w", pady=2)
        
        self.lbl_occupied = tk.Label(stats_frame, text="Occupied: 0", bg="#34495E", fg="#E74C3C", font=("Arial", 11, "bold"))
        self.lbl_occupied.pack(anchor="w", pady=2)
        
        self.lbl_vacant = tk.Label(stats_frame, text="Vacant: 15", bg="#34495E", fg="#2ECC71", font=("Arial", 11, "bold"))
        self.lbl_vacant.pack(anchor="w", pady=2)
        
        # --- Main Display ---
        self.main_display = tk.Frame(self.root, bg="#ECF0F1")
        self.main_display.grid(row=0, column=1, sticky="nswe")
        
        self.canvas = tk.Canvas(self.main_display, bg="#BDC3C7", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Store current photo to prevent garbage collection and current raw frame to redraw on resize
        self.current_photo = None
        self.current_frame = None

    def on_canvas_resize(self, event):
        if self.current_frame is not None:
            self.display_image(self.current_frame)

    def load_image(self):
        self.stop_video()
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                processed_frame, cars, empty = self.process_frame(frame)
                self.update_stats(cars, empty)
                self.current_frame = processed_frame
                self.display_image(processed_frame)

    def load_video(self):
        self.stop_video()
        file_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4 *.avi *.mkv"), ("All files", "*.*")])
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.is_video_playing = True
                self.btn_load_img.config(state=tk.DISABLED)
                self.btn_load_vid.config(state=tk.DISABLED)
                self.btn_stop_vid.config(state=tk.NORMAL)
                self.video_loop()
            else:
                messagebox.showerror("Error", "Could not open video file.")

    def stop_video(self):
        self.is_video_playing = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        self.btn_load_img.config(state=tk.NORMAL)
        self.btn_load_vid.config(state=tk.NORMAL)
        self.btn_stop_vid.config(state=tk.DISABLED)

    def video_loop(self):
        if self.is_video_playing and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                processed_frame, cars, empty = self.process_frame(frame)
                self.update_stats(cars, empty)
                self.current_frame = processed_frame
                self.display_image(processed_frame)
                
                # Delay for ~30 FPS
                self.after_id = self.root.after(33, self.video_loop)
            else:
                self.stop_video()

    def process_frame(self, frame):
        try:
            total_slots = int(self.total_spots_var.get())
        except ValueError:
            total_slots = 15

        try:
            conf_thresh = self.conf_var.get()
        except:
            conf_thresh = 0.25
            
        try:
            imgsz = int(self.imgsz_var.get())
        except:
            imgsz = 640

        # 1. Gather all Car Bounding Boxes
        all_car_boxes = []
        all_car_scores = []
        
        img = frame.copy()
        frame_h, frame_w = img.shape[:2]
        
        # SAHI (Slicing Aided Hyper Inference) removed for native full image resolution
        h, w = frame.shape[:2]
        
        slices = [
            (0, 0, w, h)            # Full image natively
        ]
        
        vehicle_classes = [2, 5, 7] # COCO car, bus, truck
        
        for idx_slice, (sx, sy, ex, ey) in enumerate(slices):
            slice_frame = frame[sy:ey, sx:ex]
            slice_h, slice_w = slice_frame.shape[:2]
            
            # 1. ONNX Model Inference (Cars) - Only on full image to prevent zoom distortion
            if self.model_onnx is not None and idx_slice == 0:
                blob = cv2.dnn.blobFromImage(slice_frame, 1/255, (self.onnx_img_w, self.onnx_img_h), swapRB=True, mean=(0, 0, 0), crop=False)
                self.model_onnx.setInput(blob)
                outputs = self.model_onnx.forward(self.model_onnx.getUnconnectedOutLayersNames())
                out = outputs[0]
                n_detections = out.shape[1]
                
                x_scale = slice_w / self.onnx_img_w
                y_scale = slice_h / self.onnx_img_h
    
                for i in range(n_detections):
                    detect = out[0][i]
                    confidence = float(detect[4])
                    if confidence >= conf_thresh:
                        class_score = detect[5:]
                        class_id = np.argmax(class_score)
                        if class_id == 0 and float(class_score[class_id]) > 0.5:
                            x, y, det_w, det_h = detect[0], detect[1], detect[2], detect[3]
                            left = int((x - det_w/2) * x_scale) + sx
                            top = int((y - det_h/2) * y_scale) + sy
                            box_width = int(det_w * x_scale)
                            box_height = int(det_h * y_scale)
                            
                            all_car_boxes.append([left, top, box_width, box_height])
                            all_car_scores.append(confidence)
            
            # YOLOv8n Inference
            if self.model_yolov8 is not None:
                results_v8 = self.model_yolov8(slice_frame, verbose=False, conf=conf_thresh, imgsz=imgsz)
                for box in results_v8[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1 += sx
                        x2 += sx
                        y1 += sy
                        y2 += sy
                        all_car_boxes.append([x1, y1, x2 - x1, y2 - y1])
                        all_car_scores.append(float(box.conf[0]))
                        
            # YOLO11n Inference
            if self.model_yolo11 is not None:
                results_v11 = self.model_yolo11(slice_frame, verbose=False, conf=conf_thresh, imgsz=imgsz)
                for box in results_v11[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1 += sx
                        x2 += sx
                        y1 += sy
                        y2 += sy
                        all_car_boxes.append([x1, y1, x2 - x1, y2 - y1])
                        all_car_scores.append(float(box.conf[0]))

        # Run NMS to merge overlapping car bounding boxes from both models
        cars_count = 0
        final_cars = []
        if len(all_car_boxes) > 0:
            indices = cv2.dnn.NMSBoxes(all_car_boxes, all_car_scores, conf_thresh, 0.30)
            
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                box = all_car_boxes[idx]
                final_cars.append(box)
                left, top, w, h = box[0], box[1], box[2], box[3]
                conf = all_car_scores[idx]
                
                cars_count += 1
                
                # Draw Red Car Box
                cv2.rectangle(img, (left, top), (left + w, top + h), (0, 0, 255), 2)
                label = f"car {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                dim_text, baseline = text_size[0], text_size[1]
                cv2.rectangle(img, (left, top - 20), (left + dim_text[0], top + dim_text[1] + baseline - 20), (0, 0, 0), cv2.FILLED)
                cv2.putText(img, label, (left, top + dim_text[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 2. Custom Model Inference (Empty and Occupied Spaces)
        custom_spaces_count = 0
        vacant_spaces_count = 0
        if self.model_custom is not None:
            results_custom = self.model_custom(frame, verbose=False, conf=conf_thresh, imgsz=imgsz)
            for box in results_custom[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = results_custom[0].names[int(box.cls[0])]
                
                if "space" in cls_name.lower():
                    custom_spaces_count += 1
                if "empty" in cls_name.lower():
                    vacant_spaces_count += 1
                
                # Draw Green Empty/Occupied Space Box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                dim_text, baseline = text_size[0], text_size[1]
                cv2.rectangle(img, (x1, y1 - 20), (x1 + dim_text[0], y1 + dim_text[1] + baseline - 20), (0, 0, 0), cv2.FILLED)
                cv2.putText(img, label, (x1, y1 + dim_text[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 3. Auto-Detect Total Spots Logic
        mode = self.mode_var.get()
        
        if mode == "Geometric Math" and len(final_cars) > 0:
            widths = [box[2] for box in final_cars]
            heights = [box[3] for box in final_cars]
            median_w = np.median(widths)
            median_h = np.median(heights)
            
            if median_w > 0 and median_h > 0:
                is_vertical_parking = median_h > median_w
                lanes = []
                
                for box in final_cars:
                    x, y, w, h = box
                    cx = x + w / 2
                    cy = y + h / 2
                    
                    placed = False
                    for lane in lanes:
                        if is_vertical_parking:
                            if abs(lane['avg_center'] - cy) < median_h * 0.75:
                                lane['cars'].append(box)
                                lane['avg_center'] = sum(b[1] + b[3]/2 for b in lane['cars']) / len(lane['cars'])
                                placed = True
                                break
                        else:
                            if abs(lane['avg_center'] - cx) < median_w * 0.75:
                                lane['cars'].append(box)
                                lane['avg_center'] = sum(b[0] + b[2]/2 for b in lane['cars']) / len(lane['cars'])
                                placed = True
                                break
                    if not placed:
                        lanes.append({'cars': [box], 'avg_center': cy if is_vertical_parking else cx})
                
                num_lanes = len(lanes)
                if is_vertical_parking:
                    slots_per_lane = frame_w / median_w
                else:
                    slots_per_lane = frame_h / median_h
                    
                calc_total = int(num_lanes * slots_per_lane)
                self.max_auto_spots = max(self.max_auto_spots, calc_total)
                total_slots = self.max_auto_spots
                self.total_spots_var.set(str(total_slots))
                
        elif mode == "AI Counting":
            # Minimum possible spots = detected cars + detected empty spaces
            min_spots = cars_count + vacant_spaces_count
            self.max_auto_spots = max(self.max_auto_spots, min_spots, custom_spaces_count)
            total_slots = self.max_auto_spots
            self.total_spots_var.set(str(total_slots))

        empty_slots = total_slots - cars_count
        return img, cars_count, empty_slots

    def update_stats(self, cars, empty):
        try:
            total = int(self.total_spots_var.get())
        except ValueError:
            total = 0
            
        self.lbl_total.config(text=f"Total Spots: {total}")
        self.lbl_occupied.config(text=f"Occupied: {cars}")
        self.lbl_vacant.config(text=f"Vacant: {empty}")

    def display_image(self, frame):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get canvas dimensions
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 10 and canvas_h > 10:
            # Resize image to fit canvas while maintaining aspect ratio
            img_h, img_w = rgb_image.shape[:2]
            ratio = min(canvas_w / img_w, canvas_h / img_h)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)
            
            resized_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(resized_image)
            self.current_photo = ImageTk.PhotoImage(image=pil_image)
            
            # Clear canvas and draw new image centered
            self.canvas.delete("all")
            x_center = canvas_w // 2
            y_center = canvas_h // 2
            self.canvas.create_image(x_center, y_center, image=self.current_photo, anchor=tk.CENTER)

def main():
    root = tk.Tk()
    app = ParkingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
