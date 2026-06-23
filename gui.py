import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from ultralytics import YOLO
import customtkinter as ctk

# Set Appearance and Theme to look premium
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ParkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Parking Vision | Enterprise Dashboard")
        self.root.geometry("1280x800")
        self.root.configure(fg_color="#0F111A") # Deep sleek background
        
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
            print(f"Warning: ONNX Model '{self.model_onnx_path}' not found! Ensemble will skip it.")

        # Load standard YOLOv8 model
        try:
            self.model_yolov8 = YOLO(self.model_yolov8_path)
        except Exception as e:
            self.model_yolov8 = None
            print(f"Warning: Failed to load YOLOv8 model: {e}")
            
        # Load standard YOLO11n model
        self.model_yolo11_path = 'yolo11n.pt'
        try:
            self.model_yolo11 = YOLO(self.model_yolo11_path)
        except Exception as e:
            self.model_yolo11 = None
            print(f"Warning: Failed to load YOLO11n model: {e}")
            
        # Load custom YOLO model
        if os.path.exists(self.model_custom_path):
            try:
                self.model_custom = YOLO(self.model_custom_path)
            except Exception as e:
                self.model_custom = None
                print(f"Warning: Failed to load custom YOLO model: {e}")
        else:
            self.model_custom = None
            print(f"Warning: Custom Model '{self.model_custom_path}' not found! Ensemble will skip it.")
        
        # App state
        self.cap = None
        self.is_video_playing = False
        self.after_id = None
        self.max_auto_spots = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main Layout Configuration
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # --- Sidebar (Left Panel) ---
        self.sidebar = ctk.CTkFrame(self.root, width=320, corner_radius=0, fg_color="#151821", border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1) # Push stats to bottom
        
        # 1. Branding Header
        self.header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=25, pady=(30, 20), sticky="ew")
        
        self.logo_label = ctk.CTkLabel(self.header_frame, text="SmartPark", font=ctk.CTkFont(family="Helvetica", size=26, weight="bold"), text_color="#FFFFFF")
        self.logo_label.pack(anchor="w")
        
        self.subtitle_label = ctk.CTkLabel(self.header_frame, text="AI VISION SYSTEM", font=ctk.CTkFont(family="Helvetica", size=10, weight="bold"), text_color="#3B82F6")
        self.subtitle_label.pack(anchor="w")

        # Separator
        separator1 = ctk.CTkFrame(self.sidebar, height=1, fg_color="#2A2D3E")
        separator1.grid(row=1, column=0, sticky="ew", padx=25, pady=(0, 20))

        # 2. Media Controls Group (Cards)
        self.media_group = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.media_group.grid(row=2, column=0, padx=25, pady=(0, 20), sticky="ew")
        
        ctk.CTkLabel(self.media_group, text="DATA SOURCE", font=ctk.CTkFont(size=11, weight="bold"), text_color="#64748B").pack(anchor="w", pady=(0, 10))
        
        self.btn_load_vid = ctk.CTkButton(self.media_group, text="▶ Live Video Stream", font=ctk.CTkFont(weight="bold"), height=40, command=self.load_video, fg_color="#2563EB", hover_color="#1D4ED8")
        self.btn_load_vid.pack(fill="x", pady=(0, 8))

        self.btn_load_img = ctk.CTkButton(self.media_group, text="📷 Analyze Static Image", font=ctk.CTkFont(weight="bold"), height=40, command=self.load_image, fg_color="#334155", hover_color="#475569")
        self.btn_load_img.pack(fill="x", pady=(0, 8))

        self.btn_stop_vid = ctk.CTkButton(self.media_group, text="⏹ Stop Processing", font=ctk.CTkFont(weight="bold"), height=40, command=self.stop_video, state="disabled", fg_color="transparent", border_width=1, border_color="#334155", text_color="#94A3B8")
        self.btn_stop_vid.pack(fill="x")
        
        # Separator
        separator2 = ctk.CTkFrame(self.sidebar, height=1, fg_color="#2A2D3E")
        separator2.grid(row=3, column=0, sticky="ew", padx=25, pady=20)

        # 3. Configuration Group
        self.config_group = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.config_group.grid(row=4, column=0, padx=25, sticky="nsew")
        
        ctk.CTkLabel(self.config_group, text="CONFIGURATION", font=ctk.CTkFont(size=11, weight="bold"), text_color="#64748B").pack(anchor="w", pady=(0, 15))
        
        # Conf Slider
        self.conf_frame = ctk.CTkFrame(self.config_group, fg_color="transparent")
        self.conf_frame.pack(fill="x", pady=(0, 15))
        self.conf_label = ctk.CTkLabel(self.conf_frame, text="Confidence Threshold", font=ctk.CTkFont(size=12))
        self.conf_label.pack(anchor="w")
        self.conf_var = tk.DoubleVar(value=0.20)
        self.conf_slider = ctk.CTkSlider(self.conf_frame, from_=0.01, to=1.0, variable=self.conf_var, button_color="#3B82F6", progress_color="#3B82F6")
        self.conf_slider.pack(fill="x", pady=(5, 0))

        # Mode Menu
        self.mode_label = ctk.CTkLabel(self.config_group, text="Detection Mode", font=ctk.CTkFont(size=12))
        self.mode_label.pack(anchor="w")
        self.mode_var = ctk.StringVar(value="AI Counting")
        self.mode_menu = ctk.CTkOptionMenu(self.config_group, variable=self.mode_var, values=["Manual", "Geometric Math", "AI Counting"], fg_color="#1E293B", button_color="#334155")
        self.mode_menu.pack(fill="x", pady=(5, 15))
        
        # Spot Entry
        self.spots_label = ctk.CTkLabel(self.config_group, text="Manual Spot Capacity", font=ctk.CTkFont(size=12))
        self.spots_label.pack(anchor="w")
        self.total_spots_var = tk.StringVar(value="15")
        self.total_spots_entry = ctk.CTkEntry(self.config_group, textvariable=self.total_spots_var, fg_color="#1E293B", border_color="#334155")
        self.total_spots_entry.pack(fill="x", pady=(5, 15))

        # Imgsz Menu
        self.imgsz_label = ctk.CTkLabel(self.config_group, text="Inference Resolution", font=ctk.CTkFont(size=12))
        self.imgsz_label.pack(anchor="w")
        self.imgsz_var = ctk.StringVar(value="1920")
        self.imgsz_menu = ctk.CTkOptionMenu(self.config_group, variable=self.imgsz_var, values=["640", "1024", "1280", "1920", "2560", "3200"], fg_color="#1E293B", button_color="#334155")
        self.imgsz_menu.pack(fill="x", pady=(5, 0))

        # 4. Live Analytics Dashboard (Bottom of sidebar)
        self.analytics_group = ctk.CTkFrame(self.sidebar, fg_color="#1E293B", corner_radius=12)
        self.analytics_group.grid(row=5, column=0, padx=20, pady=25, sticky="ew")
        
        # Occupied Card (Red)
        self.occ_card = ctk.CTkFrame(self.analytics_group, fg_color="transparent")
        self.occ_card.pack(side="left", expand=True, fill="both", pady=15, padx=10)
        ctk.CTkLabel(self.occ_card, text="OCCUPIED", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94A3B8").pack()
        self.lbl_occupied = ctk.CTkLabel(self.occ_card, text="0", font=ctk.CTkFont(size=32, weight="bold"), text_color="#EF4444")
        self.lbl_occupied.pack()

        # Divider line
        ctk.CTkFrame(self.analytics_group, width=1, fg_color="#334155").pack(side="left", fill="y", pady=15)

        # Vacant Card (Green)
        self.vac_card = ctk.CTkFrame(self.analytics_group, fg_color="transparent")
        self.vac_card.pack(side="left", expand=True, fill="both", pady=15, padx=10)
        ctk.CTkLabel(self.vac_card, text="VACANT", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94A3B8").pack()
        self.lbl_vacant = ctk.CTkLabel(self.vac_card, text="15", font=ctk.CTkFont(size=32, weight="bold"), text_color="#10B981")
        self.lbl_vacant.pack()

        # Total Card (Blue - Full width below)
        self.tot_card = ctk.CTkFrame(self.analytics_group, fg_color="transparent")
        self.tot_card.pack(side="bottom", fill="x", pady=(0, 15))
        self.lbl_total = ctk.CTkLabel(self.tot_card, text="Total Capacity: 15", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3B82F6")
        self.lbl_total.pack()

        # --- Main Display Area ---
        self.display_container = ctk.CTkFrame(self.root, fg_color="#0F111A", corner_radius=0)
        self.display_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.display_container.grid_rowconfigure(0, weight=1)
        self.display_container.grid_columnconfigure(0, weight=1)

        # Inner rounded frame for the canvas
        self.canvas_frame = ctk.CTkFrame(self.display_container, fg_color="#18181B", corner_radius=15, border_width=1, border_color="#2A2D3E")
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")

        # Tkinter Canvas wrapped beautifully
        self.canvas = tk.Canvas(self.canvas_frame, bg="#18181B", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Placeholder text
        self.canvas.create_text(
            400, 300, 
            text="AWAITING VIDEO FEED...", 
            fill="#334155", 
            font=("Helvetica", 20, "bold"),
            tags="placeholder"
        )
        
        # Store current photo to prevent garbage collection and current raw frame to redraw on resize
        self.current_photo = None
        self.current_frame = None

    def on_canvas_resize(self, event):
        if self.current_frame is not None:
            self.display_image(self.current_frame)
        else:
            self.canvas.delete("all")
            x_center = self.canvas.winfo_width() // 2
            y_center = self.canvas.winfo_height() // 2
            self.canvas.create_text(
                x_center, y_center, 
                text="AWAITING VIDEO FEED...", 
                fill="#334155", 
                font=("Helvetica", 20, "bold"),
                tags="placeholder"
            )

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
                self.btn_load_img.configure(state="disabled", fg_color="transparent", text_color="gray")
                self.btn_load_vid.configure(state="disabled", fg_color="transparent", text_color="gray")
                self.btn_stop_vid.configure(state="normal", fg_color="#EF4444", text_color="white", border_color="#EF4444")
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
            
        self.btn_load_img.configure(state="normal", fg_color="#334155", text_color="white")
        self.btn_load_vid.configure(state="normal", fg_color="#2563EB", text_color="white")
        self.btn_stop_vid.configure(state="disabled", fg_color="transparent", text_color="#94A3B8", border_color="#334155")

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
                        if class_id == 0 and float(class_score[class_id]) > conf_thresh:
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
            indices = cv2.dnn.NMSBoxes(all_car_boxes, all_car_scores, conf_thresh, 0.50)
            
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                box = all_car_boxes[idx]
                final_cars.append(box)
                left, top, w, h = box[0], box[1], box[2], box[3]
                conf = all_car_scores[idx]
                
                cars_count += 1
                
                # Premium Bounding Box UI - Subtle Red Overlay with sharp borders
                overlay = img.copy()
                cv2.rectangle(overlay, (left, top), (left + w, top + h), (36, 36, 239), -1)  # BGR Red
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
                cv2.rectangle(img, (left, top), (left + w, top + h), (36, 36, 239), 2)
                
                # Sleek Label
                label = f"car {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)[0]
                cv2.rectangle(img, (left, top - 22), (left + text_size[0] + 10, top), (36, 36, 239), -1)
                cv2.putText(img, label, (left + 5, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

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
                
                # Premium Bounding Box UI - Subtle Green Overlay with sharp borders
                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (129, 185, 16), -1) # BGR Green
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
                cv2.rectangle(img, (x1, y1), (x2, y2), (129, 185, 16), 2)
                
                # Sleek Label
                label = f"{cls_name} {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)[0]
                cv2.rectangle(img, (x1, y1 - 22), (x1 + text_size[0] + 10, y1), (129, 185, 16), -1)
                cv2.putText(img, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

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
            
        self.lbl_total.configure(text=f"Total Capacity: {total}")
        self.lbl_occupied.configure(text=f"{cars}")
        self.lbl_vacant.configure(text=f"{empty}")

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
    root = ctk.CTk()
    app = ParkingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
