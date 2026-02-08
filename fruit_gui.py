"""
Fruit Recognition App - Modern GUI
A real-time fruit recognition application with a modern dark-themed interface.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

# TensorFlow imports with warning suppression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import configuration
try:
    from config import (
        MODEL_PATH, CLASS_NAMES_PATH, IMG_SIZE,
        CAMERA_INDEX, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT,
        TOP_K_PREDICTIONS, COLORS
    )
except ImportError:
    # Fallback defaults
    MODEL_PATH = "fruit_model.h5"
    CLASS_NAMES_PATH = "class_names.txt"
    IMG_SIZE = (100, 100)
    CAMERA_INDEX = 0
    ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = 200, 100, 224, 224
    TOP_K_PREDICTIONS = 5
    COLORS = {
        "bg_dark": "#1a1a2e",
        "bg_medium": "#16213e",
        "bg_light": "#0f3460",
        "accent": "#e94560",
        "success": "#00d4aa",
        "warning": "#ffa726",
        "text_primary": "#ffffff",
        "text_secondary": "#b0b0b0",
    }


class FruitRecognitionApp:
    """Modern GUI application for real-time fruit recognition."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fruit Recognition App")
        self.root.configure(bg=COLORS["bg_dark"])
        self.root.resizable(False, False)
        
        # Load model and class names
        print("Loading model...")
        self.model = load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(self.class_names)} classes")
        
        # Camera variables
        self.cap = None
        self.running = False
        self.current_predictions = []
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Setup UI
        self._setup_ui()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _setup_ui(self):
        """Setup the modern UI components."""
        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS["bg_dark"])
        main_frame.pack(padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Fruit Recognition",
            font=("Segoe UI", 24, "bold"),
            fg=COLORS["accent"],
            bg=COLORS["bg_dark"]
        )
        title_label.pack(pady=(0, 15))
        
        # Content frame (camera + predictions side by side)
        content_frame = tk.Frame(main_frame, bg=COLORS["bg_dark"])
        content_frame.pack()
        
        # Left: Camera frame
        camera_container = tk.Frame(content_frame, bg=COLORS["bg_medium"], padx=3, pady=3)
        camera_container.pack(side=tk.LEFT, padx=(0, 15))
        
        self.camera_label = tk.Label(camera_container, bg=COLORS["bg_dark"])
        self.camera_label.pack()
        
        # Right: Predictions panel
        predictions_frame = tk.Frame(
            content_frame,
            bg=COLORS["bg_medium"],
            width=300,
            height=400
        )
        predictions_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        predictions_frame.pack_propagate(False)
        
        # Predictions title
        pred_title = tk.Label(
            predictions_frame,
            text="Top Predictions",
            font=("Segoe UI", 14, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_medium"]
        )
        pred_title.pack(pady=15)
        
        # Predictions container
        self.predictions_container = tk.Frame(predictions_frame, bg=COLORS["bg_medium"])
        self.predictions_container.pack(fill=tk.BOTH, expand=True, padx=15)
        
        # Create prediction bars
        self.prediction_bars = []
        for i in range(TOP_K_PREDICTIONS):
            bar_frame = tk.Frame(self.predictions_container, bg=COLORS["bg_medium"])
            bar_frame.pack(fill=tk.X, pady=8)
            
            # Class name label
            name_label = tk.Label(
                bar_frame,
                text="Waiting...",
                font=("Segoe UI", 11),
                fg=COLORS["text_primary"],
                bg=COLORS["bg_medium"],
                anchor="w"
            )
            name_label.pack(fill=tk.X)
            
            # Progress bar background
            bar_bg = tk.Frame(bar_frame, bg=COLORS["bg_light"], height=20)
            bar_bg.pack(fill=tk.X, pady=(3, 0))
            
            # Progress bar fill
            bar_fill = tk.Frame(bar_bg, bg=COLORS["success"], height=20, width=0)
            bar_fill.place(x=0, y=0, height=20)
            
            # Confidence label
            conf_label = tk.Label(
                bar_bg,
                text="0%",
                font=("Segoe UI", 9, "bold"),
                fg=COLORS["text_primary"],
                bg=COLORS["bg_light"]
            )
            conf_label.place(relx=0.5, rely=0.5, anchor="center")
            
            self.prediction_bars.append({
                "name": name_label,
                "bar_bg": bar_bg,
                "bar_fill": bar_fill,
                "conf": conf_label
            })
        
        # Bottom: Status bar
        status_frame = tk.Frame(main_frame, bg=COLORS["bg_dark"])
        status_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.fps_label = tk.Label(
            status_frame,
            text="FPS: --",
            font=("Segoe UI", 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_dark"]
        )
        self.fps_label.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(
            status_frame,
            text="Press Q to quit",
            font=("Segoe UI", 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_dark"]
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Instructions
        instruction_label = tk.Label(
            main_frame,
            text="Place fruit in the green box for recognition",
            font=("Segoe UI", 10, "italic"),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_dark"]
        )
        instruction_label.pack(pady=(10, 0))
        
    def _start_camera(self):
        """Initialize and start the camera feed."""
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.status_label.config(text="Error: Cannot open camera", fg=COLORS["accent"])
            return False
        
        self.running = True
        self.status_label.config(text="Camera active - Press Q to quit")
        return True
        
    def _update_frame(self):
        """Capture and process camera frames."""
        if not self.running or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self._update_frame)
            return
        
        # Draw ROI rectangle
        cv2.rectangle(
            frame,
            (ROI_X, ROI_Y),
            (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT),
            (0, 255, 0), 3
        )
        
        # Extract and process ROI
        roi = frame[ROI_Y:ROI_Y + ROI_HEIGHT, ROI_X:ROI_X + ROI_WIDTH]
        
        final_input_rgb = None # To store debug image
        
        try:
            # Preprocess for prediction
            roi_resized = cv2.resize(roi, IMG_SIZE)
            
            # --- Background Suppression ---
            # Isolate fruit by saturation
            hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Increase threshold to 50 to filter out skin/dull objects
            _, mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)
            
            # --- Auto-Zoom / Centering ---
            # Find contours to crop the fruit
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                # Check if object is significant
                if w > 10 and h > 10:
                    pad = 10
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end = min(roi_resized.shape[1], x + w + pad)
                    y_end = min(roi_resized.shape[0], y + h + pad)
                    
                    roi_processing = roi_resized[y_start:y_end, x_start:x_end]
                    mask_processing = mask[y_start:y_end, x_start:x_end]
                else:
                    roi_processing = roi_resized
                    mask_processing = mask
            else:
                 roi_processing = roi_resized
                 mask_processing = mask

            # Create white background
            white_bg = np.ones_like(roi_processing) * 255
            
            # Combine
            mask_3ch = cv2.merge([mask_processing, mask_processing, mask_processing])
            foreground = cv2.bitwise_and(roi_processing, mask_3ch)
            background = cv2.bitwise_and(white_bg, cv2.bitwise_not(mask_3ch))
            final_roi = cv2.add(foreground, background)
            
            # Resize to model input
            final_input = cv2.resize(final_roi, IMG_SIZE)
            
            # Convert to RGB for model AND display
            final_input_rgb = cv2.cvtColor(final_input, cv2.COLOR_BGR2RGB)
            
            img_array = final_input_rgb.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)[0]
            top_indices = predictions.argsort()[-TOP_K_PREDICTIONS:][::-1]
            self.current_predictions = [
                (self.class_names[i], float(predictions[i]))
                for i in top_indices
            ]
            
            self._update_predictions()
            
        except Exception as e:
            print(f"Error in processing: {e}")
            import traceback
            traceback.print_exc()
            pass
        
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        
        # Convert main frame for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use corner of frame to show debug view if available
        if final_input_rgb is not None:
             # Resize debug view to larger size for visibility, e.g. 150x150
             debug_view = cv2.resize(final_input_rgb, (150, 150))
             # Overlay on top-right of the main frame
             h, w, _ = frame_rgb.shape
             dh, dw, _ = debug_view.shape
             # Ensure it fits
             if h > dh and w > dw:
                 frame_rgb[10:10+dh, w-10-dw:w-10] = debug_view
                 cv2.rectangle(frame_rgb, (w-10-dw, 10), (w-10, 10+dh), (255, 255, 255), 2)
                 cv2.putText(frame_rgb, "Model Input", (w-10-dw, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)
        
        # Schedule next frame
        self.root.after(10, self._update_frame)
        
    def _update_predictions(self):
        """Update the prediction display bars."""
        for i, bar in enumerate(self.prediction_bars):
            if i < len(self.current_predictions):
                name, conf = self.current_predictions[i]
                bar["name"].config(text=name)
                bar["conf"].config(text=f"{conf*100:.1f}%")
                
                # Update bar width (max 270 pixels)
                bar_width = int(conf * 270)
                bar["bar_fill"].place(x=0, y=0, width=bar_width, height=20)
                
                # Color based on confidence
                if conf >= 0.7:
                    bar["bar_fill"].config(bg=COLORS["success"])
                elif conf >= 0.4:
                    bar["bar_fill"].config(bg=COLORS["warning"])
                else:
                    bar["bar_fill"].config(bg=COLORS["accent"])
            else:
                bar["name"].config(text="--")
                bar["conf"].config(text="--")
                bar["bar_fill"].place(x=0, y=0, width=0, height=20)
                
    def _on_close(self):
        """Handle window close event."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()
        
    def _check_key(self, event):
        """Handle keyboard events."""
        if event.char.lower() == 'q':
            self._on_close()
            
    def run(self):
        """Start the application."""
        if self._start_camera():
            self.root.bind("<Key>", self._check_key)
            self._update_frame()
            self.root.mainloop()
        else:
            print("Failed to start camera")
            self.root.destroy()


def main():
    """Entry point for the application."""
    print("Starting Fruit Recognition App...")
    print("-" * 40)
    app = FruitRecognitionApp()
    app.run()
    print("Application closed.")


if __name__ == "__main__":
    main()
