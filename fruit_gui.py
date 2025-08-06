import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time

# Load model and class names
model = load_model("fruit_model.h5")
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Image size used during training
IMG_SIZE = (100, 100)

# Start webcam
cap = cv2.VideoCapture(0)
print("⏳ Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a rectangle for user to place fruit
    x, y, w, h = 200, 100, 224, 224  # ROI region
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = frame[y:y+h, x:x+w]

    # Preprocess ROI
    try:
        roi_resized = cv2.resize(roi, IMG_SIZE)
        img_array = image.img_to_array(roi_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict top 3 classes
        predictions = model.predict(img_array, verbose=0)[0]
        top_indices = predictions.argsort()[-3:][::-1]
        top_classes = [(class_names[i], predictions[i]) for i in top_indices]

        # Display top-3 predictions on frame
        for i, (label, prob) in enumerate(top_classes):
            text = f"{label}: {prob*100:.2f}%"
            cv2.putText(frame, text, (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    except:
        cv2.putText(frame, "Detecting...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("🍎 Fruit Recognition - Press 'q' to Quit", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
