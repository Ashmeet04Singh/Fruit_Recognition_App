import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and class labels
model = load_model("fruit_model.h5")
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Set image size (must match training size)
IMG_SIZE = 100

# Initialize webcam
cap = cv2.VideoCapture(0)

print("📷 Press 'q' to quit the camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a rectangle as a guide for user
    x, y, w, h = 200, 100, 224, 224
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = frame[y:y+h, x:x+w]

    # Preprocess image
    try:
        img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display result
        cv2.putText(frame, f"{predicted_class} ({confidence:.2f}%)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
    except:
        cv2.putText(frame, "Align fruit in box!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Fruit Classifier", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
