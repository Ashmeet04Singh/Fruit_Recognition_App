from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Load model and class names
model = load_model("fruit_model.h5")
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load and predict image
def predict_image(img_path):
    image = load_img(img_path, target_size=(100, 100))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    class_index = np.argmax(pred)
    print(f"✅ Prediction: {class_names[class_index]}")

# Test on saved images
predict_image("test_apple.jpg")
