import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths (assuming folder is now named "fruits-360")
train_dir = "fruits-360/Training"
val_dir = "fruits-360/Test"

# Image size
img_width, img_height = 100, 100

# Preprocessing
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode="categorical"
)

# Number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of fruit classes: {num_classes}")
print("Class Indices:", train_generator.class_indices)

# Save class names to file
sorted_classes = sorted(train_generator.class_indices.items(), key=lambda x: x[1])
class_names = [name for name, index in sorted_classes]
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save model
model.save("fruit_model.h5")
print("✅ Model saved as fruit_model.h5")
print("✅ Class names saved to class_names.txt")
