"""
Fruit Recognition App - Configuration
Centralized settings for the application.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Configuration
MODEL_PATH = os.path.join(BASE_DIR, "fruit_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")

# Image Configuration
IMG_WIDTH = 100
IMG_HEIGHT = 100
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# Camera Configuration
CAMERA_INDEX = 0
ROI_X = 200
ROI_Y = 100
ROI_WIDTH = 224
ROI_HEIGHT = 224

# Prediction Configuration
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display
TOP_K_PREDICTIONS = 5       # Number of top predictions to show

# Training Configuration
TRAIN_DIR = os.path.join(BASE_DIR, "fruits-360", "Training")
VAL_DIR = os.path.join(BASE_DIR, "fruits-360", "Test")
BATCH_SIZE = 32
EPOCHS = 15

# GUI Colors (Dark Theme)
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
