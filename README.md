# Fruit Recognition App

A deep learning-powered fruit and vegetable recognition application using TensorFlow/Keras CNN trained on the **Fruits-360 dataset**. Features real-time webcam detection with a modern GUI interface.

## Features

- **207 Classes**: Recognizes a wide variety of fruits and vegetables
- **Real-time Detection**: Live webcam feed with instant predictions
- **Modern GUI**: Dark-themed Tkinter interface with confidence bars
- **Top-5 Predictions**: Shows multiple possible matches with confidence scores
- **Easy to Use**: Place item in the detection zone and see results

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time detection)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Fruit_Recognition_App.git
   cd Fruit_Recognition_App
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Real-time Detection (GUI)
```bash
python fruit_gui.py
```
- Position a fruit/vegetable in the green detection box
- View predictions with confidence scores
- Press **Q** or close the window to exit

### Test on Static Image
```bash
python test_predict.py path/to/your/image.jpg
```

### Train Your Own Model
```bash
python train_model.py
```
> Note: Requires the Fruits-360 dataset in the `fruits-360/` folder

## Model Architecture

The CNN model uses the following architecture:

| Layer | Type | Output Shape |
|-------|------|--------------|
| 1 | Conv2D (32 filters) + ReLU | (98, 98, 32) |
| 2 | MaxPooling2D | (49, 49, 32) |
| 3 | Conv2D (64 filters) + ReLU | (47, 47, 64) |
| 4 | MaxPooling2D | (23, 23, 64) |
| 5 | Conv2D (128 filters) + ReLU | (21, 21, 128) |
| 6 | MaxPooling2D | (10, 10, 128) |
| 7 | Flatten | (12800) |
| 8 | Dense (256) + ReLU + Dropout | (256) |
| 9 | Dense (207) + Softmax | (207) |

## Project Structure

```
Fruit_Recognition_App/
├── fruit_gui.py        # Main GUI application
├── train_model.py      # Model training script
├── test_predict.py     # Static image prediction
├── config.py           # Configuration settings
├── fruit_model.h5      # Trained model weights
├── class_names.txt     # List of 207 class names
├── requirements.txt    # Python dependencies
└── fruits-360/         # Dataset (not included)
    ├── Training/
    └── Test/
```

## Dataset

This project uses the [Fruits-360 dataset](https://www.kaggle.com/moltean/fruits) which contains:
- **90,000+** images
- **207** classes of fruits and vegetables
- **100x100** pixel images with white background

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Fruits-360 Dataset](https://www.kaggle.com/moltean/fruits) by Horea Muresan and Mihai Oltean
- TensorFlow/Keras team for the deep learning framework
