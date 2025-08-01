# 🤖 Hand Gesture Recognition using CNN

This project implements a Convolutional Neural Network (CNN) model to recognize hand gestures using the **LeapGestRecog** dataset. It includes image preprocessing, data augmentation, training, model evaluation, and a real-time prediction pipeline.

![Model Architecture](https://img.shields.io/badge/Model-CNN-blue) ![Language](https://img.shields.io/badge/Python-3.8%2B-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📁 Dataset

- **Source**: [Kaggle - LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Size**: ~20,000 grayscale PNG images
- **Classes**: 10 hand gestures
  - `palm`, `l`, `fist`, `fist_moved`, `thumb`, `index`, `ok`, `palm_moved`, `c`, `down`

The dataset is automatically downloaded using `kagglehub`.

---

## 🧠 Model Overview

The CNN model architecture:


Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Conv2D(256) → MaxPool
→ Flatten → Dense(512) → Dropout(0.5) → Dense(10, softmax)
## 📦 Folder Structure

├── app.py                  # Main training & evaluation script
├── realtime_predictor.py   # Real-time webcam prediction (to be added)
├── hand_gesture_model.h5   # Saved trained model
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
🚀 Getting Started
### 1. Clone the Repository

git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
### 2. Install Dependencies
Make sure you have Python 3.8+ installed.

pip install -r requirements.txt
Note: You'll need to authenticate with Kaggle to use kagglehub. Place your Kaggle API token in the right location (~/.kaggle/kaggle.json).

### 3. Train the Model
Run the app.py script to:

Download the dataset

Preprocess and augment the data

Train the CNN model

Save the trained model

Predict a test sample

python app.py
## 🎯 Results
Test Accuracy: ~98–99% (may vary slightly)

Training Time: ~5–10 mins on CPU (faster with GPU)

📈 Accuracy & Loss Curves
The script also plots training vs validation accuracy and loss:


## 🎥 Real-time Prediction (Coming Soon)
In realtime_predictor.py, you will be able to:

Open a webcam feed using OpenCV

Detect hand gesture using trained model

Display predicted gesture in real-time

Stay tuned for updates!

## 🧠 Labels
python
Copy
Edit
gesture_names = {
    0: 'palm', 1: 'l', 2: 'fist', 3: 'fist_moved', 4: 'thumb',
    5: 'index', 6: 'ok', 7: 'palm_moved', 8: 'c', 9: 'down'
}
## 📚 Tech Stack
Python 🐍

TensorFlow / Keras

OpenCV

NumPy, Matplotlib

Scikit-learn

kagglehub

## 📄 License
This project is licensed under the MIT License.
