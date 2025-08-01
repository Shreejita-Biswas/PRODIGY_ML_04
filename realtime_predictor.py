import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

# --- 1. Download Dataset from KaggleHub ---
base_path = kagglehub.dataset_download("gti-upm/leapgestrecog")
DATA_DIR = os.path.join(base_path, "leapGestRecog") 
IMG_SIZE = (128, 128)

# --- 2. Load and Preprocess the Dataset ---
image_paths, labels = [], []

from utils.model_utils import build_gesture_model
model = build_gesture_model()


for subject_folder in os.listdir(DATA_DIR):
    subject_path = os.path.join(DATA_DIR, subject_folder)
    if not os.path.isdir(subject_path):
        continue

    for gesture_folder in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        label = int(gesture_folder.split('_')[0]) - 1
        for img_file in os.listdir(gesture_path):
            if img_file.endswith('.png'):
                image_paths.append(os.path.join(gesture_path, img_file))
                labels.append(label)

# Load and preprocess images
images = []
valid_labels = []
for i, path in enumerate(image_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        valid_labels.append(labels[i])

images = np.array(images, dtype='float32') / 255.0
labels = np.array(valid_labels)
images = np.expand_dims(images, axis=-1)
labels = to_categorical(labels, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- 3. Define CNN Model ---
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. Train the Model ---
print("\n📦 Training the model (this may take a few minutes)...")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2
)
datagen.fit(X_train)

model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=10,
          validation_data=(X_test, y_test))

# --- 5. Real-Time Gesture Recognition ---
gesture_names = {
    0: 'palm', 1: 'l', 2: 'fist', 3: 'fist_moved', 4: 'thumb',
    5: 'index', 6: 'ok', 7: 'palm_moved', 8: 'c', 9: 'down'
}

print("\n🎥 Starting webcam... (press 'q' to quit)")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    roi = frame[50:350, 50:350]
    cv2.rectangle(frame, (50, 50), (350, 350), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_input = np.reshape(img_normalized, (1, IMG_SIZE[0], IMG_SIZE[1], 1))

    prediction = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(prediction)
    gesture = gesture_names.get(predicted_class, "Recognizing...")

    cv2.putText(frame, f"Prediction: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
