import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import kagglehub

# --- 1. Download and Prepare Dataset ---

print("Downloading dataset from KaggleHub...")
raw_path = kagglehub.dataset_download("gti-upm/leapgestrecog")

# Print path to check
print("Dataset downloaded to:", raw_path)

# If 'leapGestRecog/leapGestRecog' exists, fix path
if os.path.exists(os.path.join(raw_path, "leapGestRecog")):
    DATA_DIR = os.path.join(raw_path, "leapGestRecog")
else:
    DATA_DIR = raw_path
IMG_SIZE = (128, 128)

image_paths = []
labels = []

# Traverse the dataset directory
for subject_folder in os.listdir(DATA_DIR):
    subject_path = os.path.join(DATA_DIR, subject_folder)
    if not os.path.isdir(subject_path):
        continue

    for gesture_folder in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        # Check for valid gesture folder format
        if '_' not in gesture_folder or not gesture_folder.split('_')[0].isdigit():
            print(f"Skipping invalid folder: {gesture_folder}")
            continue

        label = int(gesture_folder.split('_')[0]) - 1

        for img_file in os.listdir(gesture_path):
            if img_file.endswith('.png'):
                image_paths.append(os.path.join(gesture_path, img_file))
                labels.append(label)

print(f"Found {len(image_paths)} images.")

# Load and preprocess images
images = []
valid_labels = []

for i, path in enumerate(image_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        valid_labels.append(labels[i])
    else:
        print(f"Warning: Could not read image at {path}. Skipping.")

images = np.array(images, dtype='float32') / 255.0
labels = np.array(valid_labels)
images = np.expand_dims(images, axis=-1)
labels = to_categorical(labels, num_classes=10)

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- 2. Build Model ---

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

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Train Model ---

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2 
)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=20,
                    validation_data=(X_test, y_test))

model.save('hand_gesture_model.h5')
print("\nModel saved as hand_gesture_model.h5")

# --- 4. Evaluate Model ---

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# --- 5. Predict Sample Image ---

gesture_names = {
    0: 'palm', 1: 'l', 2: 'fist', 3: 'fist_moved', 4: 'thumb',
    5: 'index', 6: 'ok', 7: 'palm_moved', 8: 'c', 9: 'down'
}

def predict_gesture(image_path, model_to_use):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        img_resized = cv2.resize(img, IMG_SIZE)
        img_normalized = img_resized / 255.0
        img_reshaped = np.reshape(img_normalized, (1, IMG_SIZE[0], IMG_SIZE[1], 1))

        prediction = model_to_use.predict(img_reshaped)
        predicted_class_index = np.argmax(prediction)

        return gesture_names.get(predicted_class_index, "Unknown")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

# Example usage
test_image_path = os.path.join(DATA_DIR, '00', '01_palm', 'frame_00_01_0001.png')
if os.path.exists(test_image_path):
    loaded_model = load_model('hand_gesture_model.h5')
    predicted_gesture = predict_gesture(test_image_path, loaded_model)
    if predicted_gesture:
        print(f"\nThe predicted gesture for '{test_image_path}' is: {predicted_gesture}")
else:
    print(f"\nTest image not found at '{test_image_path}'. Please check the path.")
