
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

IMG_SIZE = 224
DATASET_DIR = "eye_dataset"  # Folder with subfolders: open, closed, yawn, no_yawn

# Step 1: Load and label data
def load_images_and_labels():
    data = []
    labels = []

    for label_type in ["open", "closed", "yawn", "no_yawn"]:
        folder = os.path.join(DATASET_DIR, label_type)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            # Eye state label: 1=open, 0=closed
            eye_label = 1 if label_type == "open" else 0 if label_type == "closed" else None
            # Mouth state label: 1=yawn, 0=no_yawn
            mouth_label = 1 if label_type == "yawn" else 0 if label_type == "no_yawn" else None

            if eye_label is not None:
                data.append(img)
                labels.append((eye_label, 0))  # No mouth label

            elif mouth_label is not None:
                data.append(img)
                labels.append((0, mouth_label))  # No eye label

    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="float32")
    return data, labels

X, Y = load_images_and_labels()

# Step 2: Split into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 3: Build dual-output model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

eye_output = Dense(1, activation="sigmoid", name="eye_output")(x)
mouth_output = Dense(1, activation="sigmoid", name="mouth_output")(x)

model = Model(inputs=base_model.input, outputs=[eye_output, mouth_output])

model.compile(optimizer=Adam(1e-4),
              loss={"eye_output": "binary_crossentropy", "mouth_output": "binary_crossentropy"},
              metrics={"eye_output": "accuracy", "mouth_output": "accuracy"})

# Step 4: Train
model.fit(X_train, {"eye_output": Y_train[:, 0], "mouth_output": Y_train[:, 1]},
          validation_data=(X_val, {"eye_output": Y_val[:, 0], "mouth_output": Y_val[:, 1]}),
          epochs=5,
          batch_size=32)

# Step 5: Save
model.save("eye_mouth_state_model.h5")
print("✅ Model saved as eye_mouth_state_model.h5")
