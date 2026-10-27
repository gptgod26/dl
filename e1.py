import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import (
    preprocess_input, decode_predictions
)
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    Embedding
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

DATASET_DIR = "imagenet_subset"
IMG_SIZE = (224, 224)


# ============================================================
# LOAD LOCAL DATASET AND LABELS
# ============================================================

image_paths = []
labels = []

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_dir = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_dir):
        continue

    for file in os.listdir(class_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(class_dir, file))
            labels.append(class_name)

if not image_paths:
    raise ValueError(
        f"No images found in '{DATASET_DIR}'. "
        "Expected one subfolder per class, containing JPG/PNG images."
    )

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
class_names = encoder.classes_

print("Images:", len(image_paths))
print("Classes:", class_names)

X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    image_paths,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


def load_preprocess_image(path):
    """Load one local image and preprocess it for ResNet50."""
    image = load_img(path, target_size=IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


# Display an original image
image = load_img(X_train_paths[0], target_size=IMG_SIZE)
plt.imshow(image)
plt.title(class_names[y_train[0]])
plt.axis("off")
plt.show()
model = ResNet50(
    weights="imagenet",
    include_top=True
)

image = load_preprocess_image(X_test_paths[0])
predictions = model.predict(image, verbose=0)

print("Prediction shape:", predictions.shape)

top5 = decode_predictions(predictions, top=5)[0]

for _, name, probability in top5:
    print(name, f"{probability * 100:.2f}%")
