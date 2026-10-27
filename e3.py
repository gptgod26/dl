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
base_model = ResNet50(
    weights="imagenet",
    include_top=False
)
base_model.trainable = False
feature_extractor = Model(
    inputs=base_model.input,
    outputs=GlobalAveragePooling2D()(base_model.output)
)




def extract_features(paths):
    features = []
    for path in paths:
        image = load_preprocess_image(path)
        feature = feature_extractor.predict(image, verbose=0)
        features.append(feature[0])
    return np.array(features, dtype=np.float32)
X_train_features = extract_features(X_train_paths)
X_test_features = extract_features(X_test_paths)
print("Training features:", X_train_features.shape)
print("Testing features:", X_test_features.shape)




num_classes = len(class_names)
classifier = tf.keras.Sequential([
    Input(shape=(2048,)),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(num_classes, activation="softmax")
])
classifier.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = classifier.fit(
    X_train_features,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



# Predict test classes.
probabilities = classifier.predict(X_test_features, verbose=0)
y_pred = np.argmax(probabilities, axis=1)
print("Actual:", y_test[:10])
print("Predicted:", y_pred[:10])
# Calculate evaluation metrics.
print("Accuracy:",
      accuracy_score(y_test, y_pred))
print("Precision:",
      precision_score(
          y_test, y_pred,
          average="weighted", zero_division=0
      ))
print("Recall:",
      recall_score(
          y_test, y_pred,
          average="weighted", zero_division=0
      ))
print("F1 Score:",
      f1_score(
          y_test, y_pred,
          average="weighted", zero_division=0
      ))



# Display confusion matrix.
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
display.plot(xticks_rotation=90)
plt.tight_layout()
plt.show()

