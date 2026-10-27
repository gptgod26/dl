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
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
# Load class labels
with open("labels.json") as f:
    data = json.load(f)
label_map = {item[1]: item[2] for item in data}
# Load images
dataset = tf.keras.utils.image_dataset_from_directory(
    "imagenet_folder",
    image_size=(224, 224),
    batch_size=20,
    shuffle=False
)
classes = [label_map.get(name, name) for name in dataset.class_names]



# Convert images to arrays
images = np.concatenate([x.numpy() for x, y in dataset.take(10)])
labels = np.concatenate([y.numpy() for x, y in dataset.take(10)])
# Train-test split
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
# CNN feature extraction
cnn = VGG16(weights="imagenet", include_top=False, pooling="avg")
train_features = cnn.predict(preprocess_input(train_images))
test_features = cnn.predict(preprocess_input(test_images))
print("CNN Feature Vector:", train_features.shape)



# Reshape for LSTM
X = train_features.reshape(train_features.shape[0], 1, train_features.shape[1])
# LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, 512)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(classes), activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(X, train_labels, epochs=10)




# Prediction
test_X = test_features.reshape(test_features.shape[0], 1, test_features.shape[1])
pred = model.predict(test_X)
for i in range(min(5, len(pred))):
    predicted_class = classes[np.argmax(pred[i])]
    expected = "A photo of a " + classes[test_labels[i]] + "."
    generated = "A photo of a " + predicted_class + "."
    print("Expected :", expected)
    print("Generated:", generated)
    print()
