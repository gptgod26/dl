import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report

# Load only a small dataset
train = tf.keras.utils.image_dataset_from_directory(
    archive + "/imagenette2-160/train",
    image_size=(224, 224),
    batch_size=20,
    shuffle=True
).take(10)

test = tf.keras.utils.image_dataset_from_directory(
    archive + "/imagenette2-160/val",
    image_size=(224, 224),
    batch_size=20,
    shuffle=True
).take(3)

# Preprocess
train = train.map(lambda x, y: (preprocess_input(x), y))
test = test.map(lambda x, y: (preprocess_input(x), y))

# VGG16 feature extractor
base = VGG16(weights="imagenet", include_top=False)
print("Feature map shape:", base.output.shape)
base.trainable = False

# Classifier
model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(10, activation="softmax")
])
print("Feature vector size: 512")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(train, epochs=1)

# Test
y_true = []
y_pred = []

for x, y in test:
    pred = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(pred, axis=1))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nAccuracy, Precision, Recall and F1:")
print(classification_report(y_true, y_pred))
