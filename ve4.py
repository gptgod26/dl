# EX 4 - Image Caption Generation using CNN + LSTM

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Imagenette
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
archive = tf.keras.utils.get_file("imagenette2-160.tgz", url, untar=True)

data = tf.keras.utils.image_dataset_from_directory(
    archive + "/imagenette2-160/train",
    image_size=(224, 224), batch_size=20, shuffle=True
).take(10)

# CNN feature extraction
cnn = VGG16(weights="imagenet", include_top=False, pooling="avg")
images, labels = next(iter(data))
features = cnn.predict(preprocess_input(images), verbose=0)

# Create captions from actual dataset class labels
classes = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute"
]
captions = ["A photo of " + classes[int(y)] for y in labels.numpy()]

# Display images and captions
for i in range(3):
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(captions[i])
    plt.axis("off")
    plt.show()

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
seq = pad_sequences(tokenizer.texts_to_sequences(captions))

# Simple LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(features.shape[1],)),
    tf.keras.layers.RepeatVector(seq.shape[1]),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(features, seq[:, -1], epochs=5, verbose=1)

print("CNN feature shape:", features.shape)
print("Expected Caption:", captions[0])
