# EX 5 - Conditional GAN

import tensorflow as tf
import matplotlib.pyplot as plt
import json

SIZE = 64
NOISE = 100

# Load class labels from JSON
with open("labels.json") as f:
    data_json = json.load(f)

label_map = {item[1]: item[2] for item in data_json}

# Load dataset from local folder
data = tf.keras.utils.image_dataset_from_directory(
    "imagenet_folder",
    image_size=(SIZE, SIZE),
    batch_size=32,
    shuffle=True
)

# Folder names like ncx101, ncx102 ...
folder_names = data.class_names

# Actual class names like cat, dog ...
classes = [label_map.get(name, name) for name in folder_names]
n = len(classes)

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Input((NOISE + n,)),
    tf.keras.layers.Dense(8 * 8 * 128),
    tf.keras.layers.Reshape((8, 8, 128)),
    tf.keras.layers.Conv2DTranspose(64, 4, 2, padding="same", activation="relu"),
    tf.keras.layers.Conv2DTranspose(32, 4, 2, padding="same", activation="relu"),
    tf.keras.layers.Conv2DTranspose(3, 4, 2, padding="same", activation="tanh")
])

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Input((SIZE, SIZE, 3 + n)),
    tf.keras.layers.Conv2D(64, 4, 2, padding="same"),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

# Generate image from class label
class_id = 0
text = tf.one_hot([class_id], n)
noise = tf.random.normal((1, NOISE))

generated = generator(tf.concat([noise, text], axis=1))
image = (generated[0] + 1) / 2

print("Text: A photo of a", classes[class_id])

plt.imshow(image)
plt.title("Generated " + classes[class_id])
plt.axis("off")
plt.show()
