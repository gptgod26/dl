# EX 5 - Conditional GAN

import tensorflow as tf
import matplotlib.pyplot as plt

SIZE = 64
NOISE = 100

# Download Imagenette
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

archive = tf.keras.utils.get_file(
    "imagenette2-160.tgz",
    url,
    untar=True
)

# Load dataset
data = tf.keras.utils.image_dataset_from_directory(
    archive + "/imagenette2-160/train",
    image_size=(SIZE, SIZE),
    batch_size=32
)

classes = data.class_names
n = len(classes)

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Input((NOISE + n,)),
    tf.keras.layers.Dense(8*8*128),
    tf.keras.layers.Reshape((8,8,128)),
    tf.keras.layers.Conv2DTranspose(64, 4, 2, padding="same", activation="relu"),
    tf.keras.layers.Conv2DTranspose(32, 4, 2, padding="same", activation="relu"),
    tf.keras.layers.Conv2DTranspose(3, 4, 2, padding="same", activation="tanh")
])

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Input((SIZE, SIZE, 3+n)),
    tf.keras.layers.Conv2D(64, 4, 2, padding="same"),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

# Generate image from text
class_id = 0
text = tf.one_hot([class_id], n)
noise = tf.random.normal((1, NOISE))

generated = generator(
    tf.concat([noise, text], axis=1)
)

image = (generated[0] + 1) / 2

print("Text: A photo of a", classes[class_id])

plt.imshow(image)
plt.title("Generated " + classes[class_id])
plt.axis("off")
plt.show()
