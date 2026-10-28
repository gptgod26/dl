import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

ds = tf.keras.utils.image_dataset_from_directory(
    "dataset", image_size=(32, 32), batch_size=32
)
classes = ds.class_names
n = len(classes)
ds = ds.take(5)

ds = ds.map(lambda x, y: (x / 127.5 - 1, y))
emb = layers.Embedding(n, 16)

G = tf.keras.Sequential([
    layers.Input((116,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(3072, activation="tanh"),
    layers.Reshape((32, 32, 3))
])

img = layers.Input((32, 32, 3))
lab = layers.Input((), dtype="int32")
e = layers.Dense(1024)(emb(lab))
e = layers.Reshape((32, 32, 1))(e)
x = layers.Concatenate()([img, e])
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
D = tf.keras.Model([img, lab], layers.Dense(1, activation="sigmoid")(x))

optG = tf.keras.optimizers.Adam(0.0002)
optD = tf.keras.optimizers.Adam(0.0002)
loss = tf.keras.losses.BinaryCrossentropy()

for epoch in range(5):
    for real, y in ds:
        z = tf.random.normal((tf.shape(real)[0], 100))
        with tf.GradientTape() as t:
            fake = G(tf.concat([z, emb(y)], axis=1))
            r = D([real, y])
            f = D([fake, y])
            dl = loss(tf.ones_like(r), r) + loss(tf.zeros_like(f), f)
        optD.apply_gradients(zip(t.gradient(dl, D.trainable_variables), D.trainable_variables))

        with tf.GradientTape() as t:
            fake = G(tf.concat([z, emb(y)], axis=1))
            f = D([fake, y])
            gl = loss(tf.ones_like(f), f)
        optG.apply_gradients(zip(t.gradient(gl, G.trainable_variables), G.trainable_variables))

# Generate images
plt.figure(figsize=(10, 2))
for i in range(min(5, n)):
    z = tf.random.normal((1, 100))
    fake = G(tf.concat([z, emb(tf.constant([i]))], axis=1))
    plt.subplot(1, 5, i + 1)
    plt.imshow(np.clip((fake[0].numpy() + 1) / 2, 0, 1))
    plt.title(classes[i])
    plt.axis("off")
plt.show()
