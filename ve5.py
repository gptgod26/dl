# EX 1 - Conditional GAN

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

SIZE, BATCH, NOISE = 64, 32, 100

# 1. Download and load Imagenette
archive =f””

folder = os.path.join(archive, "train")
if not os.path.exists(folder):
    folder = os.path.join(os.path.dirname(archive), "imagenette2-160", "train")

data = tf.keras.utils.image_dataset_from_directory(
    folder, image_size=(SIZE, SIZE), batch_size=BATCH
)

classes = data.class_names
N = len(classes)

data = data.map(lambda x, y: (x / 127.5 - 1, y))

# 2. Text embedding
embedding = layers.Embedding(N, 16)

# 3. Generator
G = tf.keras.Sequential([
    layers.Input((NOISE + 16,)),
    layers.Dense(8 * 8 * 128, activation="relu"),
    layers.Reshape((8, 8, 128)),
    layers.UpSampling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.UpSampling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.UpSampling2D(),
    layers.Conv2D(3, 3, padding="same", activation="tanh")
])

# 4. Discriminator
D = tf.keras.Sequential([
    layers.Input((SIZE, SIZE, 4)),
    layers.Conv2D(32, 4, strides=2, padding="same"),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid")
])

g_opt = tf.keras.optimizers.Adam(0.0002)
d_opt = tf.keras.optimizers.Adam(0.0002)
loss = tf.keras.losses.BinaryCrossentropy()

# 5. Train GAN
for epoch in range(2):
    for real, label in data:
        n = tf.shape(real)[0]
        text = embedding(label)
        noise = tf.random.normal((n, NOISE))
        condition = tf.tile(
            text[:, None, None, :], [1, SIZE, SIZE, 1]
        )
        condition_small = tf.reduce_mean(condition, axis=[1, 2])
        fake = G(tf.concat([noise, condition_small], axis=1))

        real_input = tf.concat([real, condition], axis=-1)
        fake_input = tf.concat([fake, condition], axis=-1)

        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = G(tf.concat([noise, condition_small], axis=1))
            fake_input = tf.concat([fake, condition], axis=-1)

            real_out = D(real_input)
            fake_out = D(fake_input)

            d_loss = loss(tf.ones_like(real_out), real_out) + \
                     loss(tf.zeros_like(fake_out), fake_out)
            g_loss = loss(tf.ones_like(fake_out), fake_out)

        d_opt.apply_gradients(zip(
            dt.gradient(d_loss, D.trainable_variables),
            D.trainable_variables
        ))
        g_opt.apply_gradients(zip(
            gt.gradient(g_loss, G.trainable_variables),
            G.trainable_variables
        ))

    print("Epoch:", epoch + 1, "Generator loss:", float(g_loss))

# 6. Generate image from a text description
prompt = "a photo of a tiger"
class_name = next((c for c in classes if c.lower() in prompt.lower()), classes[0])
label = tf.constant([classes.index(class_name)])

text = embedding(label)
noise = tf.random.normal((1, NOISE))
generated = G(tf.concat([noise, text], axis=1))[0]
generated = (generated + 1) / 2

plt.imshow(generated.numpy())
plt.title(prompt + " (" + class_name + ")")
plt.axis("off")
plt.show() ‎<This message was edited>
