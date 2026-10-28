
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import numpy as np

SIZE = 64
NOISE = 100
EPOCHS = 20

# ============================================================
# 1. LOAD CLASS LABELS
# ============================================================

with open("labels.json") as f:
    data_json = json.load(f)

label_map = {item[1]: item[2] for item in data_json}

# ============================================================
# 2. LOAD DATASET
# ============================================================

data = tf.keras.utils.image_dataset_from_directory(
    "imagenet_folder",
    image_size=(SIZE, SIZE),
    batch_size=32,
    shuffle=True
)

folder_names = data.class_names
classes = [label_map.get(name, name) for name in folder_names]
n = len(classes)

# Normalize images from [0,255] to [-1,1]
data = data.map(
    lambda images, labels: (images / 127.5 - 1.0, labels)
)

print("Classes:", classes)

# ============================================================
# 3. GENERATOR
# ============================================================

generator = tf.keras.Sequential([
    tf.keras.layers.Input((NOISE + n,)),

    tf.keras.layers.Dense(8 * 8 * 128),
    tf.keras.layers.Reshape((8, 8, 128)),

    tf.keras.layers.Conv2DTranspose(
        64, 4, 2, padding="same", activation="relu"
    ),

    tf.keras.layers.Conv2DTranspose(
        32, 4, 2, padding="same", activation="relu"
    ),

    tf.keras.layers.Conv2DTranspose(
        3, 4, 2, padding="same", activation="tanh"
    )
])

# ============================================================
# 4. DISCRIMINATOR
# ============================================================

discriminator = tf.keras.Sequential([
    tf.keras.layers.Input((SIZE, SIZE, 3 + n)),

    tf.keras.layers.Conv2D(
        64, 4, 2, padding="same"
    ),

    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

# ============================================================
# 5. LOSS AND OPTIMIZERS
# ============================================================

loss_fn = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)

g_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

# ============================================================
# 6. TRAINING STEP
# ============================================================

@tf.function
def train_step(real_images, labels):

    batch = tf.shape(real_images)[0]

    # Convert class IDs to one-hot encoding
    text = tf.one_hot(labels, n)

    # Generate random noise
    noise = tf.random.normal((batch, NOISE))

    # Combine noise and class label
    gen_input = tf.concat([noise, text], axis=1)

    # Convert class labels to image-shaped channels
    label_images = tf.reshape(text, (batch, 1, 1, n))
    label_images = tf.tile(label_images, [1, SIZE, SIZE, 1])

    # ----------------------------
    # Train Discriminator
    # ----------------------------

    with tf.GradientTape() as d_tape:

        fake_images = generator(gen_input, training=True)

        real_input = tf.concat(
            [real_images, label_images], axis=-1
        )

        fake_input = tf.concat(
            [tf.stop_gradient(fake_images), label_images],
            axis=-1
        )

        real_output = discriminator(real_input, training=True)
        fake_output = discriminator(fake_input, training=True)

        real_loss = loss_fn(
            tf.ones_like(real_output), real_output
        )

        fake_loss = loss_fn(
            tf.zeros_like(fake_output), fake_output
        )

        d_loss = real_loss + fake_loss

    d_gradients = d_tape.gradient(
        d_loss, discriminator.trainable_variables
    )

    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables)
    )

    # ----------------------------
    # Train Generator
    # ----------------------------

    with tf.GradientTape() as g_tape:

        fake_images = generator(gen_input, training=True)

        fake_input = tf.concat(
            [fake_images, label_images], axis=-1
        )

        fake_output = discriminator(fake_input, training=True)

        g_loss = loss_fn(
            tf.ones_like(fake_output), fake_output
        )

    g_gradients = g_tape.gradient(
        g_loss, generator.trainable_variables
    )

    g_optimizer.apply_gradients(
        zip(g_gradients, generator.trainable_variables)
    )

    return g_loss, d_loss


# ============================================================
# 7. TRAIN THE GAN
# ============================================================

g_losses = []
d_losses = []

for epoch in range(EPOCHS):

    g_total = 0.0
    d_total = 0.0
    steps = 0

    for images, labels in data:

        g_loss, d_loss = train_step(images, labels)

        g_total += float(g_loss.numpy())
        d_total += float(d_loss.numpy())
        steps += 1

    g_avg = g_total / steps
    d_avg = d_total / steps

    g_losses.append(g_avg)
    d_losses.append(d_avg)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"G Loss: {g_avg:.4f} | "
        f"D Loss: {d_avg:.4f}"
    )


# ============================================================
# 8. PLOT GENERATOR AND DISCRIMINATOR LOSSES
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CGAN Training Loss")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 9. GENERATE IMAGE FROM CLASS LABEL
# ============================================================

class_id = 0

text = tf.one_hot([class_id], n)
noise = tf.random.normal((1, NOISE))

generated = generator(
    tf.concat([noise, text], axis=1),
    training=False
)

image = (generated[0] + 1) / 2

print("Text: A photo of a", classes[class_id])

plt.figure(figsize=(5, 5))
plt.imshow(tf.clip_by_value(image, 0, 1))
plt.title("Generated " + classes[class_id])
plt.axis("off")
plt.show()


# ============================================================
# 10. GENERATE MULTIPLE IMAGES
# ============================================================

plt.figure(figsize=(10, 5))

for i in range(min(8, n)):

    text = tf.one_hot([i], n)
    noise = tf.random.normal((1, NOISE))

    generated = generator(
        tf.concat([noise, text], axis=1),
        training=False
    )

    image = (generated[0] + 1) / 2

    plt.subplot(2, 4, i + 1)
    plt.imshow(tf.clip_by_value(image, 0, 1))
    plt.title(classes[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
