import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load VGG16
model = VGG16(weights="imagenet")

# Take one image
image = images[0:1]
x = preprocess_input(image)

# Select 3 layers
layers = ["block1_conv1", "block2_conv2", "block5_conv3"]

# Create model to get feature maps
feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=[model.get_layer(layer).output for layer in layers]
)

features = feature_model.predict(x)

# Display feature maps
for layer, feature in zip(layers, features):

    print(layer, "shape:", feature.shape)

    plt.figure(figsize=(10, 5))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(feature[0, :, :, i], cmap="viridis")
        plt.axis("off")

    plt.suptitle(layer)
    plt.show()

# Display filters from first convolutional layer
filters = model.get_layer("block1_conv1").get_weights()[0]

print("Filter shape:", filters.shape)

plt.figure(figsize=(10, 5))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(filters[:, :, :, i])
    plt.axis("off")

plt.suptitle("VGG16 Convolutional Filters")
plt.show()
