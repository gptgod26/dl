import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Download Imagenette
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

archive = tf.keras.utils.get_file(
    "imagenette2-160.tgz",
    url,
    untar=True
)

# Load validation images
dataset = tf.keras.utils.image_dataset_from_directory(
    archive + "/imagenette2-160/val",
    image_size=(224, 224),
    batch_size=5,
    shuffle=True
)

print("Classes:", dataset.class_names)

# Get 5 images
images, labels = next(iter(dataset))

print("Original shape:", images.shape)
print("Preprocessed shape: (5, 224, 224, 3)")

# Display images and original labels
plt.figure(figsize=(12, 5))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(dataset.class_names[labels[i]])
    plt.axis("off")

plt.show()

# Load pretrained VGG16
model = VGG16(weights="imagenet")

# Preprocess images
x = preprocess_input(images)

# Show preprocessed representation
print("Pixel value before preprocessing:", images[0][0][0])
print("Pixel value after preprocessing :", x[0][0][0])

# Predict
predictions = model.predict(x)

# Top 5 predictions
results = decode_predictions(predictions, top=5)

# Display predictions and compare with original labels
for i in range(5):
    print("\nImage", i + 1)
    print("Original label:", dataset.class_names[labels[i]])
    print("Top-5 predictions:")

    for _, label, probability in results[i]:
        print(label,_, ":", round(probability * 100, 2), "%")

    print("Predicted label:", results[i][0][1])
