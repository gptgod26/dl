# Deep Learning Lab Master Code: Experiments 1–4
# Dataset: local class folders (e.g., imagenet_subset/class_name/image.jpg)
# Default backbone: ResNet50
#
# IMPORTANT:
# - Run the notebook/script sections in order.
# - Experiment 1 uses the pretrained ImageNet 1000-class head.
# - Experiment 2 visualizes filters and intermediate feature maps.
# - Experiment 3 trains a classifier for YOUR local folder classes.
# - Experiment 4 trains an illustrative CNN+LSTM captioner using captions
#   synthesized from class labels (not human-written captions).
# - This file intentionally preserves the code structure from the study guide.

# ============================================================
# COMMON SETUP
# ============================================================

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


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
# ============================================================
# EXPERIMENT 1 — IMAGENET CLASSIFICATION
# ============================================================

# include_top=True retains the original ImageNet 1000-class head.
model = ResNet50(
    weights="imagenet",
    include_top=True
)

image = load_preprocess_image(X_test_paths[0])
predictions = model.predict(image, verbose=0)

print("Prediction shape:", predictions.shape)

top5 = decode_predictions(predictions, top=5)[0]

for _, name, probability in top5:
    print(name, f"{probability * 100:.2f}%")


# ============================================================
# EXPERIMENT 2 — FILTERS AND FEATURE MAPS
# ============================================================

# include_top=False exposes the convolutional feature-extraction layers.
cnn_model = ResNet50(
    weights="imagenet",
    include_top=False
)
cnn_model.trainable = False
layer_names = [
    "conv1_relu",
    "conv3_block4_out",
    "conv5_block3_out"
]
for name in layer_names:
    print(name, cnn_model.get_layer(name).output.shape)
outputs = []
for name in layer_names:
    outputs.append(cnn_model.get_layer(name).output)



image = load_preprocess_image(X_test_paths[0])
feature_model = Model(
    inputs=cnn_model.input,
    outputs=outputs
)
feature_maps = feature_model.predict(image, verbose=0)
for name, feature in zip(layer_names, feature_maps):
    print(name, "feature shape:", feature.shape)
    plt.figure(figsize=(5, 5))
    plt.imshow(feature[0, :, :, 0], cmap="viridis")
    plt.title(name)
    plt.axis("off")
    plt.show()



# Visualize learned filters from the first convolutional layer.
layer = cnn_model.get_layer("conv1_conv")
filters, biases = layer.get_weights()
print("Filter shape:", filters.shape)
print("Bias shape:", biases.shape)
plt.figure(figsize=(12, 6))
for i in range(8):
    f = filters[:, :, :, i]
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    plt.subplot(2, 4, i + 1)
    plt.imshow(f)
    plt.title(f"Filter {i + 1}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# ============================================================
# EXPERIMENT 3 — TRANSFER LEARNING, CLASSIFIER AND METRICS
# ============================================================

base_model = ResNet50(
    weights="imagenet",
    include_top=False
)
base_model.trainable = False
feature_extractor = Model(
    inputs=base_model.input,
    outputs=GlobalAveragePooling2D()(base_model.output)
)




def extract_features(paths):
    features = []
    for path in paths:
        image = load_preprocess_image(path)
        feature = feature_extractor.predict(image, verbose=0)
        features.append(feature[0])
    return np.array(features, dtype=np.float32)
X_train_features = extract_features(X_train_paths)
X_test_features = extract_features(X_test_paths)
print("Training features:", X_train_features.shape)
print("Testing features:", X_test_features.shape)




num_classes = len(class_names)
classifier = tf.keras.Sequential([
    Input(shape=(2048,)),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(num_classes, activation="softmax")
])
classifier.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = classifier.fit(
    X_train_features,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



# Predict test classes.
probabilities = classifier.predict(X_test_features, verbose=0)
y_pred = np.argmax(probabilities, axis=1)
print("Actual:", y_test[:10])
print("Predicted:", y_pred[:10])
# Calculate evaluation metrics.
print("Accuracy:",
      accuracy_score(y_test, y_pred))
print("Precision:",
      precision_score(
          y_test, y_pred,
          average="weighted", zero_division=0
      ))
print("Recall:",
      recall_score(
          y_test, y_pred,
          average="weighted", zero_division=0
      ))
print("F1 Score:",
      f1_score(
          y_test, y_pred,
          average="weighted", zero_division=0
      ))



# Display confusion matrix.
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
display.plot(xticks_rotation=90)
plt.tight_layout()
plt.show()


# ============================================================
# EXPERIMENT 4 — CNN + LSTM IMAGE CAPTIONING
# ============================================================

# This dataset has class labels, not human-written captions.
# We create simple illustrative captions from each training label.
train_captions = []
for label in y_train:
    name = class_names[label].replace("_", " ")
    caption = "<start> a photo of " + name + " <end>"
    train_captions.append(caption)
tokenizer = Tokenizer(
    filters="",
    lower=True
)
tokenizer.fit_on_texts(train_captions)
train_sequences = tokenizer.texts_to_sequences(train_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(seq) for seq in train_sequences)
print("Vocabulary size:", vocab_size)
print("Maximum caption length:", max_length)




# Create next-word input and target sequences.
X_sequences = []
y_sequences = []
for seq in train_sequences:
    X_sequences.append(seq[:-1])
    y_sequences.append(seq[1:])
X_sequences = pad_sequences(
    X_sequences,
    maxlen=max_length - 1,
    padding="post"
)
y_sequences = pad_sequences(
    y_sequences,
    maxlen=max_length - 1,
    padding="post"
)
X_sequences = np.array(X_sequences, dtype=np.int32)
y_sequences = np.array(y_sequences, dtype=np.int32)





# Define CNN+LSTM caption model.
LSTM_UNITS = 128
EMBEDDING_DIM = 128
image_input = Input(
    shape=(2048,),
    name="image_features"
)

caption_input = Input(
    shape=(max_length - 1,),
    name="caption_sequence"
)

image_hidden = Dense(
    LSTM_UNITS,
    activation="tanh"
)(image_input)

image_cell = Dense(
    LSTM_UNITS,
    activation="tanh"
)(image_input)

embedding = Embedding(
    vocab_size,
    EMBEDDING_DIM
)(caption_input)

lstm_output = LSTM(
    LSTM_UNITS,
    return_sequences=True
)(
    embedding,
    initial_state=[image_hidden, image_cell]
)





# Predict a vocabulary probability distribution at every time step.
output = Dense(
    vocab_size,
    activation="softmax"
)(lstm_output)

caption_model = Model(
    inputs=[image_input, caption_input],
    outputs=output
)

caption_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

caption_model.summary()
caption_history = caption_model.fit(
    [X_train_features, X_sequences],
    y_sequences,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


# ============================================================
# CAPTION GENERATOR AND PREDICTION
# ============================================================

def generate_caption(image_path):
    """Generate a caption one word at a time for a local image."""

    # Load and preprocess image.
    image = load_preprocess_image(image_path)

    # Extract CNN feature vector.
    feature = feature_extractor.predict(image, verbose=0)

    # Begin with the start token.
    caption = ["<start>"]

    for _ in range(max_length - 1):

        # Convert generated words to token IDs.
        sequence = tokenizer.texts_to_sequences(
            [" ".join(caption)]
        )[0]

        current_length = len(sequence)

        # Pad the sequence to the expected input length.
        padded_sequence = pad_sequences(
            [sequence],
            maxlen=max_length - 1,
            padding="post"
        )

        # Predict next-word probabilities.
        predictions = caption_model.predict(
            [feature, padded_sequence],
            verbose=0
        )

        # Choose the highest-probability word at the last real position.
        next_word_id = np.argmax(
            predictions[0, current_length - 1, :]
        )

        next_word = tokenizer.index_word.get(
            next_word_id,
            ""
        )

        # Stop at the end token or an unknown token.
        if next_word in ("<end>", ""):
            break

        caption.append(next_word)

    return " ".join(caption[1:])


# Generate and display a caption for a test image.
test_image = X_test_paths[0]
generated_caption = generate_caption(test_image)

plt.figure(figsize=(7, 6))
plt.imshow(load_img(test_image))
plt.title("Generated: " + generated_caption)
plt.axis("off")
plt.show()

print("Generated caption:", generated_caption)
7

# ============================================================
# QUICK REVISION
# ============================================================
# Experiment 1: Predict with pretrained ImageNet classifier.
# Experiment 2: Visualize convolutional filters and feature maps.
# Experiment 3: Reuse CNN features, train classifier, evaluate metrics.
# Experiment 4: Use CNN features + tokenized text to train LSTM captioner.
#
# If changing the backbone:
# - Update IMG_SIZE and preprocess_input import.
# - Update layer names in Experiment 2.
# - Update pooled feature dimension in classifier/LSTM inputs if needed.
