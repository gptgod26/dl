
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASET_DIR = "imagenet_subset"
IMG_SIZE = (224, 224)



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
    raise ValueError("No images found in dataset")

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
    image = load_img(path, target_size=IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


# Display original image
image = load_img(X_train_paths[0], target_size=IMG_SIZE)

plt.imshow(image)
plt.title(class_names[y_train[0]])
plt.axis("off")
plt.show()


# 3. CREATE TRAINING CAPTIONS


train_captions = []

for label in y_train:
    name = class_names[label].replace("_", " ")
    caption = "<start> a photo of " + name + " <end>"
    train_captions.append(caption)

print("Example caption:", train_captions[0])



# 4. TOKENIZATION


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



# 5. CREATE INPUT AND TARGET SEQUENCES

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

# Ignore padding positions while calculating loss and accuracy
sample_weights = (y_sequences != 0).astype("float32")


# 6. CNN FEATURE EXTRACTION USING RESNET50
feature_extractor = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

feature_extractor.trainable = False

X_train_features = []

for path in X_train_paths:
    image = load_preprocess_image(path)

    feature = feature_extractor.predict(
        image,
        verbose=0
    )

    X_train_features.append(feature[0])

X_train_features = np.array(
    X_train_features,
    dtype=np.float32
)

print("CNN Features Shape:", X_train_features.shape)



# 7. DEFINE CNN + LSTM CAPTION MODEL


LSTM_UNITS = 128
EMBEDDING_DIM = 128

# Image input
image_input = Input(
    shape=(2048,),
    name="image_features"
)

# Caption input
caption_input = Input(
    shape=(max_length - 1,),
    name="caption_sequence"
)

# Initialize LSTM hidden and cell states using image features
image_hidden = Dense(
    LSTM_UNITS,
    activation="tanh"
)(image_input)

image_cell = Dense(
    LSTM_UNITS,
    activation="tanh"
)(image_input)

# Word embedding
embedding = Embedding(
    vocab_size,
    EMBEDDING_DIM
)(caption_input)

# LSTM
lstm_output = LSTM(
    LSTM_UNITS,
    return_sequences=True
)(
    embedding,
    initial_state=[image_hidden, image_cell]
)

# Predict next word
output = Dense(
    vocab_size,
    activation="softmax"
)(lstm_output)

# Create model
caption_model = Model(
    inputs=[image_input, caption_input],
    outputs=output
)

caption_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    weighted_metrics=["accuracy"]
)

caption_model.summary()



# 8. TRAIN THE MODEL
caption_history = caption_model.fit(
    [X_train_features, X_sequences],
    y_sequences,
    sample_weight=sample_weights,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)



# 9. CAPTION GENERATOR

def generate_caption(image_path):

    # Load and preprocess image
    image = load_preprocess_image(image_path)

    # Extract CNN features
    feature = feature_extractor.predict(
        image,
        verbose=0
    )

    # Start caption
    caption = ["<start>"]

    for _ in range(max_length - 1):

        # Convert words into token IDs
        sequence = tokenizer.texts_to_sequences(
            [" ".join(caption)]
        )[0]

        current_length = len(sequence)

        # Pad the sequence
        padded_sequence = pad_sequences(
            [sequence],
            maxlen=max_length - 1,
            padding="post"
        )

        # Predict next word
        predictions = caption_model.predict(
            [feature, padded_sequence],
            verbose=0
        )

        next_word_id = np.argmax(
            predictions[0, current_length - 1, :]
        )

        # Convert token ID to word
        next_word = tokenizer.index_word.get(
            int(next_word_id),
            ""
        )

        # Stop if caption ends
        if next_word in ("<end>", ""):
            break

        caption.append(next_word)

    return " ".join(caption[1:])


# 10. GENERATE AND DISPLAY CAPTION

test_image = X_test_paths[0]

generated_caption = generate_caption(test_image)

plt.figure(figsize=(7, 6))
plt.imshow(load_img(test_image))
plt.title("Generated: " + generated_caption)
plt.axis("off")
plt.show()

print("Generated caption:", generated_caption)



# 11. PLOT TRAINING ACCURACY AND LOSS


plt.figure(figsize=(6, 4))
plt.plot(caption_history.history["loss"], label="Training Loss")
plt.plot(caption_history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(caption_history.history["weighted_accuracy"], label="Training Accuracy")
plt.plot(caption_history.history["val_weighted_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
