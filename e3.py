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

