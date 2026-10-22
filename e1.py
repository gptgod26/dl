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
