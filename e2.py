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
