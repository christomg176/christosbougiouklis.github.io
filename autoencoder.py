import keras
from keras.api.datasets import mnist
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Callbacks
early = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience=10)
check = ModelCheckpoint(monitor="val_loss", mode="min", save_best_only=True,
                        filepath="best_autoencoder_model.keras", verbose=1)

# Define input
input_shape = (28, 28, 1)
inputs = keras.Input(shape=input_shape, name="input_layer")

# Encoder
x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(inputs)
x = keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)

x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
encoded = keras.layers.MaxPooling2D((2, 2), padding="same", name="encoded")(x)

# Decoder
x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_conv1")(encoded)

x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv2")(x)
x = keras.layers.UpSampling2D((2, 2), name="dec_upsample2")(x)

decoded = keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="output")(x)

# Build model
autoencoder = keras.Model(inputs=inputs, outputs=decoded, name="autoencoder_model")

# Compile
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss="binary_crossentropy")

# Train
autoencoder.fit(x_train, x_train, epochs=20, batch_size=64,
                validation_split=0.2, callbacks=[early, check])

# Predict
decoded_imgs = autoencoder.predict(x_test)

# Display original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.suptitle("Autoencoder Results")
plt.show()
