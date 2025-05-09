# unet_model.py
import keras
import numpy as np

class UNet:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build(self):
        inputs = keras.layers.Input(self.input_shape)
        c1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
        p1 = keras.layers.MaxPooling2D()(c1)
        c2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
        u1 = keras.layers.UpSampling2D()(c2)
        concat1 = keras.layers.Concatenate()([u1, c1])
        outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(concat1)
        self.model = keras.models.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x, y, val):
        self.model.fit(x, y, validation_data=val, epochs=10)

    def predict(self, x):
        return self.model.predict(x)
# run_segmentation.py




# Dummy synthetic data
x_train = np.random.rand(100, 64, 64, 1)
y_train = (x_train > 0.5).astype("float32")
x_val = np.random.rand(10, 64, 64, 1)
y_val = (x_val > 0.5).astype("float32")
x_test = np.random.rand(100,64,64,1)

net = UNet((64, 64, 1))
net.build()
net.train(x_train, y_train, val=(x_val, y_val))
net.predict(x_test)