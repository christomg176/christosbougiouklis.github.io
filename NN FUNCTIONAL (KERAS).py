import keras
import numpy as np

X =np.random.rand(10000,20)
y = (X.sum(axis=1)>14).astype(int)

# Input shape (e.g., 20 features)
input_shape = (20,)
inputs = keras.Input(shape=input_shape, name="input_layer")

# Hidden layers
x = keras.layers.Dense(64, activation='relu', name="dense_1")(inputs)
x = keras.layers.Dropout(0.5, name="dropout_1")(x)
x = keras.layers.Dense(32, activation='relu', name="dense_2")(x)

# Output layer for binary classification
outputs = keras.layers.Dense(1, activation='sigmoid', name="output_layer")(x)

# Define model
model = keras.Model(inputs=inputs, outputs=outputs, name="functional_binary_model")

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X,y, epochs=20, batch_size=32, validation_split=0.3)
# Summary
model.summary()
