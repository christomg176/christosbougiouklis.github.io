import tensorflow as tf
from tensorflow import keras
import numpy as np

X = np.random.rand(1000)
y = (X>0.4).astype(int)

model = keras.Sequential([
    keras.layers.Dense(4, activation = "relu", input_shape = (1,)),
    keras.layers.Dense(2, activation = "relu"),
    keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(X,y, epochs = 40)

X1 = np.array([0.6])

predictions = model.predict(X1)

label_x1 = (predictions>0.4).astype(int)

print(predictions,label_x1)