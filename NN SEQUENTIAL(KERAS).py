import  keras
import numpy as np
from keras.api.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor="val_loss", patience=4,mode="min", restore_best_weights=True)
check = ModelCheckpoint(filepath="best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)


X = np.random.rand(10000,20)
y = (X.sum(axis=1)>10).astype(int)

model = keras.Sequential([
    keras.layers.Dense(128,activation="relu", input_shape=(20,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")

])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X,y, epochs=25,batch_size=12, validation_split=0.2)

X1test = np.random.rand(1,20)

prediction = model.predict(X1test)
print(prediction)
label = (prediction>0.5).astype(int)
print(label)