import keras
from keras.api.utils import to_categorical
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from keras.api.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


(X, y), _ = mnist.load_data()

X_train, X_test, y_train,y_test = train_test_split(X,y, shuffle=True, stratify=y,test_size=0.2,random_state=42)

X_train = X_train.reshape(-1,28,28,1).astype("float32")/255
X_test = X_test.reshape(-1,28,28,1).astype("float32")/255
y_train_c = to_categorical(y_train,10)
y_test_c = to_categorical(y_test,10)

early = EarlyStopping(monitor="val_loss", patience=4,mode="min", restore_best_weights=True)
check = ModelCheckpoint(monitor="val_accuracy", filepath="best_cnn.keras", save_best_only=True,mode="max", verbose=1)

inout_shape = (28,28,1)

inputs = keras.Input(shape=inout_shape, name="input_layer")

x = keras.layers.Conv2D(32,(3,3),activation = "relu",padding="same",name="conv1")(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((2,2))(x)

x = keras.layers.Conv2D(64,(3,3),activation="relu",padding="same", name="con2")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((2,2))(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64,activation="relu", name="dense1")(x)
x = keras.layers.Dropout(0.2)(x)

outputs = keras.layers.Dense(10, activation = "softmax", name="output")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_model")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train,y_train_c, epochs=7, batch_size=23, validation_split=0.2, callbacks=[early,check])

predictions = model.predict(X_test)
label_y = predictions.argmax(axis=1)

print(accuracy_score(y_test_c,label_y))
cn = confusion_matrix(y_test_c, label_y)
sns.heatmap(cn, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
