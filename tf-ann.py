import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_breast_cancer

df = load_breast_cancer()

# print(df)

X, y = df["data"], df["target"]

# scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

X.shape

# input data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


# ann with three layers (16-16-1)
model = k.Sequential(
    [
        k.layers.Dense(16, activation="relu", input_shape=(30,)),
        k.layers.Dense(16, activation="relu"),
        k.layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.binary_crossentropy,
    metrics=[k.metrics.binary_accuracy],
)

epochs = 100

model.fit(X, y, epochs=epochs)

# loss graph
plt.plot(range(epochs), model.history.history["loss"])
plt.grid(True)
plt.show()

# accuracy graph
plt.plot(range(epochs), model.history.history["binary_accuracy"])
plt.grid(True)
plt.show()
