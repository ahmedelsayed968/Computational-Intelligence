from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from optimization.SGD import SGDTrainer


def get_model(input_shape: Tuple[int], neurons: List[int], output_neurons: int):
    input_layer = Input(shape=input_shape)
    x = input_layer

    for i in neurons:
        x = Dense(units=i, activation="relu")(x)
    output = Dense(units=output_neurons, activation="linear")(x)
    return Model(input_layer, output)


X = np.linspace(1, 10000, num=10000)
y = X**2
y_train, x_train = tf.cast(y, dtype=tf.float32), tf.cast(X, dtype=tf.float32)

model = get_model(input_shape=(1,), neurons=[10], output_neurons=1)
loss_fn = tf.keras.losses.MeanSquaredError()
trainer = SGDTrainer(model, x_train[:1000], y_train[:1000], 2, loss_fn, momentum=0.9)
