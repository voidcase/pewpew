from tensorflow import keras
import numpy as np
from tensorflow.nn import relu

IMAGE_SHAPE = (1000, 1000) #TEMP

def build_model():
    md = keras.models.Sequential()
    md.add(keras.layers.Conv2D(filters=32,
        kernel_size=3,
        activation=relu,
        ))
    md.add(keras.layers.MaxPool2D(pool_size=2))
    md.add(keras.layers.Conv2D(filters=32,
        kernel_size=3,
        activation=relu,
        ))
    md.add(keras.layers.MaxPool2D(pool_size=2))
    md.add(keras.layers.Flatten())
    md.add(keras.layers.Dense(32, activation=relu))
    md.add(keras.layers.Dense(1, activation=relu))
