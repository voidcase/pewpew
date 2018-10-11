import cv2
import numpy as np
from tensorflow import keras
from tensorflow.nn import relu

IMAGE_SHAPE = (1000, 1000) #TEMP

def build_model():
    md = keras.models.Sequential()
    md.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation=relu,
        ))
    md.add(keras.layers.MaxPool2D(pool_size=2))
    md.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation=relu,
        ))
    md.add(keras.layers.MaxPool2D(pool_size=2))
    md.add(keras.layers.Flatten())
    md.add(keras.layers.Dense(32, activation=relu))
    md.add(keras.layers.Dense(1, activation=relu))

def load_data(path):
    from os import listdir
    filenames = listdir(path)
    return np.stack([cv2.imread(path+fn, 0) for fn in filenames])

if __name__ == '__main__':
    data = load_data('./data/')
    assert False
