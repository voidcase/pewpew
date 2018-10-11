import cv2
import numpy as np
from scipy.misc import imresize
from tensorflow import keras
from tensorflow.nn import relu


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
    md.compile(
            optimizer='adam',
            loss='mean_squared_error',
            )



def load_data(path):
    from os import listdir
    filenames = listdir(path)
    return np.stack([imresize(cv2.imread(path+fn, 1), 0.2) for fn in filenames])


def gen_bullshit_y_data(length):
    from random import randrange
    return np.array([randrange(0,1000) for i in range(length)])


def trained_model():
    x = load_data('data/')
    y = gen_bullshit_y_data(x.shape[0])
    md = build_model()
    md.fit(x,y)
    return md


if __name__ == '__main__':
    trained_model()
