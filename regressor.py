import cv2
import numpy as np
from skimage.transform import resize
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
    return md


def load_data(path):
    from os import listdir
    filenames = listdir(path)
    return np.stack([prep_img(cv2.imread(path+fn, 1)) for fn in filenames])


def gen_bullshit_y_data(length):
    from random import randrange
    return np.array([randrange(0,1000) for i in range(length)])


def stupid_model(x):
    y = gen_bullshit_y_data(x.shape[0])
    md = build_model()
    md.fit(x,y)
    return md


def grid(origin, distance, shape):
    """
    origin: (x, y)
    """
    return [
        (origin[0] + distance * i, origin[1] + distance * j)
        for i in range(shape[0])
        for j in range(shape[1])
        ]

def center_of(img):
    return tuple(a//2 for a in img.shape)

def prep_img(img):
    # TODO cropping
    return resize(img, (100,100))


def create_heatmap(img, model, origo, mapshape, spacing):
    import matplotlib.pyplot as plt
    points = grid(origo, spacing, mapshape)
    plt.imshow(img)
    plt.scatter([x for x,y in points], [y for x,y in points])
    x = np.stack([prep_img(img) for p in points])
    yp = model.predict(x)
    return points, yp


if __name__ == '__main__':
    x = load_data('data/')
    md = stupid_model(x)
    pts, yp = create_heatmap(x[0], md, center_of(x[0]), (3,3), 5)
