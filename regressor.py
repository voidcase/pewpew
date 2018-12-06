import cv2
import numpy as np
from skimage.transform import resize
from tensorflow import keras
from tensorflow.nn import relu
from pathlib import Path

def build_model():
    md = keras.models.Sequential()
    md.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=relu))
    md.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=relu))
    md.add(keras.layers.Flatten())
    md.add(keras.layers.Dense(32, activation=relu))
    md.add(keras.layers.Dense(1, activation=relu))
    md.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return md


def load_data(path):
    from os import listdir

    filenames = listdir(path)
    return np.stack([cv2.imread(path + fn, 1) for fn in filenames])



def load_data_2(csv_stream, max_rows=256):
    '''
    no headers plz
    first column: image path
    second column: y value
    '''
    import csv
    images = []
    y = []
    reader = csv.reader(csv_stream)
    for row in reader:
        if not Path(row[0]).exists():
            print(f'no path: "{row[0]}"')
            continue
        images.append(prep_img(cv2.imread(row[0], cv2.IMREAD_COLOR)))  # loads in grayscale
        # y.append(np.array([row[1]]))
        y.append(row[1])
        if reader.line_num >= max_rows:
            break
    x = np.stack(images)
    y = np.array(y)
    return x, y


def gen_bullshit_y_data(length):
    from random import randrange

    return np.array([randrange(0, 1000) for i in range(length)])


def stupid_model(x):
    y = gen_bullshit_y_data(x.shape[0])
    md = build_model()
    md.fit(x, y)
    return md


def grid(origin, distance, shape):
    '''
    origin: (x, y)
    '''
    return [
        (origin[0] + distance * i, origin[1] + distance * j)
        for i in range(shape[0])
        for j in range(shape[1])
    ]


def center_of(img):
    return tuple(img.shape[a] // 2 for a in [1, 0])


def prep_img(img, center_on=None):
    # TODO cropping
    if not center_on:
        center_on = center_of(img)
    cx, cy = center_on
    imrad = min(cx, cy, img.shape[0] - cy, img.shape[1] - cx)
    img = img[cy - imrad : cy + imrad, cx - imrad : cx + imrad]
    return resize(img, (128, 128))


def create_heatmap(img, model, origo, mapshape, spacing):
    points = grid(origo, spacing, mapshape)
    x = np.stack([prep_img(img, center_on=p) for p in points])
    yp = model.predict(x).reshape(-1)
    return points, yp


def draw_heatmap(img, points, yp):
    import matplotlib.pyplot as plt

    plt.imshow(img)
    px, py = tuple([point[d] for point in points] for d in [0, 1])
    plt.scatter(px, py, c=yp, cmap='hot')
    plt.show()


if __name__ == '__main__':
    # raw = load_data('data/')
    # x = np.stack(prep_img(im) for im in raw)
    # md = stupid_model(x)
    # pts, yp = create_heatmap(raw[0], md, (10, 10), (30, 30), 3)
    # draw_heatmap(x[0], pts, yp)
    datafile = open('/data/staff/common/ML-crystals/csv/data_0.5.json.csv', 'r')
    md = build_model()
    for i in range(5):
        x, y = load_data_2(datafile, max_rows=500)
        md.fit(x, y)
    print('ta-daaa')
