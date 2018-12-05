import cv2
import re
import numpy as np
import pandas as pd
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
    md.compile(optimizer='adam', loss='mean_squared_error')
    return md


def load_data(path):
    from os import listdir

    filenames = listdir(path)
    return np.stack([cv2.imread(path + fn, 1) for fn in filenames])


def get_dataset_df(csv_path: Path):
    df = pd.read_csv(str(csv_path))
    df['sample'] = df['filename'].map(
        lambda x: re.search('Sample-([0-9]+-[0-9]+)', x).group(1)
        )
    df['scan'] = df['filename'].map(
        lambda x: re.search('local-user_([0-9]+)_', x).group(1)
        )
    return df



def load_data_2(df: pd.DataFrame):
    images = [
        prep_img(cv2.imread(fname, cv2.IMREAD_COLOR))
        for fname in df['filename']
        if Path(fname).exists()
    ]
    y = np.array(df['y'])
    x = np.stack(images)
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
    md = build_model()
    df = get_dataset_df(Path('/mnt/staff/common/ML-crystals/csv/data_0.5.csv'))
    train_df = df[df['sample'] != '3-12'].sample(frac=1).reset_index(drop=True)
    test_df  = df[df['sample'] == '3-12']
    val_x, val_y = load_data_2(test_df)
    for i in range(5):
        x, y = load_data_2(train_df[i*300:(i+1)*300])
        md.fit(x, y, validation_data=(val_x, val_y))
    print('ta-daaa')
