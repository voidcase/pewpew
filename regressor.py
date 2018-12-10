import cv2
import re
import json
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

def get_meta_path(sample, scan):
    base_raw_path = Path('/data/visitors/biomax/20180479/20181119/raw')
    tss_dir = base_raw_path / f'Sample-{sample}/timed_snapshots/'
    metaglob = tss_dir.glob(f'local-user_{scan}_*.meta.txt')
    try:
        return next(metaglob)
    except StopIteration:
        print(f'no meta file for {sample} {scan}')
        return None


def get_meta_file(row):
    with open(str(get_meta_path(row['sample'], row['scan'])), 'r') as f:
        return json.load(f)


def get_sample(x: Path):
    return str(re.search('Sample-([0-9]+-[0-9]+)', str(x)).group(1))


def get_scan(x: Path):
    return str(re.search('local-user_([0-9]+)_', str(x)).group(1))


def get_dataset_df(csv_path = Path('/data/staff/common/ML-crystals/csv/data_0.5.csv')):
    base_raw_path = Path('/data/visitors/biomax/20180479/20181119/raw')

    df = pd.read_csv(str(csv_path))
    df['sample'] = df['filename'].map(get_sample)
    df['scan'] = df['filename'].map(get_scan)
    print('loading meta files')
    metas = {
        (get_sample(p), get_scan(p)): json.load(open(str(p), 'r'))
        for p in base_raw_path.rglob('*.meta.txt')
        if 'Sample-' in str(p)
        }
    print('meta loaded')
    df['zoom'] = df.apply(lambda x: metas[(x['sample'], x['scan'])].get('zoom1','AAA'),axis=1)
    return df


def load_data_2(df: pd.DataFrame):
    images = [prep_img(cv2.imread(fname, cv2.IMREAD_COLOR)) for fname in df['filename'] if Path(fname).exists()]
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
    return [(origin[0] + distance * i, origin[1] + distance * j) for i in range(shape[0]) for j in range(shape[1])]


def center_of(img):
    return tuple(img.shape[a] // 2 for a in [1, 0])


def prep_img(img, center_on=None, crop_radius=None):
    # TODO cropping
    if not center_on:
        center_on = center_of(img)
    cx, cy = center_on
    imrad = min(cx, cy, img.shape[0] - cy, img.shape[1] - cx)
    if crop_radius and crop_radius < imrad:
        imrad = crop_radius
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
    train_df = df[df['sample'] != '3-09']
    test_df = df[df['sample'] == '3-09']
    val_x, val_y = load_data_2(test_df)
    for i in range(5):
        x, y = load_data_2(datafile, max_rows=500)
        md.fit(x, y)
