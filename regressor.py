import cv2 as cv
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten

INPUT_SHAPE = (128, 128, 3)


def build_model():
    md = Sequential()
    md.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=INPUT_SHAPE))
    md.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    md.add(Flatten())
    md.add(Dense(32, activation='relu'))
    md.add(Dense(1, activation='relu'))
    md.compile(optimizer='adam', loss='mean_squared_error')
    return md


def load_data(path):
    from os import listdir

    filenames = listdir(path)
    return np.stack([cv.imread(path + fn, 1) for fn in filenames])

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

def sample_dir(image_path):
    return '/'.join(Path(image_path).parts[:8])[1:]

def split_dataset(df):
    # df -> train, valid, test
    samples = set(map(sample_dir, df['filename']))
    train, rest = train_test_split(list(samples), test_size=0.4, random_state=42)
    valid, test = train_test_split(list(samples - set(train)), test_size=0.5, random_state=42)
    return train, valid, test

def samples_to_xy(df, samples: list):
    """df, ['sample_dir'] -> ([img], ['y'])"""
    flag = cv.IMREAD_COLOR if INPUT_SHAPE[2] == 3 else cv.IMREAD_GRAYSCALE
    rows = df[df['filename'].apply(sample_dir).isin(samples)]
    return rows['filename'].progress_apply(lambda path: prep_img(cv.imread(path, flag))), rows['y']

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

def get_dataset(df):
    train, valid, test = split_dataset(df)
    x_train, y_train = samples_to_xy(df, train)
    x_valid, y_valid = samples_to_xy(df, valid)
    x_test, y_test = samples_to_xy(df, test)
    X_train = np.stack(x_train.values).reshape(len(x_train), *INPUT_SHAPE)
    X_valid = np.stack(x_valid.values).reshape(len(x_valid), *INPUT_SHAPE)
    X_test = np.stack(x_test.values).reshape(len(x_test), *INPUT_SHAPE)
    return [dict(x=X_train, y=y_train), dict(x=X_valid, y=y_valid), dict(x=X_test, y=y_test)]

def load_data_2(df: pd.DataFrame):
    flag = cv.IMREAD_COLOR if INPUT_SHAPE[2] == 3 else cv.IMREAD_GRAYSCALE
    images = [prep_img(cv.imread(fname, flag)) for fname in df['filename'] if Path(fname).exists()]
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
    return cv.resize(img, (128, 128))


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
    df = get_dataset_df(Path('/data/staff/common/ML-crystals/csv/data_0.5.csv'))
    train_df = df[df['sample'] != '3-09']
    test_df = df[df['sample'] == '3-09']
    val_x, val_y = load_data_2(test_df)
    for i in range(5):
        x, y = load_data_2(datafile, max_rows=500)
        md.fit(x, y)
