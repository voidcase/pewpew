
import cv2 as cv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import get_sample, get_scan
from tqdm import tqdm

tqdm.pandas()


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


def split_dataset(df):
    # df -> train, valid, test
    samples = set(map(get_sample, df['filename']))
    train, rest = train_test_split(list(samples), test_size=0.4, random_state=42)
    valid, test = train_test_split(list(samples - set(train)), test_size=0.5, random_state=42)
    return train, valid, test


def find_crop_radius(row):
    return 64 + 64 * row['zoom']


def samples_to_xy(df, samples: list, input_shape: tuple):
    """df, ['sample_dir'] -> ([img], ['y_norm'])"""
    channels = input_shape[2]
    flag = cv.IMREAD_COLOR if channels == 3 else cv.IMREAD_GRAYSCALE
    rows = df[df['filename'].apply(get_sample).isin(samples)]
    x = rows.progress_apply(
        lambda row: prep_img(cv.imread(row['filename'], flag), input_shape[:2], crop_radius=find_crop_radius(row)),
        axis=1)
    x = np.stack(x.values).reshape(len(x), *input_shape)
    y = rows['y']
    return x, y


def get_dataset_df(csv_path=Path('/data/staff/common/ML-crystals/csv/data_0.5.csv')):
    base_raw_path = Path('/data/staff/common/ML-crystals/meta_sandbox')
    df = pd.read_csv(str(csv_path))
    df = df[df['y'] > 0]
    df['sample'] = df['filename'].map(get_sample)
    df['scan'] = df['filename'].map(get_scan)
    print('loading meta files')
    metas = {
        (get_sample(p), get_scan(p)): json.load(open(str(p), 'r'))
        for p in base_raw_path.rglob('*.meta.txt')
        if 'Sample-' in str(p)
    }

    def get_zoom_int(row):
        zoomtext = metas[(row['sample'], row['scan'])].get('zoom1', 'AAA') or 'Zoom 0'
        return int(re.search('Zoom ([0-9]+)$', zoomtext).group(1))

    print('meta loaded')
    df['zoom'] = df.apply(get_zoom_int, axis=1)
    return df


def get_dataset(df, input_shape):
    train, valid, test = split_dataset(df)
    x_train, y_train = samples_to_xy(df, train, input_shape)
    mean, std = norm_values(y_train)
    y_train = y_norm(y_train, mean, std)
    x_valid, y_valid = samples_to_xy(df, valid, input_shape)
    y_valid = y_norm(y_valid, mean, std)
    x_test, y_test = samples_to_xy(df, test, input_shape)
    y_test = y_norm(y_test, mean, std)
    return [dict(x=x_train, y=y_train), dict(x=x_valid, y=y_valid), dict(x=x_test, y=y_test)]


def center_of(img):
    return tuple(img.shape[a] // 2 for a in [1, 0])


def y_norm(y, mean, std):
    y = np.log(y)
    return (y - mean) / std


def norm_values(y):
    y = np.log(y)
    return np.mean(y), np.std(y)


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv.LUT(image, table)


def prep_img(img, shape: tuple, center_on=None, crop_radius=None):
    if not center_on:
        center_on = center_of(img)
    cx, cy = center_on
    imrad = min(cx, cy, img.shape[0] - cy, img.shape[1] - cx)
    if crop_radius and crop_radius < imrad:
        imrad = crop_radius
    img = img[cy - imrad: cy + imrad, cx - imrad: cx + imrad]
    img = adjust_gamma(img, 2.5)
    return cv.resize(img, shape)


def show_some(data: pd.DataFrame):
    fig = plt.figure(figsize=(25, 25))
    for i, (_, row) in enumerate(data.sample(n=min(9, len(data))).iterrows()):
        fig.add_subplot(3, 3, i + 1, title=f'{row["sample"]}-{row["scan"]} y:{row["y"]} z:{row["zoom"]}')
        img = cv.imread(row['filename'])
        img = prep_img(img, crop_radius=find_crop_radius(row))
        plt.imshow(img)
        plt.scatter(img.shape[1] / 2, img.shape[0] / 2, c='red', marker='x')

