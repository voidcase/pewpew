import cv2 as cv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def samples_to_xy(df, samples: list, channels: int):
    """df, ['sample_dir'] -> ([img], ['y'])"""
    flag = cv.IMREAD_COLOR if channels == 3 else cv.IMREAD_GRAYSCALE
    rows = df[df['filename'].apply(get_sample).isin(samples)]
    return rows['filename'].progress_apply(lambda path: prep_img(cv.imread(path, flag))), rows['y']


def get_dataset_df(csv_path=Path('/data/staff/common/ML-crystals/csv/data_0.5.csv')):
    base_raw_path = Path('/data/staff/common/ML-crystals/meta_sandbox')
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
    df['zoom'] = df.apply(lambda x: metas[(x['sample'], x['scan'])].get('zoom1', 'AAA'), axis=1)
    return df


def get_dataset(df, input_shape):
    channels = input_shape[2]
    train, valid, test = split_dataset(df)
    x_train, y_train = samples_to_xy(df, train, channels)
    x_valid, y_valid = samples_to_xy(df, valid, channels)
    x_test, y_test = samples_to_xy(df, test, channels)
    X_train = np.stack(x_train.values).reshape(len(x_train), *input_shape)
    X_valid = np.stack(x_valid.values).reshape(len(x_valid), *input_shape)
    X_test = np.stack(x_test.values).reshape(len(x_test), *input_shape)
    return [dict(x=X_train, y=y_train), dict(x=X_valid, y=y_valid), dict(x=X_test, y=y_test)]


def center_of(img):
    return tuple(img.shape[a] // 2 for a in [1, 0])


def prep_img(img, center_on=None, crop_radius=None):
    if not center_on:
        center_on = center_of(img)
    cx, cy = center_on
    imrad = min(cx, cy, img.shape[0] - cy, img.shape[1] - cx)
    if crop_radius and crop_radius < imrad:
        imrad = crop_radius
    img = img[cy - imrad: cy + imrad, cx - imrad: cx + imrad]
    return cv.resize(img, (128, 128))


def show_some(data: pd.DataFrame):
    fig = plt.figure(figsize=(25, 25))
    for i, (_, row) in enumerate(data.sample(n=min(9, len(data))).iterrows()):
        fig.add_subplot(3, 3, i + 1, title=f'{row["sample"]}-{row["scan"]} y:{row["y"]} z:{row["zoom"]}')
        img = cv.imread(row['filename'])
        img = prep_img(img)
        plt.imshow(img)
        plt.scatter(img.shape[1] / 2, img.shape[0] / 2, c='red', marker='x')
