import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import dataset.transform as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import get_sample, get_scan
from tqdm import tqdm

tqdm.pandas()


def get_meta_path(sample, scan):
    base_raw_path = Path('/data/visitors/biomax/20180479/20181119/raw')
    tss_dir = base_raw_path / f'{sample}/timed_snapshots/'
    metaglob = tss_dir.glob(f'{scan}_*.meta.txt')
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
    samples = set(df['sample'])
    train, test = train_test_split(list(samples), test_size=0.4, random_state=42)
    valid, test = train_test_split(list(samples - set(train)), test_size=0.5, random_state=42)
    return train, valid, test


def pick_samples(df, samples):
    return df[df['sample'].isin(samples)]


def df_to_xy(df, transform_conf: dict):
    """df, ['sample_dir'] -> ([img], ['y'])"""
    x = np.stack(df['img'].values).reshape(len(df), *(transform_conf['input_shape']))
    y = df['y'].values
    return x, y


def get_dataset_df(csv_path=Path('/data/staff/common/ML-crystals/csv/data_0.5.csv'), has_meta=True):
    from datetime import datetime
    base_raw_path = Path('/data/staff/common/ML-crystals/meta_sandbox')
    df = pd.read_csv(str(csv_path))
    # df = df[df['y'] > 0]
    image_path_pattern = r'raw/([^/]+)/timed_snapshots/(.+)_([0-9.]+).jpeg'
    grouptups = df['filename'].map(lambda x: re.search(image_path_pattern, x).groups())
    df['sample'] = grouptups.map(lambda x: x[0])
    df['scan'] = grouptups.map(lambda x: x[1])
    df['time'] = grouptups.map(lambda x: datetime.fromtimestamp(float(x[2])))

    if has_meta:
        print('loading meta files')
        metas = {
            (get_sample(p), get_scan(p)): json.load(open(str(p), 'r'))
            for p in base_raw_path.rglob('*.meta.txt')
        }

        def get_zoom_int(row):
            zoomtext = metas[(row['sample'], row['scan'])].get('zoom1', 'AAA') or 'Zoom 0'
            return int(re.search('Zoom ([0-9]+)$', zoomtext).group(1))

        print('meta loaded')
        df['zoom'] = df.apply(get_zoom_int, axis=1)
    return df


def get_dataset(df, input_shape):
    train, valid, test = split_dataset(df)
    transform_conf = dict(norm_after_samples=train, input_shape=input_shape)
    df_final = tf.apply_all_transforms(df, conf=transform_conf)
    x_train, y_train = df_to_xy(pick_samples(df_final, train), transform_conf)
    x_test, y_test = df_to_xy(pick_samples(df_final, test), transform_conf)
    x_valid, y_valid = df_to_xy(pick_samples(df_final, valid), transform_conf)
    return [dict(x=x_train, y=y_train), dict(x=x_valid, y=y_valid), dict(x=x_test, y=y_test)]


def dataset_batch_generator(df: pd.DataFrame, batch_size=32):
    i = 0
    while True:
        batch = df[i:min([len(df), i+batch_size])]
        if i+batch_size >= len(df):
            overflow = max([i+batch_size - len(df), 0])
            overflow_part = df[0:overflow]
            batch = pd.concat([batch, overflow_part], axis=0)
        i = (i + batch_size) % len(df)
        yield batch
