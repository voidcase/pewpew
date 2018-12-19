import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

def _center_of(img):
    return tuple(img.shape[a] // 2 for a in [1, 0])

def _norm_zoom_radius(row):
    return 64+64*row['zoom']

# ============ROWTRANSFORMS===============

def raw_img(row, color=True):
    return cv.imread(
        row['filename'],
        cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE
        )

def resized_img(row, size=(128,128)):
    return cv.resize(row['img'], size)

def relit_img(row, gamma=1.0):
    image = row['img']
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

def cropped_img(row, center_on=None, crop_radius=None):
    img = row['img']
    if not crop_radius:
        crop_radius = _norm_zoom_radius(row)
    if not center_on:
        center_on = _center_of(img)
    cx, cy = center_on
    imrad = min(cx, cy, img.shape[0] - cy, img.shape[1] - cx)
    if crop_radius and crop_radius < imrad:
        imrad = crop_radius
    img = img[cy - imrad: cy + imrad, cx - imrad: cx + imrad]
    return img

def normed_img(row):
    return (row['img'] - np.mean(row['img'])) / np.std(row['img'])

# =======================================

def norm_y(df: pd.DataFrame, conf: dict) -> pd.DataFrame:
    print('Y normalization')
    df = df.copy()
    if 'norm_after_samples' in conf:
        yn = df[df['sample'].isin(conf['norm_after_samples'])]['y']  # the Y we will scale after
    else:
        yn = df['y']
    mean = np.mean(yn)
    std = np.std(yn)
    df['y'] = np.log(df['y'])
    df['y'] = (df['y'] - mean) / std
    return df

def aug_hflip(df: pd.DataFrame) -> pd.DataFrame:
    print('flip augmentation')
    flipped = df.copy()
    flipped['img'] = flipped['img'].progress_apply(lambda img: cv.flip(img, 1))  # flip along Y axis.
    df = pd.concat([df, flipped], axis=0, ignore_index=True)
    return df

def row_map(df: pd.DataFrame, dst: str, func: callable, args=tuple()):
    print(f'row mapping {func.__name__}')
    df = df.copy()
    df[dst] = df.progress_apply(
        func,
        axis=1,
        args=args,
        )
    return df

def row_pipeline(row, funcs, col):
    tmp = row.copy()
    for f, kwargs in funcs:
        tmp[col] = f(tmp, **kwargs)
    return tmp[col]

def load_and_znorm(df, conf=dict()):
    df = df.copy()
    # I admit this is a bit to meta.
    return row_map(df, 'img', row_pipeline, args=(
            [
            (raw_img, {'color': (conf['input_shape'][2] == 3)}),
            (cropped_img, {}),
            (resized_img, {'size': conf['input_shape'][:2]}),
            ], 'img')
        )

def apply_all_transforms(df: pd.DataFrame, conf: dict):
    df = load_and_znorm(df, conf)
    df = row_map(df, 'img', relit_img)
    df = row_map(df, 'img', normed_img)
    df = norm_y(df, conf)
    df = aug_hflip(df)
    return df

