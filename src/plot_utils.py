import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ceil(n, base):
    """Return closest multiple of base with ceiling rounding"""
    return int(base * np.ceil(float(n) / base))

def imshow(axes, img, title = '', mark_center=True, param_dict = None):
    if param_dict is None:
        param_dict = {}
    axes.imshow(img, **param_dict)
    if mark_center:
        axes.scatter(img.shape[1]/2,img.shape[0]/2, c='red', marker='x')
    axes.set_title(title)

def image_grid(images: list, titles: list = None, max_cols = 4):
    if titles is None:
        titles = [''] * len(images)
    else:
        assert(len(images) == len(titles))
        
    cols = min(len(images), max_cols)
    rows = ceil(len(images), max_cols) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows))
    axes = axes.flatten()
    l = min(len(images), rows*cols)
    for i in range(l):
        imshow(axes[i], images[i], titles[i])


def show_some(data: pd.DataFrame, seed=None):
    fig = plt.figure(figsize=(25, 25))
    for i, (_, row) in enumerate(data.sample(n=min(9, len(data)), random_state=seed).iterrows()):
        fig.add_subplot(3, 3, i + 1, title=f'{row["sample"]}-{row["scan"]} y:{row["y"]} z:{row["zoom"]}')
        img = row['img']
        plt.imshow(img)
        plt.scatter(img.shape[1] / 2, img.shape[0] / 2, c='red', marker='x')
