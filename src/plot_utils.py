import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import History


def ceil(n, base):
    """Return closest multiple of base with ceiling rounding"""
    return int(base * np.ceil(float(n) / base))


def imshow(axes, img, title='', mark_center=True, param_dict=None):
    if param_dict is None:
        param_dict = {}

    shape = img.shape
    if len(shape) == 3 and shape[2] == 1:
        img = np.reshape(img, shape[:-1])

    axes.imshow(img, **param_dict)
    if mark_center:
        axes.scatter(img.shape[1] / 2, img.shape[0] / 2, c='red', marker='x')
    axes.set_title(title)


def image_grid(images: list, titles: list = None, max_cols=4):
    if titles is None:
        titles = [''] * len(images)
    else:
        assert len(images) == len(titles)

    cols = min(len(images), max_cols)
    rows = ceil(len(images), max_cols) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows))
    axes = axes.flatten()
    l = min(len(images), rows * cols)
    for i in range(l):
        imshow(axes[i], images[i], titles[i])
    return fig


def show_some(data: pd.DataFrame, seed=None, **kwargs):
    df_sample = data.sample(n=min(9, len(data)), random_state=seed)
    zoom = df_sample['zoom'] if 'zoom' in df_sample.columns else '-'
    titles = df_sample['sample'] + '-' + df_sample['scan'] + ' y:' + df_sample['y'].map(str) + ' z:' + zoom.map(str)
    return image_grid(df_sample['img'].values, titles.values)


def plot_history(history: History):
    fig, (row1, row2) = plt.subplots(2, 2, figsize=(12, 6))
    row1[0].plot(history.history['acc'])
    row1[0].set_title('acc')
    row1[1].plot(history.history['val_acc'])
    row1[1].set_title('val_acc')
    row2[0].plot(history.history['loss'])
    row2[0].set_title('loss')
    row2[1].plot(history.history['val_loss'])
    row2[1].set_title('val_loss')


def plot_from_generator(gen, y=None, **kwargs):
    x_batch, y_batch = next(gen)
    if y is not None:
        y_batch = y
    images = [i for i in x_batch]
    image_grid(x_batch, y_batch, **kwargs)


def plot_confusion_matrix(ground_truth: list, predictions: list):
    assert len(ground_truth) == len(predictions)
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(ground_truth, predictions)
    labels = sorted(set(ground_truth))
    data = {}
    for i, label in enumerate(labels):
        data[label] = matrix[:, i]
    df = pd.DataFrame(data, index=labels)
    axes = sns.heatmap(df, annot=True, fmt='d')
    axes.set_ylabel('predicted', fontsize='large')
    axes.set_xlabel('actual', fontsize='x-large')
    return axes

