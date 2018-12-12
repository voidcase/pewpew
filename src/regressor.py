from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from dataset.compile import get_dataset_df, get_dataset

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


if __name__ == '__main__':
    df = get_dataset_df(Path('/data/staff/common/ML-crystals/csv/data_0.5.csv'))
    train_df = df[df['y'] > 100]
    train, valid, test = get_dataset(train_df, INPUT_SHAPE)
    model = build_model()
    model.fit(train['x'], train['y'], epochs=3, batch_size=32, validation_data=(valid['x'], valid['y']))
    score = model.evaluate(test['x'], test['y'], batch_size=32)
