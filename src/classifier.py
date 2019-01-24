from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, Flatten
from keras.utils import multi_gpu_model


def build_model2():
    nb_filters = 8
    model = Sequential()
    model.add(Conv2D(filters=nb_filters, kernel_size=5, activation='relu', input_shape=(128,128,1)))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, 5))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, 5))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(nb_filters*2, 5))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters*2, 5))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters*2, 5))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters*2, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    model = multi_gpu_model(model, gpus=4)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
