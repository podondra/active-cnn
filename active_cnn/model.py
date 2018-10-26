from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.utils import to_categorical
import h5py


def get_model():
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(140, 1)),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(256, 3, activation='relu'),
        Conv1D(256, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
        ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def train(model, X, y):
    one_hot_y = to_categorical(y, num_classes=3)
    callback = EarlyStopping(
            monitor='loss', min_delta=10e-4, patience=10,
            restore_best_weights=True
            )
    model.fit(
            X.reshape(-1, 140, 1), one_hot_y, batch_size=64, epochs=1000,
            callbacks=[callback]
            )
