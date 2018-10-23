import keras
import h5py


class CNN:
    def __init__(self):
        # input's height is 1, width is 140 and depth is 1
        self.cnn = keras.models.Sequential([
            keras.layers.Conv2D(64, (1, 3), activation='relu',
                input_shape=(1, 140, 1)),
            keras.layers.Conv2D(64, (1, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(1, 2)),
            keras.layers.Conv2D(128, (1, 3), activation='relu'),
            keras.layers.Conv2D(128, (1, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(1, 2)),
            keras.layers.Conv2D(256, (1, 3), activation='relu'),
            keras.layers.Conv2D(256, (1, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(1, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            # 3 neurouns in ouput layer
            keras.layers.Dense(3, activation='sigmoid')
            ])
        self.cnn.compile(loss='categorical_crossentropy', optimizer='adam')
        self.callback = keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=10e-4, patience=10,
                restore_best_weights=True
                )

    def train(self, X, y):
        self.cnn.fit(
                X.reshape(-1, 1, 140, 1), y,
                batch_size=64,
                epochs=1000, callbacks=[self.callback]
                )

    def predict(self, X):
        return self.cnn.predict(X.reshape(-1, 1, 140, 1)).reshape(-1)

    def save(self, filepath):
        self.cnn.save(filepath)


def get_training_data(it, hdf5):
    with h5py.File(hdf5, 'r+') as f:
        it_gr = f['iteration_{:02}'.format(it)]
        X_tr = it_gr['X_train'][...]
        y_tr = it_gr['y_train'][...]
        X = it_gr['X'][...]
    return X_tr, y_tr, X


def save_predictions(it, hdf5, y_pred):
    with h5py.File(hdf5, 'r+') as f:
        it_gr = f['iteration_{:02}'.format(it)]
        # probabilities
        dt = it_gr.create_dataset('y_pred', y_pred.shape, y_pred.dtype)
        dt[...] = y_pred
        # TODO labels


def learning(it, hdf5):
    X_tr, y_tr, X = get_training_data(it, hdf5)
    X_tr_bal, y_tr_bal = balance(X_tr, y_tr)
    # build the model and train it
    cnn = model.CNN()
    cnn.train(X_tr_bal, y_tr_bal)
    # save network's weights
    cnn.save('data/cnn-bu-it-{:02}.hdf5'.format(it))
    # get and save predictions
    y_pred = cnn.predict(X)
    save_predictions(it, hdf5, y_pred)


if __name__ == '__main__':
    import sys
    learning(int(sys.argv[1]), sys.argv[2])
