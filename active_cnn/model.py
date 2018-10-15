import keras


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
            keras.layers.Dense(1, activation='sigmoid')
            ])
        self.cnn.compile(loss='binary_crossentropy', optimizer='adam')
        self.callback = keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=10e-5, patience=50
                )

    def train(self, X, y):
        self.cnn.fit(
                X.reshape(-1, 1, 140, 1), y,
                epochs=1000, callbacks=[self.callback]
                )

    def predict(self, X):
        return self.cnn.predict(X.reshape(-1, 1, 140, 1)).reshape(-1)

    def save(self, filepath):
        self.cnn.save(filepath)
