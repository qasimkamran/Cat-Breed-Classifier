"""
This module contains the tensorflow model class.
"""

from keras import layers, models


def CustomModel():
    def __init__(self):
        pass

    def train(self, train_ds, val_ds, epochs=10):
        pass

    def test(self, test_ds):
        pass

    def validate(self, val_ds):
        pass

    def predict(self, test_ds):
        pass


class SimpleCNN(CustomModel):
    def __init__(self):
        input_tensor = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        output_tensor = layers.Dense(1, activation='sigmoid')(x)
        self.model = models.Model(input_tensor, output_tensor)
        self.model.summary()

    def train(self, train_ds, val_ds, epochs=10):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    def test(self, test_ds):
        self.model.evaluate(test_ds)

    def validate(self, val_ds):
        self.model.evaluate(val_ds)

    def predict(self, test_ds):
        self.model.predict(test_ds)


if '__main__' == __name__:
    simple_cnn = SimpleCNN().model
