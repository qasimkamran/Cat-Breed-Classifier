"""
This module contains the tensorflow model class.
"""

from keras import layers, models


class SimpleCNN():
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
        self.model._name = "SimpleCNN"
        self.model.summary()


if '__main__' == __name__:
    simple_cnn = SimpleCNN().model
