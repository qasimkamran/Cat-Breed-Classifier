"""
This module contains the data processors per model.
"""

import tensorflow as tf
from data import TFDSUtility


class SimpleCNNProcessor(TFDSUtility):
    def __init__(self, custom_tfds):
        super().__init__(custom_tfds)

    def preprocess(self, image, label):
        image = tf.image.resize(image, (28, 28))
        image = tf.image.rgb_to_grayscale(image)
        image = image / 255.0
        return image, label

    def preprocess_ds(self, batch_size):
        self.custom_tfds.ds = self.custom_tfds.ds.map(self.preprocess).batch(batch_size)
