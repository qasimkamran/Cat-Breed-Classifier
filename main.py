"""
This module contains the main function.
"""
import os

import data
import model
import processor


def main():
    dataset_id = "cats_vs_dogs"
    custom_tfds = data.CustomTFDS(dataset_id)

    simple_cnn_processor = processor.SimpleCNNProcessor(custom_tfds)
    simple_cnn_processor.preprocess_ds(32)
    train_ds, val_ds, test_ds = custom_tfds.get_train_val_test_ds(0.2, 0.2)

    simple_cnn = model.SimpleCNN().model
    simple_cnn.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy'])

    simple_cnn.fit(train_ds, validation_data=val_ds, epochs=10)
    simple_cnn.evaluate(test_ds)


if '__main__' == __name__:
    main()
