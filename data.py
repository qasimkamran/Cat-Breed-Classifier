"""
This module contains dataset dataclass and a utility class to perform operations on it.
"""

import dataclasses
import cv2
import logging
import util

logging.basicConfig(filename='data.log', encoding='utf-8', level=logging.INFO)
try:
    import tensorflow_datasets as tfds
except ImportError:
    util.handle_tfds_protobuf_winerror(
        'D:\\Projects\\Cats-v-Dogs-Classifier\\venv\\lib\\site-packages\\tensorflow_datasets\\core\\shuffle.py')
    print("Import error handled")
    print("Verify shuffle.py contents and re-run test_data.py")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import os
import re

@dataclasses.dataclass
class CustomTFDS:
    ds = None
    ds_info = None
    average_image_size = 0
    SHARDS_DIR = ".\\shards"

    def __init__(self, dataset_id):
        self.load(dataset_id)

    def load(self, dataset_id):
        self.load_dataset(dataset_id)
        self.load_all_shards()
        self.calculate_size_averages()

    def load_dataset(self, dataset_id):
        util.print_and_log(f"Loading dataset {dataset_id}...")
        data_dir = os.path.join(os.curdir, 'dataset')
        self.ds, self.ds_info = tfds.load(
            name=dataset_id,
            with_info=True,
            as_supervised=True,
            data_dir=data_dir
        )
        self.ds = self.ds['train']
        util.print_and_log(f"Dataset {dataset_id} loaded")

    def get_np_batch(self, batch_size):
        batch = self.ds.take(batch_size)
        if not batch:
            raise Exception("Error taking batch from tensorflow dataset object")
        return tfds.as_numpy(batch)

    def calculate_size_averages(self):
        util.print_and_log("Calculating size averages...")
        num_examples = self.ds.cardinality().numpy()
        np_batch = self.get_np_batch(num_examples)
        bytes_total = 0
        for image, _ in np_batch:
            bytes_total += image.nbytes
        self.average_image_size = util.get_bytes_to_megabytes(bytes_total / num_examples)
        util.print_and_log(f"Average image size: {self.average_image_size}mb")

    def load_all_shards(self):
        util.print_and_log("Loading all shards...")
        if not os.path.exists(self.SHARDS_DIR):
            os.mkdir(self.SHARDS_DIR)
        for dir in os.listdir(self.SHARDS_DIR):
            dir_path = os.path.join(self.SHARDS_DIR, dir)
            self.ds = self.ds.concatenate(tf.data.Dataset.load(dir_path))
        util.print_and_log("All shards loaded")

    def get_train_val_test_ds(self, val_split, test_split):
        num_examples = self.ds.cardinality().numpy()
        val_size = int(num_examples * val_split)
        test_size = int(num_examples * test_split)
        train_size = num_examples - val_size - test_size
        train_ds = self.ds.take(train_size)
        val_ds = self.ds.skip(train_size).take(val_size)
        test_ds = self.ds.skip(train_size + val_size).take(test_size)
        return train_ds, val_ds, test_ds


class TFDSUtility:
    THRESHOLD = 0.2

    def __init__(self, custom_tfds):
        self.custom_tfds = custom_tfds
        if not isinstance(self.custom_tfds, CustomTFDS):
            raise Exception("CustomTFDS object is not instantiated")

    def get_np_batch(self, batch_size):
        batch = self.custom_tfds.ds.take(batch_size)
        if not batch:
            raise Exception("Error taking batch from tensorflow dataset object")
        return tfds.as_numpy(batch)

    def get_tfds_batch(self, batch_size):
        return self.custom_tfds.ds.take(batch_size)

    def map_label(self, label):
        return self.custom_tfds.ds_info.features['label'].int2str(label)

    def view_batch(self, np_batch):  # View batch in equal cols and rows
        num_examples = util.get_length_of_iterable(np_batch)
        cols = int(np.ceil(np.sqrt(num_examples)))
        rows = int(np.ceil(num_examples / cols))
        fig = plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(np_batch):
            label = self.map_label(label)
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(label)
            ax.axis('off')
            ax.grid(False)
            ax.imshow(image)
        fig.tight_layout()
        plt.show()

    def create_example(self, image_id, label):
        image_dir = os.path.join("tmp", image_id)
        if not os.path.exists(image_dir):
            raise Exception(f"Image {image_id} does not exist in tmp directory")
        image = cv2.imread(image_dir)
        image_size = util.get_bytes_to_megabytes(image.nbytes)
        if image_size > (self.THRESHOLD * self.custom_tfds.average_image_size + self.custom_tfds.average_image_size):
            x = round(image_size - self.custom_tfds.average_image_size, 2)
            raise Exception(f"Image is {x}mb larger than average image size")
        image_dtype = self.custom_tfds.ds.element_spec[0].dtype
        label_dtype = self.custom_tfds.ds.element_spec[1].dtype
        image_tensor = tf.convert_to_tensor(image, dtype=image_dtype)
        image_tensor.set_shape([None, None, 3])
        label_tensor = tf.convert_to_tensor(label, dtype=label_dtype)
        return image_tensor, label_tensor

    def create_shard_id(self):
        shards_dir = self.custom_tfds.SHARDS_DIR
        return len(os.listdir(shards_dir))

    def add_example_to_ds(self, example):
        new_set = tf.data.Dataset.from_tensors(example)
        self.custom_tfds.ds = self.custom_tfds.ds.concatenate(new_set)
        shard_id = self.create_shard_id()
        shard_dir = os.path.join(self.custom_tfds.SHARDS_DIR, f"shard_{shard_id}")
        new_set.save(shard_dir)

    def view_last_shard(self):
        shard_id = len(os.listdir(self.custom_tfds.SHARDS_DIR)) - 1
        shard_dir = os.path.join(self.custom_tfds.SHARDS_DIR, f"shard_{shard_id}")
        shard = tf.data.Dataset.load(shard_dir)
        self.view_batch(tfds.as_numpy(shard))

    def get_image_label_to_remove(self, index):
        batch = self.get_tfds_batch(index + 1)
        image, label = batch.skip(index).take(1).as_numpy_iterator().next()
        return image, label

    def index_remove_filter(self, image, label, image_to_remove, label_to_remove):
        image_dtype = self.custom_tfds.ds.element_spec[0].dtype
        label_dtype = self.custom_tfds.ds.element_spec[1].dtype
        exclude_tensor_image = tf.convert_to_tensor(image_to_remove, dtype=image_dtype)
        exclude_tensor_label = tf.convert_to_tensor(label_to_remove, dtype=label_dtype)
        image_shape = tf.shape(image)
        exclude_tensor_image_shape = tf.shape(exclude_tensor_image)
        if tf.reduce_all(tf.equal(image_shape, exclude_tensor_image_shape)):
            return tf.math.reduce_all(tf.not_equal(image, exclude_tensor_image)) and tf.reduce_all(
                tf.not_equal(label, exclude_tensor_label))
        return True

    def remove_example_at_index(self, index):
        image_to_remove, label_to_remove = self.get_image_label_to_remove(index)
        self.custom_tfds.ds = self.custom_tfds.ds.filter(lambda image, label:
                                                         self.index_remove_filter(
                                                             image, label, image_to_remove, label_to_remove))
