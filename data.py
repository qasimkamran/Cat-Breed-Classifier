import cv2
from tensorflow_datasets.core.dataset_utils import _IterableDataset


def handle_tfds_protobuf_winerror(shuffle_py_dir):
    # Fix windows error for not including resource
    shuffle_py = shuffle_py_dir
    with open(shuffle_py, 'r') as file:
        lines = file.readlines()

    line_number_to_edit = 20  # import resource
    new_content = ""
    if 0 < line_number_to_edit <= len(lines):
        lines[line_number_to_edit - 1] = new_content  # Subtract 1 because list index starts from 0
    line_number_to_edit = 73  # """Attempts to increase the maximum number of open file descriptors."""
    new_content = "  pass"
    if 0 < line_number_to_edit <= len(lines):
        lines[line_number_to_edit - 1] = new_content + '\n'  # Subtract 1 because list index starts from 0

    with open(shuffle_py, 'w') as file:
        file.writelines(lines)


try:
    import tensorflow_datasets as tfds
except ImportError:
    handle_tfds_protobuf_winerror(
        'D:\\Projects\\Cats-v-Dogs-Classifier\\venv\\lib\\site-packages\\tensorflow_datasets\\core\\shuffle.py')
    print("Import error handled")
    print("Verify shuffle.py contents and re-run data.py")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import os
import re


def get_average_dir_file_size(directory, exclude_list=None, threshold=sys.float_info.max):
    files = []
    file_sizes = []
    for file in os.listdir(directory):
        if (exclude_list and file in exclude_list) or not file.__contains__('tfrecord'):
            continue
        files.append(file)
    for i, file in enumerate(files):
        filepath = os.path.join(directory, file)
        size = os.path.getsize(filepath)
        if (i + 1) != len(files):
            next_filepath = os.path.join(directory, files[i + 1])
            next_size = os.path.getsize(next_filepath)
            if abs(size - next_size) <= threshold:
                file_sizes.append(size)
        else:
            previous_filepath = os.path.join(directory, files[i - 1])
            previous_size = os.path.getsize(previous_filepath)
            if abs(size - previous_size) <= threshold:
                file_sizes.append(size)
    if not file_sizes:
        return -1
    average_dir_file_size = sum(file_sizes) / len(file_sizes)
    return get_bytes_to_megabytes(average_dir_file_size)


def extract_set_name_and_extension(file_path):
    filename = os.path.basename(file_path)
    name, extension = os.path.splitext(filename)
    set_name = name.split('.')[0]
    return set_name, extension


def get_bytes_to_megabytes(bytes_size):
    return bytes_size / (1024 * 1024)


def rename_tfrecords(file_path, total_records):
    directory, _ = os.path.split(file_path)
    set_name, extension = extract_set_name_and_extension(file_path)
    pattern = r'\-(\d+)-of-\d+'
    match = re.search(pattern, extension)
    if match:
        index = int(match.group(1))
    else:
        return None
    new_filename = f"{set_name}.tfrecord-{index:05d}-of-{total_records:05d}"
    new_file_path = os.path.join(directory, new_filename)
    os.rename(file_path, new_file_path)
    print(f"File {file_path} renamed to {new_file_path}")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def is_binary(num):
    try:
        result = int(num)
        return result == 0 or result == 1
    except (ValueError, TypeError):
        return False


def get_length_of_iterable(iterable):
    count = 0
    for _ in iterable:
        count += 1
    return count


class TfdsUtility:
    ds = None
    ds_info = None

    average_image_size = 0

    SHARDS_DIR = ".\\shards"

    def __init__(self, dataset_id):
        self.load_data(dataset_id)

    def load_data(self, dataset_id):
        data_dir = os.path.join(os.curdir, 'dataset')
        self.ds, self.ds_info = tfds.load(
            name=dataset_id,
            with_info=True,
            as_supervised=True,
            data_dir=data_dir
        )
        self.ds = self.ds['train']
        print("Loading all shards...")
        self.load_all_shards()
        print(f"{dataset_id} dataset loaded")
        print("Calculating size averages...")
        self.calculate_size_averages()
        print("Size averages calculated")

    def calculate_size_averages(self):
        num_examples = self.ds.cardinality().numpy()
        np_batch = self.get_np_batch(num_examples)
        bytes_total = 0
        for image, _ in np_batch:
            bytes_total += image.nbytes
        self.average_image_size = get_bytes_to_megabytes(bytes_total / num_examples)

    def get_np_batch(self, batch_size):
        batch = self.ds.take(batch_size)
        if not batch:
            raise Exception("Error taking batch from tensorflow dataset object")
        return tfds.as_numpy(batch)

    def get_tfds_batch(self, batch_size):
        return self.ds.take(batch_size)

    def map_label(self, label):
        return self.ds_info.features['label'].int2str(label)

    def view_batch(self, np_batch):  # View batch in equal cols and rows
        num_examples = get_length_of_iterable(np_batch)
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
        threshold = 0.2
        image_dir = os.path.join("tmp", image_id)
        if not os.path.exists(image_dir):
            raise Exception(f"Image {image_id} does not exist in tmp directory")
        image = cv2.imread(image_dir)
        image_size = get_bytes_to_megabytes(image.nbytes)
        if image_size > (0.2 * self.average_image_size + self.average_image_size):
            x = round(image_size - self.average_image_size, 2)
            raise Exception(f"Image is {x}mb larger than average image size")
        image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
        image_tensor.set_shape([None, None, 3])
        label_tensor = tf.convert_to_tensor(label, dtype=tf.int64)
        return image_tensor, label_tensor

    def create_shard_id(self):
        return len(os.listdir(self.SHARDS_DIR))

    def add_example_to_ds(self, example):
        new_set = tf.data.Dataset.from_tensors(example)
        self.ds = self.ds.concatenate(new_set)
        shard_id = self.create_shard_id()
        shard_dir = os.path.join(self.SHARDS_DIR, f"shard_{shard_id}")
        new_set.save(shard_dir)

    def view_last_shard(self):
        shard_id = len(os.listdir(self.SHARDS_DIR)) - 1
        shard_dir = os.path.join(self.SHARDS_DIR, f"shard_{shard_id}")
        shard = tf.data.Dataset.load(shard_dir)
        self.view_batch(tfds.as_numpy(shard))

    def load_all_shards(self):
        for dir in os.listdir(self.SHARDS_DIR):
            dir_path = os.path.join(self.SHARDS_DIR, dir)
            self.ds = self.ds.concatenate(tf.data.Dataset.load(dir_path))

    def get_image_label_to_remove(self, index):
        batch = self.get_tfds_batch(index + 1)
        image, label = batch.skip(index).take(1).as_numpy_iterator().next()
        return image, label

    def filter_func(self, image, label, image_to_remove, label_to_remove):
        exclude_tensor_image = tf.convert_to_tensor(image_to_remove, dtype=tf.uint8)
        exclude_tensor_label = tf.convert_to_tensor(label_to_remove, dtype=tf.int64)
        image_shape = tf.shape(image)
        exclude_tensor_image_shape = tf.shape(exclude_tensor_image)
        if tf.reduce_all(tf.equal(image_shape, exclude_tensor_image_shape)):
            return tf.math.reduce_all(tf.not_equal(image, exclude_tensor_image)) and tf.reduce_all(tf.not_equal(label, exclude_tensor_label))
        return True

    def remove_example_at_index(self, index):
        image_to_remove, label_to_remove = self.get_image_label_to_remove(index)
        self.ds = self.ds.filter(lambda image, label: self.filter_func(image, label, image_to_remove, label_to_remove))


if '__main__' == __name__:
    tfds_utility = TfdsUtility(dataset_id='cats_vs_dogs')

    # image_id = 'NG_Cat.png'
    # label = 0
    # example = tfds_utility.create_example(image_id, label)
    #
    # tfds_utility.add_example_to_ds(example)

    tfds_utility.remove_example_at_index(1)

    tfds_utility.view_batch(tfds_utility.get_np_batch(2))
