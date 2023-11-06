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


class TFDataset:
    tfds_train_dataset = None
    tfds_test_dataset = None
    tfds_dataset_info = None

    def __init__(self, dataset_id):
        self.load_data(dataset_id)

    def load_data(self, dataset_id):
        data_dir = os.path.join(os.curdir, 'dataset')
        (self.tfds_train_dataset, self.tfds_test_dataset), self.tfds_dataset_info = tfds.load(
            name=dataset_id,
            split=['train[:80%]', 'train[80%:]'],
            with_info=True,
            as_supervised=True,
            data_dir=data_dir
        )
        print(self.tfds_dataset_info)
        print(f"{dataset_id} dataset loaded")

    def get_batch(self, batch_size, set_id):
        batch = None
        if set_id != 'train' and set_id != 'test':
            raise Exception(f"invalid set_id - {set_id}")
        if self.tfds_train_dataset and set_id == 'train':
            batch = self.tfds_train_dataset.take(batch_size)
        if self.tfds_test_dataset and set_id == 'test':
            batch = self.tfds_test_dataset.take(batch_size)
        if not batch:
            raise Exception("Error taking batch from tensorflow dataset object")
        return tfds.as_numpy(batch)  # Saved for expansion

    def map_label(self, label):
        return self.tfds_dataset_info.features['label'].int2str(label)

    def view_batch(self, np_batch):
        images, labels = [], []
        for image, label in np_batch:
            images.append(image)
            labels.append(self.map_label(label))
        num_images = len(images)
        num_cols = int(np.sqrt(num_images))
        num_rows = int(np.ceil(num_images / num_cols))
        figure, axes = plt.subplots(num_cols, num_rows, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(num_images):
            if i < num_images:
                axis = axes[i]
                axis.imshow(images[i])
                axis.set_title(labels[i])
                axis.axis('off')
            else:
                axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    def create_tfrecord(self):
        tfrecords = []
        threshold_size = 10  # In megabytes
        new_required = False
        data_dir = self.tfds_dataset_info.data_dir
        average_file_size = get_average_dir_file_size(data_dir)

        for file in os.listdir(data_dir):
            if file.__contains__('tfrecord'):
                tfrecord_filepath = os.path.join(data_dir, file)
                tfrecords.append(tfrecord_filepath)

        last_size = get_bytes_to_megabytes(os.path.getsize(tfrecords[-2]))
        if abs(last_size - average_file_size) <= threshold_size:
            new_required = True

        if new_required:
            new_total = len(tfrecords) + 1
            for tfrecord in tfrecords:
                rename_tfrecords(tfrecord, new_total)
            current_index = len(tfrecords)
            set_name, extension = extract_set_name_and_extension(tfrecords[0])
            tfrecord_filepath = os.path.join(data_dir, f"{set_name}.tfrecord-{current_index:05d}-of-{new_total:05d}")
            tf.io.TFRecordWriter(tfrecord_filepath)
            print(f"New tfrecord {tfrecord_filepath} created")
        else:
            print("New tfrecord not required")


if '__main__' == __name__:
    dataset = TFDataset(dataset_id='cats_vs_dogs')
    dataset.create_tfrecord()
    batch = dataset.get_batch(16, 'train')
    dataset.view_batch(batch)
