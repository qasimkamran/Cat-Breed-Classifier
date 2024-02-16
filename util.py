"""
This module contains utility helper functions.
"""

import os
import re
import sys
import tensorflow as tf


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


def print_and_log(message):
    print(message)
    # TODO: fix logging
    # logging.info(message)
