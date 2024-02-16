"""
This module contains test cases for test_data.py
"""

import os
import data


def test_custom_tfds_init():
    os.chdir("..")
    dataset_id = "cats_vs_dogs"
    custom_tfds = data.CustomTFDS(dataset_id)
    assert custom_tfds.SHARDS_DIR == ".\\shards"
    assert custom_tfds.ds is not None
    assert custom_tfds.ds_info is not None
    assert custom_tfds.average_image_size != 0


def test_tfds_utility_init():
    os.chdir("..")
    dataset_id = "cats_vs_dogs"
    custom_tfds = data.CustomTFDS(dataset_id)
    tfds_utility = data.TFDSUtility(custom_tfds)
    assert tfds_utility.THRESHOLD == 0.2


def test_tfds_utility_create_and_add_example():
    os.chdir("..")
    dataset_id = "cats_vs_dogs"
    custom_tfds = data.CustomTFDS(dataset_id)
    tfds_utility = data.TFDSUtility(custom_tfds)
    example = tfds_utility.create_example("NG_Cat.png", 0)
    tfds_utility.add_example_to_ds(example)
    tfds_utility.view_last_shard()
    assert os.path.exists(".\\shards\\shard_0")

def test_tfds_utility_remove_first_example():
    os.chdir("..")
    dataset_id = "cats_vs_dogs"
    custom_tfds = data.CustomTFDS(dataset_id)
    tfds_utility = data.TFDSUtility(custom_tfds)
    first_image = tfds_utility.get_np_batch(1)
    tfds_utility.remove_example_at_index(0)
    assert tfds_utility.get_np_batch(1) != first_image
