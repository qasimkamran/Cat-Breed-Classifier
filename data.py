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


class TFDataset:
    tfds_train_dataset = None
    tfds_test_dataset = None
    tfds_dataset_info = None

    def __init__(self, dataset_id):
        self.load_data(dataset_id)

    def load_data(self, dataset_id):
        (self.tfds_train_dataset, self.tfds_test_dataset), self.tfds_dataset_info = tfds.load(
            name=dataset_id,
            split=['train[:80%]', 'train[80%:]'],
            with_info=True,
            as_supervised=True,
        )
        print(self.tfds_dataset_info)
        print(f"{dataset_id} dataset loaded")

    def get_data_batch(self, batch_size, set_id):
        batch = None
        images, labels = [], []
        if set_id != 'train' and set_id != 'test':
            raise Exception(f"invalid set_id - {set_id}")
        if self.tfds_train_dataset and set_id == 'train':
            iamges, labels = self.tfds_train_dataset.take(batch_size)
        if self.tfds_test_dataset and set_id == 'test':
            batch = self.tfds_test_dataset.take(batch_size)
        if not batch:
            raise Exception("Error taking batch from tensorflow dataset object")
        print(batch)
        for example in batch:
            images.append(example['image'])
            labels.append(example['label'])
        return images, labels

    def view_data_batch(self, data_batch):
        images, labels = data_batch
        num_images = len(images)
        num_cols = int(np.sqrt(num_images))
        num_rows = int(np.ceil(num_images / num_cols))
        figure, axes = plt.subplots(num_cols, num_rows, figsize=(15, 3))
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


if '__main__' == __name__:
    dataset = TFDataset(dataset_id='cats_vs_dogs')
    images, labels = dataset.get_data_batch(16, 'train')
    dataset.view_data_batch((images, labels))
