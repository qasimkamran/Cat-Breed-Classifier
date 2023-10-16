import random
import matplotlib.pyplot as plt
import numpy as np
import subprocess

from keras_preprocessing.image import ImageDataGenerator


DATA_DIR = "C:\\Users\\qasim\\Downloads\\cat_breed_dataset\\images"

DATA_GEN = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    horizontal_flip=True
)

TARGET_SIZE = (300, 400)

train_gen = DATA_GEN.flow_from_directory(
    DATA_DIR,
    TARGET_SIZE,
    batch_size=16,
    class_mode='categorical'
)

class_ids = list(train_gen.class_indices.keys())


def plot_class_batch(input_class_id: str = 'Random') -> None:
    target_class_id = None
    target_class_index = None

    # Handle setting target_class_id and retrieving relevant target_class_index
    if input_class_id == 'Random':
        class_index = random.randint(0, (len(class_ids) - 1))
        target_class_id = class_ids[class_index]
    if not target_class_id:
        target_class_id = input_class_id
    for class_index, class_id in enumerate(class_ids):
        if class_id == target_class_id:
            target_class_index = class_index

    if not target_class_index:
        print('Could not find class id "' + input_class_id + '" in dataset')
        raise Exception

    batch = next(train_gen)
    images, labels = batch  # shape: (batch_size, num_of_classes)
    num_samples = images.shape[0]
    class_indices = np.where(labels[:, target_class_index] == 1)[0]
    plt.figure(figsize=(8, 4))
    for i in range(1, num_samples):
        plt.subplot(4, int(num_samples/4), i)
        plt.imshow(images[class_indices[i]])
        plt.title(target_class_id)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if '__main__' == __name__:
    plot_class_batch('Tuxedo')
