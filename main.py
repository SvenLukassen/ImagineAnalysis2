import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def data_parameters(train_dir, validate_dir, test_dir):
    # Define the image size and batch size
    IMG_SIZE = (96, 96)
    BATCH_SIZE = 32

    # Define the data generators
    train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        rotation_range=90,
                                        zoom_range=[0.8, 1.2],
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        shear_range=0.1)
    validate_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                  seed=123,
                                                                  image_size=IMG_SIZE,
                                                                  batch_size=BATCH_SIZE)

    validate_generator = tf.keras.utils.image_dataset_from_directory(validate_dir,
                                                                     seed=123,
                                                                     image_size=IMG_SIZE,
                                                                     batch_size=BATCH_SIZE)

    test_generator = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                 seed=123,
                                                                 image_size=IMG_SIZE,
                                                                 batch_size=BATCH_SIZE)

    return train_generator, validate_generator, test_generator


def model_parameters():
    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    return model

def main():
    train_dir = 'C:/Users/swans/Desktop/training'
    validate_dir = 'C:/Users/swans/Desktop/validatie'
    test_dir = 'C:/Users/swans/Desktop/test'
    train_generator, validate_generator, test_generator = data_parameters(train_dir, validate_dir, test_dir)
    model = model_parameters()


main()