import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import ResizeMethod


def data_parameters(train_dir, validate_dir, test_dir):
    """

    :param train_dir: Training data directory ~ string
    :param validate_dir: Validation data directory ~ string
    :param test_dir: Test data directory ~ string
    :return: train_generator, validation_generator, test_generator
    loaded datasets with parameters
    """
    img_height = 256
    img_width = 256
    BATCH_SIZE = 32

    # train_datagen = ImageDataGenerator(rescale=1. / 255,
    #                                    horizontal_flip=True,
    #                                    vertical_flip=True,
    #                                    rotation_range=90,
    #                                    zoom_range=[0.8, 1.2],
    #                                    width_shift_range=0.1,
    #                                    height_shift_range=0.1,
    #                                    shear_range=0.1)
    # validation_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Loads the training dataset with certain parameters
    train_generator = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        batch_size=BATCH_SIZE,
        image_size=(img_height, img_width))
    # Loads the validation dataset with certain parameters

    validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
        validate_dir,
        seed=123,
        batch_size=BATCH_SIZE, image_size=(img_height, img_width))

    # Loads the test dataset with certain parameters
    test_generator = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=123,
        batch_size=BATCH_SIZE,
        image_size=(img_height, img_width))

    return train_generator, validation_generator, test_generator


def model_parameters():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    return model


def run_model(model, train_generator, validation_generator, test_generator):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False),
                  metrics=['accuracy'])

    epochs = 15
    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=validation_generator)

    # Save the model to disk
    tf.saved_model.save(model, "trainedmodel/")

    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print(f"Test accuracy: {test_acc}")


if __name__ == '__main__':
    train_dir = r'./beans_data/train'
    validate_dir = r'./beans_data/validation'
    test_dir = r'./beans_data/test'
    train_generator, validation_generator, test_generator = data_parameters(
        train_dir, validate_dir, test_dir)
    model = model_parameters()
    run_model(model, train_generator, validation_generator, test_generator)
