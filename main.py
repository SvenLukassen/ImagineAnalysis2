import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import numpy as np

def data_parameters(train_dir, validate_dir, test_dir):
    # Define the image size and batch size
    IMG_SIZE = (256, 256)
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
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    return model


def run_model(model, train_generator, validate_generator, test_generator):
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    epochs = 20
    # Train the model
    history = model.fit(train_generator,
                        validation_data=validate_generator,
                        epochs=epochs)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print(f"Test accuracy: {test_acc}")

    # model_dir = 'trainedModels/'

    # Save the model to disk
    # tf.saved_model.save(model, model_dir)
    #
    # Make predictions on the test data
    # y_pred = model.predict(test_generator)

    # Convert the test data labels to one-hot encoded format
    # y_true = np.array(test_generator.classes)
    # y_true = tf.keras.utils.to_categorical(y_true,
    # num_classes=test_generator.num_classes)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

def main():
    train_dir = r'./beans_data/test'
    validate_dir = r'./beans_data/train'
    test_dir = r'./beans_data/validation'
    train_generator, validate_generator, test_generator = data_parameters(train_dir, validate_dir, test_dir)
    model = model_parameters()
    run_model(model, train_generator, validate_generator, test_generator)


main()