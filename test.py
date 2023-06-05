import tensorflow as tf
import matplotlib.pyplot as plt


def data_parameters(train_dir, validate_dir, test_dir):
    """Takes all the data from a directory and puts it into
    a generator.

    :param train_dir: Training data directory ~ string
    :param validate_dir: Validation data directory ~ string
    :param test_dir: Test data directory ~ string
    :return: train_generator, validation_generator, test_generator
    loaded datasets with parameters
    """
    img_height = 256
    img_width = 256
    BATCH_SIZE = 32

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
    """A model will be created with parameters. With Conv2D, MaxPooling2D,
    Dropout, Flatten and Dense

    :return: model ~ parameters of the model
    """
    # Keras model with multiple layers
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model


def run_model(model, train_generator, validation_generator, test_generator):
    """Compiles the model and runs the model with the train and validation data.
    The history of the model will be saved and can be runned again. The model
    will be evaluated with test data and shows the accuracy of th model.

    :param model: parameters of the model
    :param train_generator: loaded train dataset with parameters
    :param validation_generator: loaded validation dataset with parameters
    :param test_generator: loaded generator dataset with parameters
    :return:
    """
    # Compiles the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False),
                  metrics=['accuracy'])

    # Number of times that the model will be fitted
    epochs = 15
    # Fits the model with training data and validation data
    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=validation_generator)

    # Saves the model to the directory trainedmodel
    tf.saved_model.save(model, "trainedmodel/")

    # Evaluates the model
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    # Shows the test accuracy
    print(f"Test accuracy: {test_acc}")

    return epochs, history

def visualize_training_results(epochs, history):
    """Makes a plot with the train and validation accuracy and
    train and validation loss of the fitted model

    :param epochs: Number of times the model will be fit ~ int
    :param history: History of the fitted model
    :return: A plot with accuracy and loss
    """
    # Accuracy of the training data
    acc = history.history['accuracy']
    # Accuracy of the validation data
    val_acc = history.history['val_accuracy']

    # Loss of the training data
    loss = history.history['loss']
    # Loss of the validation data
    val_loss = history.history['val_loss']

    # Range of the epochs
    epochs_range = range(epochs)

    # Size of the figure
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    # Line for training accuracy
    plt.plot(epochs_range, acc, label='Training Accuracy')
    # Line for validation accuracy
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    # Line for training loss
    plt.plot(epochs_range, loss, label='Training Loss')
    # Line for validation loss
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    train_dir = r'./beans_data/train'
    validate_dir = r'./beans_data/validation'
    test_dir = r'./beans_data/test'
    train_generator, validation_generator, test_generator = data_parameters(
        train_dir, validate_dir, test_dir)
    model = model_parameters()
    epochs, history = run_model(model, train_generator, validation_generator, test_generator)
    visualize_training_results(epochs, history)
