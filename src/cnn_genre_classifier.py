import json

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "../data/data.json"


def load_data(data_path):
    """
    Loads the data from the data_path.

    Args:
        data_path: The path to the data.

    Returns:
        X: (ndarray) The input data.
        y: (ndarray) The targets.
    """

    with open(data_path, "r") as f:
        data = json.load(f)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def prepare_datasets(test_size, validation_size):
    """
    Prepares the datasets for training.

    Args:
        test_size: The size of the test set.
        validation_size: The size of the validation set.

    Returns:
        X_train: (ndarray) The training input data.
        y_train: (ndarray) The training targets.
        X_test: (ndarray) The test input data.
        y_test: (ndarray) The test targets.
        X_validation: (ndarray) The validation input data.
        y_validation: (ndarray) The validation targets.
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """
    Builds the model.

    Args:
        input_shape: The shape of the input data.

    Returns:
        model: The model.
    """

    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(
        keras.layers.Conv2D(32, (3, 3),
                            activation='relu',
                            input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                        padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                        padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2),
                                        padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y):
    X = X[np.newaxis, ...]

    # prediction = [[0.1,0.2, ...]]
    prediction = model.predict(X)  # X -> (1, 130, 13, 1)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)  # [4]

    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


if __name__ == "__main__":

    # create train, validation and test sets
    X_train, X_validation, X_test, \
        y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build the CNN
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # train the CNN
    model.fit(X_train,
              y_train,
              validation_data=(X_validation, y_validation),
              batch_size=32,
              epochs=30)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on Test set is {}".format(test_accuracy))

    # make predictions on a sample
    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)
