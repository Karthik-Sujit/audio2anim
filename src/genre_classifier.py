import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

DATASET_PATH = '../data/data.json'


def load_data(dataset_path):
    """
    Loads the dataset from the given path.
    :param dataset_path: Path to the dataset.
    :return: inputs and targets.
    """

    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    print("Data successfully loaded!")

    return inputs, targets


def plot_history(history):

    # create figure and axes for plotting
    fig, ax = plt.subplots(2)

    # create accuracy plot
    ax[0].plot(history.history['accuracy'], label='train accuracy')
    ax[0].plot(history.history['val_accuracy'], label='test accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='lower right')
    ax[0].set_title('Accuracy eval')

    # create error plot
    ax[1].plot(history.history['loss'], label='train error')
    ax[1].plot(history.history['val_loss'], label='test error')
    ax[1].set_ylabel('Error')
    ax[1].legend(loc='lower right')
    ax[1].set_title('Error eval')

    plt.show()


# load data
inputs, targets = load_data(DATASET_PATH)

# split data into training and testing sets
inputs_train, inputs_test, targets_train, targets_test = train_test_split(
    inputs, targets, test_size=0.3)

# build the network architecture
model = keras.Sequential([
    # input layer
    keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

    # hidden layer #1
    keras.layers.Dense(512,
                       activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    # dropout layer
    keras.layers.Dropout(0.3),

    # hidden layer #2
    keras.layers.Dense(256,
                       activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    # dropout layer
    keras.layers.Dropout(0.3),

    # hidden layer #3
    keras.layers.Dense(64,
                       activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    # dropout layer
    keras.layers.Dropout(0.3),

    # output layer
    keras.layers.Dense(10, activation='softmax')
])

# compile network
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(inputs_train,
                    targets_train,
                    validation_data=(inputs_test, targets_test),
                    epochs=50,
                    batch_size=32)

# plot history
plot_history(history)
