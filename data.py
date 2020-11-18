"""
Author:Junzheng Wu
Email: jwu220@uottawa.ca
Student ID: 300084962
Date:2020/11/18
Note: Use sparse_category_crossentropy rather than one-hot encoder for saving memory.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def get_mnist_data(split=0.5, verbose=False):
    """
    Fetch the MNIST dataset in tf.dataset format 
    (after simple preprocessed including normalization, shuffling, gray to rgb, resizing)
    :param split: the split rate of victim data and attack data
    :param verbose: if true, show some examples
    :return: tfds for victim, attack and test
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    x_train = x_train / 255.
    x_test = x_test / 255.

    index = np.random.permutation(x_train.shape[0])
    x_train = x_train[index]
    y_train = y_train[index]

    x_train = tf.convert_to_tensor(x_train)
    x_train = tf.image.grayscale_to_rgb(x_train)
    x_test = tf.convert_to_tensor(x_test)
    x_test = tf.image.grayscale_to_rgb(x_test)

    data_victim = tf.data.Dataset.from_tensor_slices((x_train[: int(x_train.shape[0] * split)],
                                                      y_train[: int(x_train.shape[0] * split)]))
    data_victim = data_victim.map(lambda x, y: (tf.image.resize(x, (32, 32)), y))

    data_attack= tf.data.Dataset.from_tensor_slices((x_train[int(x_train.shape[0] * split):],
                                                      y_train[int(x_train.shape[0] * split):]))
    data_attack = data_attack.map(lambda x, y: (tf.image.resize(x, (32, 32)), y))

    data_test= tf.data.Dataset.from_tensor_slices((x_test, y_test))
    data_test = data_test.map(lambda x, y: (tf.image.resize(x, (32, 32)), y))

    if verbose:
        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(data_victim.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(data_attack.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(data_test.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")
        plt.show()
    return data_victim, data_attack, data_test

def get_cifar_data(split=0.5, verbose=False):
    """
    Fetch the CIFAR10 dataset in tf.dataset format
    (after simple preprocessed including normalization, shuffling)
    :param split: the split rate of victim data and attack data
    :param verbose: if true, show some examples
    :return: tfds for victim, attack and test
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.

    index = np.random.permutation(x_train.shape[0])
    x_train = x_train[index]
    y_train = y_train[index]

    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)

    data_victim = tf.data.Dataset.from_tensor_slices((x_train[: int(x_train.shape[0] * split)],
                                                      y_train[: int(x_train.shape[0] * split)]))

    data_attack= tf.data.Dataset.from_tensor_slices((x_train[int(x_train.shape[0] * split):],
                                                      y_train[int(x_train.shape[0] * split):]))

    data_test= tf.data.Dataset.from_tensor_slices((x_test, y_test))

    if verbose:
        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(data_victim.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(data_attack.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(data_test.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")
        plt.show()
    return data_victim, data_attack, data_test

if __name__ == '__main__':
    get_mnist_data(verbose=True)
    get_cifar_data(verbose=True)