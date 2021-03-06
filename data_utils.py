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
import pandas as pd
import os

np.random.seed(1)

def get_mnist_data(split=0.5, verbose=False, target_size=48):
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
    data_victim = data_victim.map(lambda x, y: (tf.image.resize(x, (target_size, target_size)), y))

    data_attack= tf.data.Dataset.from_tensor_slices((x_train[int(x_train.shape[0] * split):],
                                                      y_train[int(x_train.shape[0] * split):]))
    data_attack = data_attack.map(lambda x, y: (tf.image.resize(x, (target_size, target_size)), y))

    data_test= tf.data.Dataset.from_tensor_slices((x_test, y_test))
    data_test = data_test.map(lambda x, y: (tf.image.resize(x, (target_size, target_size)), y))

    if verbose:
        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(data_victim.take(9)):
            print(image.shape)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(data_attack.take(9)):
            print(image.shape)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(data_test.take(9)):
            print(image.shape)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")
        plt.show()
    return data_victim, data_attack, data_test

def get_cifar_data(split=0.5, verbose=False, target_size=48):
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
    data_victim = data_victim.map(lambda x, y: (tf.image.resize(x, (target_size, target_size)), y))

    data_attack= tf.data.Dataset.from_tensor_slices((x_train[int(x_train.shape[0] * split):],
                                                      y_train[int(x_train.shape[0] * split):]))
    data_attack = data_attack.map(lambda x, y: (tf.image.resize(x, (target_size, target_size)), y))


    data_test= tf.data.Dataset.from_tensor_slices((x_test, y_test))
    data_test = data_test.map(lambda x, y: (tf.image.resize(x, (target_size, target_size)), y))

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

def find_the_highest_val_acc(csvfile):
    df = pd.read_csv(csvfile)
    for item in df.columns:
        if "val" in item and "acc" in item:
            break
    values = np.array(df[item])
    return np.max(values)

def present_val_acc_for_all():
    logdir = "logs"
    tuples_mnist = [(item.split('.csv')[0], os.path.join(logdir, item)) for item in os.listdir(logdir) if "mnist" in item]
    for tup in tuples_mnist:
        print(tup[0])
        print(find_the_highest_val_acc(tup[1]))
        print("================================")

    tuples_cifar = [(item.split('.csv')[0], os.path.join(logdir, item)) for item in os.listdir(logdir) if "cifar" in item]
    for tup in tuples_cifar:
        print(tup[0])
        print(find_the_highest_val_acc(tup[1]))
        print("================================")

if __name__ == '__main__':
    # get_mnist_data(verbose=True)
    # get_cifar_data(verbose=True)
    present_val_acc_for_all()