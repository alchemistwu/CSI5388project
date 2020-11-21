"""
Author:Junzheng Wu
Email: jwu220@uottawa.ca
Student ID: 300084962
Date:2020/11/18
Note: Use sparse_category_crossentropy rather than one-hot encoder for saving memory.
"""
import tensorflow as tf

def get_victim_model(pretrain=True, task='mnist', verbose=0, target_size=48):
    """
    Due to the API limitation, shape of MNIST has to be resized as (32, 32, 3)
    :param pretrain: if Ture, return vgg model with ImageNet weights
    :param task: 'mnist' or 'cifar'
    :param verbose: Print model information if True
    :return: keras model
    """
    inputshape_dict={'mnist':(target_size, target_size, 3), 'cifar': (target_size, target_size, 3)}
    input = tf.keras.layers.Input(shape=inputshape_dict[task])
    if pretrain:
        m = tf.keras.applications.VGG16(weights='imagenet',
                                                   include_top=False,
                                                   input_shape=inputshape_dict[task])(input)
    else:
        m = tf.keras.applications.VGG16(weights=None,
                                                   include_top=False,
                                                   input_shape=inputshape_dict[task])(input)

    flatten = tf.keras.layers.GlobalMaxPool2D()(m)
    output = tf.keras.layers.Dense(10, activation='softmax')(flatten)
    model = tf.keras.Model(inputs=[input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'])
    if verbose:
        model.summary()
    return model

def get_attack_model(pretrain=True, task='mnist', verbose=0, target_size=48, mixed=False, mixed_rate=0.5):
    """
    Due to the API limitation, shape of MNIST has to be resized as (32, 32, 3)
    :param pretrain: if Ture, return attack model with ImageNet weights
    :param task: 'mnist' or 'cifar'
    :param verbose: Print model information if True
    :return: keras model
    """
    inputshape_dict={'mnist':(target_size, target_size, 3), 'cifar': (target_size, target_size, 3)}
    input = tf.keras.layers.Input(shape=inputshape_dict[task])
    if pretrain:
        m = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=inputshape_dict[task])(input)
    else:
        m = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=inputshape_dict[task])(input)

    flatten = tf.keras.layers.GlobalMaxPool2D()(m)
    output = tf.keras.layers.Dense(10, activation='softmax')(flatten)

    if mixed:
        output2 = output
        model = tf.keras.Model(inputs=[input], outputs=[output, output2])
        model.compile(loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], metrics=['acc'],
                      loss_weights=[mixed_rate, 1. - mixed_rate])
    else:
        model = tf.keras.Model(inputs=[input], outputs=[output])
        model.compile(loss=['sparse_categorical_crossentropy'], metrics=['acc'])
    if verbose:
        model.summary()
    return model

if __name__ == '__main__':
    model = get_attack_model(verbose=True, pretrain=True, task='cifar', mixed=True)
    model = get_victim_model(target_size=224, verbose=True, task='mnist')