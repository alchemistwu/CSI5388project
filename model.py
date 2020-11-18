"""
Author:Junzheng Wu
Email: jwu220@uottawa.ca
Student ID: 300084962
Date:2020/11/18
Note: Use sparse_category_crossentropy rather than one-hot encoder for saving memory.
"""
import tensorflow as tf

def get_victim_model(pretrain=True, task='mnist', verbose=0):
    """
    Due to the API limitation, shape of MNIST has to be resized as (32, 32, 3)
    :param pretrain: if Ture, return vgg model with ImageNet weights
    :param task: 'mnist' or 'cifar'
    :param verbose: Print model information if True
    :return: keras model
    """
    inputshape_dict={'mnist':(32, 32), 'cifar': (32, 32, 3)}
    input = tf.keras.layers.Input(shape=inputshape_dict[task])

    if task == 'mnist':
        x = tf.keras.layers.Reshape((inputshape_dict[task][0], inputshape_dict[task][1], 3))(input)
    else:
        x = tf.keras.layers.Reshape(inputshape_dict[task])(input)
    
    if pretrain:
        vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, classes=10)(x)
    else:
        vgg = tf.keras.applications.VGG16(weights=None, include_top=False, classes=10)(x)
        
    model = tf.keras.Model(inputs=[input], outputs=[vgg])
    model.compile(loss='sparse_categorical_accuracy', metrics=['acc'])
    if verbose:
        model.summary()
    return model


def get_attack_model(pretrain=True, task='mnist', verbose=0):
    """
    Due to the API limitation, shape of MNIST has to be resized as (32, 32, 3)
    :param pretrain: if Ture, return attack model with ImageNet weights
    :param task: 'mnist' or 'cifar'
    :param verbose: Print model information if True
    :return: keras model
    """
    inputshape_dict = {'mnist': (32, 32), 'cifar': (32, 32, 3)}
    input = tf.keras.layers.Input(shape=inputshape_dict[task])

    if task == 'mnist':
        x = tf.keras.layers.Reshape((inputshape_dict[task][0], inputshape_dict[task][1], 3))(input)
    else:
        x = tf.keras.layers.Reshape(inputshape_dict[task])(input)

    if pretrain:
        mobile = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, classes=10)(x)
    else:
        mobile = tf.keras.applications.MobileNetV2(weights=None, include_top=False, classes=10)(x)

    model = tf.keras.Model(inputs=[input], outputs=[mobile])
    model.compile(loss='sparse_categorical_accuracy', metrics=['acc'])
    if verbose:
        model.summary()
    return model

if __name__ == '__main__':
    model = get_attack_model(verbose=True, pretrain=True, task='cifar')