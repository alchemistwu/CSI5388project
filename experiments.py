from model_utils import *
from data_utils import *
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import os
import shutil

class SaveBestModel(tf.keras.callbacks.Callback):
    """
    Callbacks for saving the model with lowest val_acc
    """
    def __init__(self, filepath, model_name, monitor='val_acc'):
        super(SaveBestModel, self).__init__()
        self.model_name = model_name
        self.best_weights = None
        self.file_path = filepath
        self.best = 0.
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if not current:
            current = 0.0

        if np.less(self.best, current):
            self.best = current
            self.best_weights = self.model.get_weights()

            if not os.path.exists(self.file_path):
                os.mkdir(self.file_path)

            new_path = os.path.join(self.file_path, self.model_name)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            else:
                shutil.rmtree(new_path)

            new_path_model = os.path.join(new_path, 'model.tf')
            self.model.save_weights(new_path_model)
            print("New best model has been saved in %s!" % new_path_model)
            print("Best acc: %.4f" % current)

def train_model(model, train_data, test_data, type, batch_size=128, attributes=[], shuffle_size=60000):
    """
    Basic method for training model
    :param model: compiled model from get_****_model() methods
    :param train_data: tensorflow dataset format
    :param test_data:  tensorflow dataset format
    :param type: 'victim' or 'attack'
    :param batch_size:
    :param attributes: a list of attributes in str format
    :return:
    """
    if not os.path.exists('logs'):
        os.mkdir('logs')
    train_data = train_data.cache()
    train_data = train_data.shuffle(shuffle_size)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    test_data = test_data.batch(batch_size)
    test_data = test_data.cache()
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)
    model_name = "_".join([str(item) for item in attributes])
    if 'mixed' in attributes:
        model.fit(train_data, epochs=100, validation_data=test_data, verbose=1, batch_size=batch_size,
                  callbacks=[SaveBestModel(type, model_name, monitor='val_dense_1_acc'),
                             CSVLogger('logs/%s.csv'% (type + '_' + model_name)),
                             EarlyStopping(monitor='val_dense_1_loss', patience=10)
                             ])
    else:
        model.fit(train_data, epochs=100, validation_data=test_data, verbose=1, batch_size=batch_size,
                  callbacks=[SaveBestModel(type, model_name), CSVLogger('logs/%s.csv'% (type + '_' + model_name)),
                             EarlyStopping(monitor='val_loss', patience=10)])

def train_victim_model(task='mnist', pretrain=False, target_size=48, split=0.7, multi_gpu=True, batch_size=256):
    if task == 'mnist':
        data_victim, data_attack, data_test = get_mnist_data(target_size=target_size, split=split)
    else:
        data_victim, data_attack, data_test = get_cifar_data(target_size=target_size, split=split)
    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.Strategy()
    attributes = [task]
    if pretrain:
        attributes.append('pretrain')

    with strategy.scope():
        model = get_victim_model(pretrain=pretrain, task=task, target_size=target_size)
        train_model(model, data_victim, data_test, 'victim', batch_size=batch_size, attributes=attributes)

def get_attack_data(task='mnist', use_pretrain_victim=False, target_size=48, split=0.7, batch_size=256, mixed=False,
                    verbose=False):
    if task == 'mnist':
        data_victim, data_attack, data_test = get_mnist_data(target_size=target_size, split=split)
    else:
        data_victim, data_attack, data_test = get_cifar_data(target_size=target_size, split=split)

    victim_model = get_victim_model(pretrain=use_pretrain_victim, task=task, target_size=target_size)
    attributes = [task]
    if use_pretrain_victim:
        attributes.append('pretrain')

    model_name = "_".join([str(item) for item in attributes])
    model_folder = os.path.join('victim', model_name)
    best_model_weights = os.path.join(model_folder,
                                      [item for item in os.listdir(model_folder) if ".index" in item][0].replace(
                                          ".index", ""))
    victim_model.load_weights(best_model_weights)
    images = np.array([list(x[0].numpy()) for x in list(data_attack)])
    labels = np.array([x[1].numpy() for x in list(data_attack)])
    
    data_attack = data_attack.batch(batch_size)
    data_attack = data_attack.cache()
    data_attack = data_attack.prefetch(tf.data.experimental.AUTOTUNE)
    
    predicted_labels = victim_model.predict(data_attack, batch_size=batch_size)
    predicted_labels = np.argmax(predicted_labels, axis=1)

    images = tf.convert_to_tensor(images)
    predicted_labels = tf.convert_to_tensor(predicted_labels)
    labels = tf.convert_to_tensor(labels)

    images_test = np.array([list(x[0].numpy()) for x in list(data_test)])
    labels_test = np.array([x[1].numpy() for x in list(data_test)])
    images_test = tf.convert_to_tensor(images_test)
    labels_test = tf.convert_to_tensor(labels_test)

    if mixed:
        data_label = tf.data.Dataset.from_tensor_slices((predicted_labels, labels))
        data_images = tf.data.Dataset.from_tensor_slices(images)
        data_victim = tf.data.Dataset.zip((data_images, data_label))

        data_label_test = tf.data.Dataset.from_tensor_slices((labels_test, labels_test))
        data_images_test = tf.data.Dataset.from_tensor_slices(images_test)
        data_test = tf.data.Dataset.zip((data_images_test, data_label_test))

    else:
        data_victim = tf.data.Dataset.from_tensor_slices((images, predicted_labels))



    if verbose:
        plt.figure(figsize=(10, 10))
        if mixed:
            for i, (image, label) in enumerate(data_victim.take(9)):
                print(image.shape)
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image)
                plt.title(str(int(label[0])) + '_' + str(int(label[1])))
                plt.axis("off")
            plt.show()
        else:
            for i, (image, label) in enumerate(data_victim.take(9)):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image)
                plt.title(int(label))
                plt.axis("off")
            plt.show()

        if mixed:
            for i, (image, label) in enumerate(data_test.take(9)):
                print(image.shape)
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image)
                plt.title(str(int(label[0])) + '_' + str(int(label[1])))
                plt.axis("off")
            plt.show()
        else:
            for i, (image, label) in enumerate(data_test.take(9)):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image)
                plt.title(int(label))
                plt.axis("off")
            plt.show()

    return data_victim, data_test

def train_attack_model(task='mnist', pretrain=False, target_size=48,
                       split=0.5, multi_gpu=True, batch_size=256, mixed=False, use_pretrain_victim=False):

    data_victim, data_test = get_attack_data(use_pretrain_victim=use_pretrain_victim,
                                             task=task,
                                             target_size=target_size,
                                             split=split, mixed=mixed)
    # if task == 'mnist':
    #     data_victim, data_attack, data_test = get_mnist_data(target_size=target_size, split=split)
    # else:
    #     data_victim, data_attack, data_test = get_cifar_data(target_size=target_size, split=split)

    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.Strategy()
    attributes = [task]
    if pretrain:
        attributes.append('pretrain')
    if mixed:
        attributes.append('mixed')
    if use_pretrain_victim:
        attributes.append('usePretrain')

    with strategy.scope():
        model = get_attack_model(pretrain=pretrain, task=task, target_size=target_size, mixed=mixed)
        train_model(model, data_victim, data_test, 'attack', batch_size=batch_size, attributes=attributes)
    

if __name__ == '__main__':
    pretrain = [True, False]
    task = ['mnist', 'cifar']
    use_pretrain_victim = [True, False]
    mixed = [True, False]

    for pretrain_item in pretrain:
        for task_item in task:
            train_victim_model(pretrain=pretrain_item, task=task_item, split=0.5)

    for pretrain_item in pretrain:
        for task_item in task:
            for use_item in use_pretrain_victim:
                for mix_item in mixed:
                    train_attack_model(task=task_item, pretrain=pretrain_item, use_pretrain_victim=use_item, multi_gpu=True,
                                       mixed=mix_item)



