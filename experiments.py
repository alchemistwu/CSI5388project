from model_utils import *
from data_utils import *
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import os
import shutil

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, filepath, model_name):
        super(SaveBestModel, self).__init__()
        self.model_name = model_name
        self.best_weights = None
        self.file_path = filepath
        self.best = 0.

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_acc')
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

def train_model(model, train_data, test_data, logs_path, type, batch_size=128, attributes=[]):
    if not os.path.exists('logs'):
        os.mkdir('logs')
    train_data = train_data.cache()
    train_data = train_data.shuffle(60000)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    test_data = test_data.batch(batch_size)
    test_data = test_data.cache()
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)
    model_name = "_".join([str(item) for item in attributes])
    model.fit(train_data, epochs=100, validation_data=test_data, verbose=1, batch_size=batch_size,
              callbacks=[SaveBestModel(type, model_name), CSVLogger('logs/%s.csv'%type + '_' + model_name),
                         EarlyStopping(monitor='val_loss', patience=10)])

def train_attack_model(task='mnist', pretrain=False, target_size=48, split=0.7, multi_gpu=True, batch_size=256):
    data_victim, data_attack, data_test = get_mnist_data(target_size=target_size, split=split)
    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.Strategy()
    attributes = [task]
    if pretrain:
        attributes.append('pretrain')

    with strategy.scope():
        model = get_victim_model(pretrain=False, task=task, target_size=target_size)
        train_model(model, data_victim, data_test, 'logs', 'victim', batch_size=batch_size, attributes=attributes)



if __name__ == '__main__':
    train_attack_model(pretrain=True)
    train_attack_model(pretrain=False, task='cifar')
    train_attack_model(pretrain=True, task='cifar')

