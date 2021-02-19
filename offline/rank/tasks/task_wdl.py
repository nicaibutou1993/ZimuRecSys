from offline.data_preprocess import DataPreprocess
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import numpy as np
import os
from offline.rank.models.metrics import get_callback
from offline.rank.models.wdl import WDL
from offline.rank.models.metrics import get_metrics
from offline.rank.inputs import get_input_tensor_data
from util.config import *

"""
embedding_size = 16
测试集：acc:  [0.89820728] precision:  0.7571860816944024 recall:  0.7228344478980252 f1:  0.7396116099394445


embedding_size = 32
测试集：acc:  [0.89796239] precision:  0.7631658850915961 recall:  0.7102131801199334 f1:  0.7357379821765433


针对 wdl 模型  embedding_size 32 的效果 比 16 效果差一点。


"""
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class TaskWDL(object):
    user_num = 4691
    movie_num = 2514
    year_num = 76
    genre_num = 9

    embedding_size = 32

    batch_size = 512

    epoch = 20

    version = "0002"

    model_path = PROJECT_PATH + 'offline/rank/tasks/checkpoint/wdl_model/'

    def __init__(self, version="0001"):
        self.data_process = DataPreprocess()
        self.version = version
        model_path = self.model_path + self.version
        # if os.path.exists(model_path):
        #     self.wdl_model = tf.saved_model.load(model_path)

    def train(self):
        self.train_dataset, \
        self.test_dataset = self.data_process.generate_data(batch_size=self.batch_size,
                                                            epoch=self.epoch)

        wdl = WDL(user_num=self.user_num,
                  movie_num=self.movie_num,
                  year_num=self.year_num,
                  genre_num=self.genre_num,
                  embedding_size=self.embedding_size)

        wdl_model = wdl.get_wdl_model()
        wdl_model.summary()
        wdl_model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

        wdl_model.fit(self.train_dataset, epochs=self.epoch, steps_per_epoch=1270010 // self.batch_size + 1,
                      callbacks=[get_callback(self.test_dataset, self.model_path, self.version)]
                      )

    def predict(self, test_dataset=None, is_print=True):

        if test_dataset is None:
            test_dataset = self.data_process.generate_test_data()

        true_y = []
        pred_y = []
        for test_x, test_y in test_dataset:
            data = get_input_tensor_data(test_x)
            array = self.wdl_model(data, training=False)
            _array = np.where(array > 0.5, 1, 0)
            pred_y.extend(_array)
            true_y.extend(test_y.numpy()[:, np.newaxis])

        get_metrics(true_y, pred_y, is_print=is_print)


if __name__ == '__main__':
    import time

    time_time = time.time()
    TaskWDL().train()
    print(time.time() - time_time)
