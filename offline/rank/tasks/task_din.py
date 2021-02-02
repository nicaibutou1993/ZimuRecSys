from offline.data_preprocess import DataPreprocess
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import numpy as np
import os
from offline.rank.models.metrics import get_callback
from offline.rank.models.din import Din

"""

din 模型

embedding_size = 16
模型预测最终指标
参数量大约：153,026
测试集：acc:  0.88480738 precision:  0.7634391823359601 recall:  0.6144234089981476 f1:  0.680873271288162

embedding_size = 32
模型预测最终指标
参数量大约：294,242
测试集：acc:  [0.907758] precision:  0.7923011412025208 recall:  0.7302125521961634 f1:  0.7599908505702055



"""


# tf.compat.v1.disable_eager_execution()


class TaskDIN(object):
    user_num = 4691
    movie_num = 2514
    year_num = 76
    genre_num = 9

    embedding_size = 16

    batch_size = 512

    epoch = 20

    model_path = "checkpoint/din_model/"

    def __init__(self):
        self.data_process = DataPreprocess()

    def train(self):
        self.train_dataset, \
        self.test_dataset = self.data_process.generate_data(batch_size=self.batch_size,
                                                            epoch=self.epoch)

        din = Din(user_num=self.user_num,
                  movie_num=self.movie_num,
                  year_num=self.year_num,
                  genre_num=self.genre_num,
                  embedding_size=self.embedding_size)

        din_model = din.get_din_model()
        din_model.summary()
        din_model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

        din_model.fit(self.train_dataset, epochs=self.epoch, steps_per_epoch=1270010 // self.batch_size + 1,
                      callbacks=[get_callback(self.test_dataset, self.model_path)]
                      )


if __name__ == '__main__':
    import time

    time_time = time.time()
    TaskDIN().train()
    print(time.time() - time_time)
