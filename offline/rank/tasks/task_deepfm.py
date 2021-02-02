from offline.data_preprocess import DataPreprocess
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import numpy as np
import os
from offline.rank.models.metrics import get_callback
from offline.rank.models.deepfm import DeepFM


"""
embedding_size = 16
测试集：acc:  [0.8998336] precision:  0.7701797913197159 recall:  0.711469027660042 f1:  0.739661193981134

技巧：
    pooling: 针对序列字段：使用avg 代替 sum， sum可能导致不收敛，或者准确率相差4%以上，使用avg 效果很好
    二分类类型： 需要sigmoid,如果不sigmoid,可能准确率相差2%以上。除非向召回双塔模型，需要通过向量搜索的，
                还有FM本质 本来就不需要sigmoid的，当然可以尝试sigmoid

"""

class TaskDeepFM(object):
    user_num = 4691
    movie_num = 2514
    year_num = 76
    genre_num = 9

    embedding_size = 32

    batch_size = 512

    epoch = 20

    model_path = "checkpoint/deepfm_model/"

    def __init__(self):
        self.data_process = DataPreprocess()

    def train(self):
        self.train_dataset, \
        self.test_dataset = self.data_process.generate_data(batch_size=self.batch_size,
                                                            epoch=self.epoch)

        deepfm = DeepFM(user_num=self.user_num,
                        movie_num=self.movie_num,
                        year_num=self.year_num,
                        genre_num=self.genre_num,
                        embedding_size=self.embedding_size)

        deepfm_model = deepfm.get_deepfm_model()
        deepfm_model.summary()
        deepfm_model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

        deepfm_model.fit(self.train_dataset, epochs=self.epoch, steps_per_epoch=1270010 // self.batch_size + 1,
                         callbacks=[get_callback(self.test_dataset, self.model_path)]
                         )


if __name__ == '__main__':
    import time

    time_time = time.time()
    TaskDeepFM().train()
    print(time.time() - time_time)
