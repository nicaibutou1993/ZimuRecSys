from offline.rank.models.fm import FM
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from offline.rank.models.layer import CrossLayer

"""
deep & cross 用于特征交叉 3阶 4阶以上  具体根据传入的层数
而 deepfm  fm 端显示的 特征交叉 2阶
"""


class DCN(object):

    def __init__(self, user_num=4691, movie_num=2514, year_num=76, genre_num=9, embedding_size=16,
                 dnn_hidden_units=(128, 128,), cross_layer_num=2):
        self.user_num = user_num
        self.movie_num = movie_num
        self.year_num = year_num
        self.genre_num = genre_num
        self.embedding_size = embedding_size

        self.dnn_hidden_units = dnn_hidden_units
        self.cross_layer_num = cross_layer_num

        self.fm = FM(user_num=self.user_num,
                     movie_num=self.movie_num,
                     year_num=self.year_num,
                     genre_num=self.genre_num,
                     embedding_size=self.embedding_size)

    """deep 端 并没有进行 Dense(1) 处理"""

    def get_deep_output(self, sparse_embedding):
        x = Concatenate(axis=2)(sparse_embedding)

        x = Flatten()(x)

        for unit in list(self.dnn_hidden_units):
            x = Dense(unit, activation="relu")(x)

        return x

    def get_cross_output(self, sparse_embedding):
        x = Concatenate(axis=2)(sparse_embedding)

        x = Flatten()(x)

        cross_output = CrossLayer(cross_layer_num=self.cross_layer_num)(x)

        return cross_output

    def get_dcn_model(self):
        linear_logit = self.fm.get_linear_logit()

        sparse_embedding = self.fm.get_sparse_embedding()
        deep_output = self.get_deep_output(sparse_embedding)

        cross_output = self.get_cross_output(sparse_embedding)

        x = Concatenate(axis=-1)([deep_output, cross_output])

        cross_logit = Dense(1)(x)

        logit = Add()([linear_logit, cross_logit])

        output = Dense(1, activation="sigmoid")(logit)

        model = Model(inputs=self.fm.inputs, outputs=output)

        return model
