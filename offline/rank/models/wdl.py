from offline.rank.models.fm import FM
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

"""
wide & deep 模型

wide : 线性端
deep : mlp
"""


class WDL(object):

    def __init__(self, user_num=4691, movie_num=2514, year_num=76, genre_num=9, embedding_size=16):
        self.user_num = user_num
        self.movie_num = movie_num
        self.year_num = year_num
        self.genre_num = genre_num
        self.embedding_size = embedding_size

        self.fm = FM(user_num=self.user_num,
                     movie_num=self.movie_num,
                     year_num=self.year_num,
                     genre_num=self.genre_num,
                     embedding_size=self.embedding_size)

    """deep 端"""

    def get_deep_logit(self, sparse_embedding):
        x = Concatenate(axis=2)(sparse_embedding)

        x = Flatten()(x)
        # x = Lambda(lambda x: tf.squeeze(x,axis=1))(x)

        x = Dense(128, activation="relu")(x)

        x = Dense(128, activation="relu")(x)

        logit = Dense(1, use_bias=False)(x)

        return logit

    def get_wdl_model(self):
        """wide 端"""
        linear_logit = self.fm.get_linear_logit()

        """deep 端"""
        sparse_embedding = self.fm.get_sparse_embedding()
        deep_logit = self.get_deep_logit(sparse_embedding)

        x = Add()([linear_logit, deep_logit])

        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=self.fm.inputs, outputs=outputs)

        return model
