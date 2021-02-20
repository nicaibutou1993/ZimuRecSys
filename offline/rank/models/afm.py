from offline.rank.models.fm import FM
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from offline.rank.models.layer import AFMLayer
from itertools import combinations

"""
afm :attention fm

FM： 将所有的特征进行交叉后，两两结合后，并不是所有的两两结合 都是有用的，
FM默认所有的两两结合的权重系数是相等的，都是1.

AFM 就是让这些两两结合 的权重不是1.是 所有的两两结合后 总权重为1.


"""


class AFM(object):

    def __init__(self, user_num=4691, movie_num=2514,
                 year_num=76, genre_num=9,
                 embedding_size=16,
                 attention_fator=16
                 ):
        """
        :param user_num:
        :param movie_num:
        :param year_num:
        :param genre_num:
        :param embedding_size:
        :param attention_fator: attention dense num
        """
        self.user_num = user_num
        self.movie_num = movie_num
        self.year_num = year_num
        self.genre_num = genre_num
        self.embedding_size = embedding_size

        self.attention_fator = attention_fator

        self.fm = FM(user_num=self.user_num,
                     movie_num=self.movie_num,
                     year_num=self.year_num,
                     genre_num=self.genre_num,
                     embedding_size=self.embedding_size)

    def get_afm_logit(self, sparse_embedding):
        afm_out = AFMLayer(self.attention_fator)(sparse_embedding)

        return afm_out

    def get_afm_model(self):
        linear_logit = self.fm.get_linear_logit()

        sparse_embedding = self.fm.get_sparse_embedding()

        afm_logit = self.get_afm_logit(sparse_embedding)

        x = Add()([linear_logit, afm_logit])

        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=self.fm.inputs, outputs=outputs)

        return model
