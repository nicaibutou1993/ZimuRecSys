from offline.rank.models.fm import FM
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.models import Model

"""
DeepFM 模型
"""
class DeepFM(object):

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

    def get_deep_logit(self, sparse_embedding):
        x = Concatenate(axis=2)(sparse_embedding)

        x = Flatten()(x)
        #x = Lambda(lambda x: tf.squeeze(x,axis=1))(x)

        x = Dense(128, activation="relu")(x)

        x = Dense(128, activation="relu")(x)

        logit = Dense(1, use_bias=False)(x)

        return logit

    def get_deepfm_model(self):
        sparse_embedding = self.fm.get_sparse_embedding()
        fm_model = self.fm.get_fm_model(sparse_embedding)

        deep_logit = self.get_deep_logit(sparse_embedding)

        x = Add()([fm_model.outputs[0], deep_logit])

        outputs = Dense(1,activation="sigmoid")(x)

        model = Model(inputs=fm_model.inputs, outputs=outputs)

        return model
