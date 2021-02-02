import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from offline.rank.models.layer import FMLayer

"""
FM 模型
"""


class FM(object):

    def __init__(self, user_num=4691, movie_num=2514, year_num=76, genre_num=9, embedding_size=16):
        self.user_num = user_num
        self.movie_num = movie_num
        self.year_num = year_num
        self.genre_num = genre_num
        self.embedding_size = embedding_size

        self.inputs = self.get_inputs()

    def get_inputs(self):
        """
        输入
        :return:
        """
        self.user_id_in = Input(shape=(1,), name="user_id", dtype=tf.int64)
        self.user_recent_click_movie_ids_in = Input(shape=(20,), name="user_recent_click_movie_ids", dtype=tf.int64)
        self.user_recent_click_labels_in = Input(shape=(20,), name="user_recent_click_labels", dtype=tf.int64)
        self.user_like_genres_in = Input(shape=(2,), name="user_like_genres", dtype=tf.int64)

        self.movie_id_in = Input(shape=(1,), name="movie_id", dtype=tf.int64)
        self.current_label_in = Input(shape=(1,), name="current_label", dtype=tf.int64)
        self.release_year_in = Input(shape=(1,), name="release_year", dtype=tf.int64)

        return [self.user_id_in, self.user_recent_click_movie_ids_in, self.user_recent_click_labels_in, \
                self.user_like_genres_in, self.movie_id_in, self.current_label_in, self.release_year_in]

    def get_linear_embedding(self):
        """
        线性端 类别 embedding ,输出维度是 1
        :return:
        """
        user = Embedding(self.user_num, 1)(self.user_id_in)
        movie_embedding = Embedding(self.movie_num, 1)
        click_movie_embedding = movie_embedding(self.user_recent_click_movie_ids_in)
        click_movie = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(click_movie_embedding)

        genre_embedding = Embedding(self.genre_num, 1)

        click_genre_embedding = genre_embedding(self.user_recent_click_labels_in)
        click_genre = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(click_genre_embedding)

        user_like_genre_embedding = genre_embedding(self.user_like_genres_in)
        user_like_genre = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(user_like_genre_embedding)

        movie = movie_embedding(self.movie_id_in)

        current_label = genre_embedding(self.current_label_in)

        year_embedding = Embedding(self.year_num, 1)
        release_year = year_embedding(self.release_year_in)

        return [user, click_movie, click_genre, user_like_genre, movie, current_label, release_year]

    def get_linear_logit(self, linear_embedding=None):
        """
        线性端 输出
        :return:
        """
        if linear_embedding is None:
            linear_embedding = self.get_linear_embedding()
        concat = Concatenate(axis=1)(linear_embedding)

        logits = Lambda(lambda x: tf.reduce_sum(x, axis=1))(concat)

        return logits

    def get_sparse_embedding(self):
        """
        类别类型 Embedding
        :return:
        """
        user = Embedding(self.user_num, self.embedding_size)(self.user_id_in)

        movie_embedding = Embedding(self.movie_num, self.embedding_size)

        click_movie_embedding = movie_embedding(self.user_recent_click_movie_ids_in)
        click_movie = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(click_movie_embedding)

        genre_embedding = Embedding(self.genre_num, self.embedding_size)
        click_genre_embedding = genre_embedding(self.user_recent_click_labels_in)
        click_genre = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(click_genre_embedding)

        user_like_genre_embedding = genre_embedding(self.user_like_genres_in)
        user_like_genre = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(user_like_genre_embedding)

        movie = movie_embedding(self.movie_id_in)

        current_label = genre_embedding(self.current_label_in)

        year_embedding = Embedding(self.year_num, self.embedding_size)
        release_year = year_embedding(self.release_year_in)

        return [user, click_movie, click_genre, user_like_genre, movie, current_label, release_year]

    def get_fm_logit(self, sparse_embedding=None):
        """
        fm 特征交叉端

        :param sparse_embedding:
        :return:
        """
        if sparse_embedding is None:
            sparse_embedding = self.get_sparse_embedding()

        concat = Concatenate(axis=1)(sparse_embedding)

        fm_logit = FMLayer()(concat)

        return fm_logit

    def get_fm_model(self, sparse_embedding=None, linear_embedding=None):
        """
        获取fm 模型
        :param sparse_embedding: 可以通过外部传入
        :return:
        """
        linear_logit = self.get_linear_logit(linear_embedding)

        fm_logit = self.get_fm_logit(sparse_embedding)

        outputs = Add()([linear_logit, fm_logit])

        model = Model(inputs=self.inputs, outputs=outputs)

        return model
