import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from offline.rank.models.layer import AttentionSequencePoolingLayer
import tensorflow.keras.backend as K


class Din(object):

    def __init__(self, user_num=4691, movie_num=2514, year_num=76, genre_num=9, embedding_size=16):
        self.user_num = user_num
        self.movie_num = movie_num
        self.year_num = year_num
        self.genre_num = genre_num
        self.embedding_size = embedding_size

        self.inputs = self.get_inputs()
        self.get_embedding()

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

    def get_embedding(self):
        self.user_embedding = Embedding(self.user_num, self.embedding_size)
        self.movie_embedding = Embedding(self.movie_num, self.embedding_size)
        self.genre_embedding = Embedding(self.genre_num, self.embedding_size)
        self.year_embedding = Embedding(self.year_num, self.embedding_size)

    def get_attention_vector(self):
        movie = self.movie_embedding(self.movie_id_in)
        genre = self.genre_embedding(self.current_label_in)

        hist_movies = self.movie_embedding(self.user_recent_click_movie_ids_in)
        hist_genres = self.genre_embedding(self.user_recent_click_labels_in)

        q = Concatenate()([movie, genre])
        k = Concatenate()([hist_movies, hist_genres])

        x = AttentionSequencePoolingLayer()([q, k])

        x = Flatten()(x)

        return x

    def get_dense_vector(self):
        user = self.user_embedding(self.user_id_in)
        user_genre = self.genre_embedding(self.user_like_genres_in)
        user_genre = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(user_genre)
        movie = self.movie_embedding(self.movie_id_in)
        genre = self.genre_embedding(self.current_label_in)
        year = self.year_embedding(self.release_year_in)

        x = Concatenate()([user, user_genre, movie, genre, year])

        x = Flatten()(x)

        return x

    def mlp(self):
        attention_vector = self.get_attention_vector()
        dense_vector = self.get_dense_vector()

        x = Concatenate()([attention_vector, dense_vector])

        x = Dense(128, activation="relu")(x)

        x = Dense(64, activation="relu")(x)

        outputs = Dense(1, activation="sigmoid")(x)

        return outputs

    def get_din_model(self):
        outputs = self.mlp()

        model = Model(inputs=self.inputs, outputs=outputs)

        return model
