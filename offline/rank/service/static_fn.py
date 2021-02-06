from offline.data_preprocess import DataPreprocess
import pickle
from util.config import PROJECT_PATH
import pandas as pd
import numpy as np
from service.redis_service import get_user_match_feature
from service.redis_service import get_movie_info
import tensorflow as tf


def get_static_data():
    """
    获取静态数据
    :return:
    """
    print("==========init static data ==================")
    data_preprocess = DataPreprocess()
    id2user, user2id = pickle.load(open(data_preprocess.user_encoder_path, 'rb'))
    id2movie, movie2id = pickle.load(open(data_preprocess.movie_encoder_path, 'rb'))

    data_path = PROJECT_PATH + "offline/data/movie_info.csv"
    movie_info_df = pd.read_csv(data_path, index_col=0)

    movie_info = np.array(movie_info_df)

    movie_ids = np.reshape(movie_info[:, 0], (-1, 1)).tolist()
    current_labels = np.reshape(movie_info[:, 1], (-1, 1)).tolist()
    release_years = np.reshape(movie_info[:, 2], (-1, 1)).tolist()
    movie_shape = len(movie_info)

    return id2user, user2id, id2movie, movie2id, movie_info, movie_ids, current_labels, release_years, movie_shape


static_id2user, static_user2id, \
static_id2movie, static_movie2id, \
static_movie_info, static_movie_ids, \
static_current_labels, static_release_years, static_movie_shape = get_static_data()


def filter_rec_movies_info(rec_movies, user_feature, is_print=True):
    """
    推荐的电影去除已经看过的电影
    :param rec_movies:
    :param user_feature:
    :param is_print:
    :return:
    """
    res = None
    if len(rec_movies) > 0:
        click_movie_ids = eval(user_feature.get("click_movie_ids"))

        click_current_labels = eval(user_feature.get("click_current_labels"))

        user_like_genres = eval(user_feature.get("user_like_genres"))

        for click_movie_id in click_movie_ids:
            if rec_movies.__contains__(click_movie_id):
                rec_movies.pop(click_movie_id)
        if is_print:
            print(click_movie_ids)
            print(click_current_labels)
            print(user_like_genres)
            print("filter after movie nums", len(rec_movies))
        if len(rec_movies) > 0:
            rec_movies = {int(static_id2movie.get(encoder_id)): score for encoder_id, score in rec_movies.items()}

            movie_ids = list(rec_movies.keys())

            movies_info = get_movie_info(movie_ids)

            rec_movies = sorted(rec_movies.items(), key=lambda x: x[1], reverse=True)

            res = [[movie_id, score, int(movies_info.get(movie_id)[2]) - 1] for movie_id, score in
                   rec_movies]

            if is_print:
                print(" ---------- rec movies ------------------")
                for i in res:
                    print(i)
    return res


def get_static_input_data(user_id):
    """
    获取模型输入数据
    :param user_id:
    :return:
    """
    encoder_user_id = static_user2id.get(user_id)

    print("input user_id is :", user_id, "current encoder user id is :", encoder_user_id)

    user_feature = get_user_match_feature(user_id)

    click_movie_ids = eval(user_feature.get("click_movie_ids"))[-20:]
    click_movie_labels = eval(user_feature.get("click_current_labels"))[-20:]
    user_like_genres = eval(user_feature.get("user_like_genres"))

    user_ids = np.repeat([[encoder_user_id]], static_movie_shape, axis=0).tolist()
    click_movie_ids = np.repeat([click_movie_ids], static_movie_shape, axis=0).tolist()
    click_movie_labels = np.repeat([click_movie_labels], static_movie_shape, axis=0).tolist()
    user_like_genres = np.repeat([user_like_genres], static_movie_shape, axis=0).tolist()

    input_x = {"user_id": user_ids, "user_recent_click_movie_ids": click_movie_ids,
               "user_recent_click_labels": click_movie_labels, "user_like_genres": user_like_genres,
               "movie_id": static_movie_ids, "current_label": static_current_labels,
               "release_year": static_release_years}

    return input_x, user_feature


def get_static_tensor_input_data(user_id):
    encoder_user_id = static_user2id.get(user_id)

    print("input user_id is :", user_id, "current encoder user id is :", encoder_user_id)

    user_feature = get_user_match_feature(user_id)

    click_movie_ids = eval(user_feature.get("click_movie_ids"))[-20:]
    click_movie_labels = eval(user_feature.get("click_current_labels"))[-20:]
    user_like_genres = eval(user_feature.get("user_like_genres"))

    user_ids = np.repeat([[encoder_user_id]], static_movie_shape, axis=0)
    click_movie_ids = np.repeat([click_movie_ids], static_movie_shape, axis=0)
    click_movie_labels = np.repeat([click_movie_labels], static_movie_shape, axis=0)
    user_like_genres = np.repeat([user_like_genres], static_movie_shape, axis=0)


    click_movie_ids = tf.constant(click_movie_ids, dtype=tf.int64,name="user_recent_click_movie_ids")
    click_movie_labels = tf.constant(click_movie_labels, dtype=tf.int64,name="user_recent_click_labels")
    user_like_genres = tf.constant(user_like_genres, dtype=tf.int64,name="user_like_genres")

    user_ids = tf.constant(user_ids, dtype=tf.int64,name="user_id")

    movie_id = tf.constant(static_movie_ids, dtype=tf.int64, name="movie_id")
    current_label = tf.constant(static_current_labels, dtype=tf.int64, name="current_label")
    release_year = tf.constant(static_release_years, dtype=tf.int64, name="release_year")

    data = [user_ids, click_movie_ids, click_movie_labels, user_like_genres, movie_id,
            current_label, release_year]

    return data, user_feature
