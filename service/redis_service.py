# -*- coding: utf-8 -*-
from util.redis_client import RedisClient
from util.config import *
import time
import json

client = RedisClient.get_redis_client()


def get_current_weight(user_id):
    """获取用户当前最敏感权重 即最新权重"""
    weight = client.hmget(REDIS_CURRENT_WEIGHTS, user_id)

    weight = weight[0]
    if weight is None:
        weight = json.dumps(INIT_WEIGHT)

    weight = json.loads(weight)

    return weight


def set_current_weight(weight, user_id):
    """设置用户当前最敏感权重 即最新权重"""
    client.hmset(REDIS_CURRENT_WEIGHTS, {user_id: json.dumps(weight)})


def get_user_history_rec_movies(user_id):
    """获取已经曝光给用户的电影列表，这里只是曝光，并不是点击"""
    watched_movies = __get_user_history_rec_movies_from_redis(user_id)
    movies = set()
    for t, m in watched_movies:
        movies = movies.union(m.split(","))
    return movies


def __get_user_history_rec_movies_from_redis(user_id):
    watched_movies = client.hmget(REDIS_USER_HISTORY_REC_MOVIES, user_id)
    watched_movie = watched_movies[0]
    if watched_movie is None:
        watched_movie = json.dumps([])

    watched_movie = json.loads(watched_movie)
    return watched_movie


def set_user_history_rec_movies(user_id, movie_ids):
    """设置已经曝光给用户的电影列表，这里只是曝光，并不是点击"""
    movies = __get_user_history_rec_movies_from_redis(user_id)

    movies_str = ",".join(str(i) for i in movie_ids)

    current_time = time.time()

    movies.append([current_time, movies_str])

    client.hmset(REDIS_USER_HISTORY_REC_MOVIES, {user_id: json.dumps(movies)})


def set_user_history_click_movies(user_id, click_movie_id):
    """设置用户点击的电影"""
    if click_movie_id != '0':
        current_time = time.time()

        client.hmset(REDIS_USER_HISTORY_CLICK_MOVIES, {user_id: json.dumps([current_time, click_movie_id])})


def get_finish_weight(user_id):
    """获取用户最终的权重"""
    finish_weights = client.hmget(REDIS_FINISH_WEIGHTS, user_id)[0]

    if finish_weights is None:
        finish_weights = json.dumps({})

    finish_weights = json.loads(finish_weights)

    return finish_weights


"""获取用户历史点击的风格"""


def get_hot_movies_by_labels(labels):
    """
    根据电影风格，获取对应热门电影
    :param labels: 可以传入数组 也可以传入字符串
    :return: 热门电影
    """

    if isinstance(labels, list):
        movies = client.hmget(REDIS_HOT_MOVIES, labels)
        hot_movies = {}
        for label, movie in zip(labels, movies):
            hot_movies[label] = movie.split(",")
    elif isinstance(labels, str):
        movies = client.hmget(REDIS_HOT_MOVIES, labels)
        hot_movies = movies[0].split(",")
    else:
        hot_movies = []
    return hot_movies


def get_user_click_genres(user_id):
    click_labels = client.hmget(REDIS_USER_HISTORY_CLICK_GENRES, user_id)

    click_labels = click_labels[0]
    if click_labels is None:
        click_labels = json.dumps([])

    click_labels = json.loads(click_labels)
    return click_labels


def get_user_match_feature(user_id):
    """获取召回阶段，用户特征信息,这里的 movie_id,user_id,genre
    等等信息都是已经编码过了 如果对外提供数据需要转码"""
    user_feature = client.hmget(REDIS_MATCH_USER_RECENT_HISTORY_CLICK_MOVIE_TRACE, user_id)[0]

    if user_feature is None:
        user_feature = json.dumps({})

    user_feature = json.loads(user_feature)

    return user_feature


def get_movie_info(movie_ids):

    movies_info_dict = {}
    movies_info = client.hmget(REDIS_MOVIE_INFO, movie_ids)

    if not isinstance(movie_ids, list):
        movie_ids = [movie_ids]

    for movie_id, movie_info in zip(movie_ids, movies_info):
        try:
            movie_info = json.loads(movie_info)

            label = GENRE2LABELMAP.get(movie_info[1])

            movie_info.append(label)

            movies_info_dict[movie_id] = movie_info
        except Exception as e:
            print(e)

    return movies_info_dict


if __name__ == '__main__':
    # labels = get_hot_movies_by_labels('1')
    # print(labels)
    info = get_movie_info([727])

    print(info)
