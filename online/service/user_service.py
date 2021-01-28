# -*- coding: utf-8 -*-
from collections import defaultdict
from online.service.weight_service import update_user_weight_data
from service.redis_service import *
client = RedisClient.get_redis_client()

"""
更新用户简易画像，并存储用户浏览记录 和 点击记录
"""

def update_user_rec_info(user_id, movie_id, data):
    user_watch_movie = movie_id
    genre_movie_nums = defaultdict(int)

    user_trace = []
    for d in data:
        _movie_id = d[0]
        label = d[1][0]
        genre_movie_nums[label] = genre_movie_nums.get(label, 0) + 1
        user_trace.append(_movie_id)

        if _movie_id == movie_id:
            movie_id = label

    update_user_weight_data(user_id, movie_id, genre_movie_nums)

    set_user_history_click_movies(user_id, user_watch_movie)


# def update_user_click_movie(user_id,movie_id):
#
#     movie_info = client.hmget(REDIS_MOVIE_INFO, movie_id)
#     label = GENRE2LABELMAP.get(json.loads(movie_info[0])[1])
#     update_user_like_genre_weight(user_id,label)


# def update_user_like_genre_weight(user_id,label):
#     weights = client.hmget(REDIS_USER_LIKE_GENRE_WEIGHT, user_id)
#
#     weight = weights[0]
#     if None == weight:
#         weight = json.dumps(INIT_USER_LIKE_GENRE_WEIGHT)
#
#     weight = json.loads(weight)
#
#     weight[label] = weight.get(label) + USER_ADD_GENRE_WEIGHT
#
#     client.hmset(REDIS_USER_LIKE_GENRE_WEIGHT,{user_id:json.dumps(weight)})
#
#     hmget = client.hmget(REDIS_USER_LIKE_GENRE_WEIGHT, user_id)
#
#     print(hmget)
#

"""
获取用户喜欢电影风格权重
"""

# def get_user_like_genre_weight(user_id):
#     weights = client.hmget(REDIS_USER_LIKE_GENRE_WEIGHT, user_id)
#     weight = weights[0]
#     if None == weight:
#         weight = json.dumps(INIT_USER_LIKE_GENRE_WEIGHT)
#
#     weight = json.loads(weight)
#     return weight


"""
获取用户已经推送过的电影列表
"""


# def get_user_history_rec_movies(user_id):
#     watched_movies = client.hmget(REDIS_USER_HISTORY_REC_MOVIES, user_id)
#
#     if None == watched_movies[0]:
#         watched_movies = json.dumps([])
#
#     return watched_movies

#
# def get_user_history_rec_movies(user_id):
#     watched_movies = __get_user_history_rec_movies_from_redis(user_id)
#
#     movies = set()
#
#     for t, m in watched_movies:
#         movies = movies.union(m.split(","))
#
#     return movies
#
#
# def __get_user_history_rec_movies_from_redis(user_id):
#     watched_movies = client.hmget(REDIS_USER_HISTORY_REC_MOVIES, user_id)
#     watched_movie = watched_movies[0]
#     if None == watched_movie:
#         watched_movie = json.dumps([])
#
#     watched_movie = json.loads(watched_movie)
#     return watched_movie
#
#
# def set_user_history_rec_movies(user_id, movie_ids):
#     movies = __get_user_history_rec_movies_from_redis(user_id)
#
#     movies_str = ",".join(str(i) for i in movie_ids)
#
#     current_time = time.time()
#
#     movies.append([current_time, movies_str])
#
#     client.hmset(REDIS_USER_HISTORY_REC_MOVIES, {user_id: json.dumps(movies)})
#
#
# def set_user_history_click_movies(user_id, click_movie_id):
#
#     if click_movie_id != '0':
#         current_time = time.time()
#
#         client.hmset(REDIS_USER_HISTORY_CLICK_MOVIES, {user_id: json.dumps([current_time, click_movie_id])})
