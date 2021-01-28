# -*- coding: utf-8 -*-
from online.service.weight_service import get_user_weight_data,rec_genre_movie_num

from service.redis_service import *

"""用户冷启动推荐"""
def get_user_cold_start_rec(user_id):

    finish_weights = get_finish_weight(user_id)

    if None == finish_weights or len(finish_weights) == 0:
        _, genre_movie_num = get_user_weight_data(user_id)
    else:
        genre_movie_num = rec_genre_movie_num(finish_weights)

    """用户历史浏览记录"""
    history_movies = get_user_history_rec_movies(user_id)

    rec_movies = []

    """排除已经浏览的电影"""
    for label, num in genre_movie_num.items():
        movies = client.hmget(REDIS_HOT_MOVIES, label)
        hot_movies = movies[0].split(",")
        hot_movies = [i for i in hot_movies if i not in history_movies]
        rec_movies += hot_movies[:num]

    """这里不能设置，到最后所有通道汇集之后，在统一设置"""
    ##set_user_history_rec_movies(user_id, rec_movies)

    return rec_movies



