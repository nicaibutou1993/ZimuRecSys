# -*- coding: utf-8 -*-
from service.redis_service import *
from util.config import *
"""
获取所有类型热点电影
"""

class HotService(object):

    def __init__(self):
        print()

    def get_hot_match(self,user_id):

        hot_label_movies = get_hot_movies_by_labels(ALL_LABELS)

        hot_movies = set()

        for label,hot_movie in hot_label_movies.items():

            hot_movies = hot_movies.union(hot_movie[:20])

        if len(hot_movies) > 0:

            history_exposure_movies = get_user_history_rec_movies(user_id)

            hot_movies = hot_movies - set(history_exposure_movies)

        return hot_movies
