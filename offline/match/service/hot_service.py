# -*- coding: utf-8 -*-
from service.redis_service import *
from util.config import *
import numpy as np

"""
获取所有类型热点电影
"""


class HotService(object):

    def __init__(self, rec_num=50):
        self.rec_num = rec_num

    def get_hot_rec_movies(self, user_id):
        hot_label_movies = get_hot_movies_by_labels(ALL_LABELS)

        history_exposure_movies = get_user_history_rec_movies(user_id)

        hot_movies = []
        for label, hot_movie in hot_label_movies.items():
            hot_movie = [i for i in hot_movie if i not in history_exposure_movies]

            hot_movies.extend(hot_movie[:10])

        return hot_movies


if __name__ == '__main__':
    hot_service = HotService()

    movies = hot_service.get_hot_rec_movies(1)

    print(movies)
