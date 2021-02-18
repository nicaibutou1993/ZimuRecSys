# -*- coding: utf-8 -*-
from service.redis_service import get_finish_weight, get_hot_movies_by_labels, get_user_history_rec_movies
import numpy as np

"""
实时召回
"""

class RealTimeService(object):

    def __init__(self):
        print()

    def get_real_time_match(self, user_id):

        real_time_rec_movies = []
        finish_weight = get_finish_weight(user_id)

        if finish_weight and len(finish_weight) > 0:

            sorted_weight = sorted(finish_weight.items(), key=lambda x: x[1], reverse=True)

            like_genre_1 = sorted_weight[0][0]
            like_genre_2 = sorted_weight[1][0]

            hot_movies_dict = get_hot_movies_by_labels([like_genre_1, like_genre_2])

            history_exposure_movies = get_user_history_rec_movies(user_id)

            for _, hot_movie in hot_movies_dict.items():

                hot_movie = [i for i in hot_movie if i not in history_exposure_movies]

                real_time_rec_movies.extend(hot_movie[:50])

        return real_time_rec_movies


if __name__ == '__main__':
    match = RealTimeService().get_real_time_match(1)

    print(match)
