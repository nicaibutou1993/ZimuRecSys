# -*- coding: utf-8 -*-
from service.redis_service import get_finish_weight,get_hot_movies_by_labels,get_user_history_rec_movies
"""
实时召回
"""

def get_real_time_match(user_id):

    real_time_rec_movies = set()
    finish_weight = get_finish_weight(user_id)

    if finish_weight and len(finish_weight) > 0:

        sorted_weight = sorted(finish_weight.items(), lambda x: x[1], reverse=True)

        like_genre_1 = sorted_weight[0][0]
        like_genre_2 = sorted_weight[1][0]

        hot_movies_dict = get_hot_movies_by_labels([like_genre_1, like_genre_2])


        for _,hot_movie in hot_movies_dict.items():

            real_time_rec_movies = real_time_rec_movies.union(hot_movie[:100])

        history_exposure_movies = get_user_history_rec_movies(user_id)

        real_time_rec_movies = real_time_rec_movies - set(history_exposure_movies)

    return real_time_rec_movies

