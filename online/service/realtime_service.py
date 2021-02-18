# -*- coding: utf-8 -*-
from online.service.weight_service import WeightService
from service.redis_service import *

from online.service.data_util import diff_and_sort

"""
这里 针对冷用户与老用户不一致

针对冷用户：这里默认推荐书目较多，可能达到 10-15条，来探索用户兴趣

针对老用户，这里指的是实时推荐，可能只能达到3-5之间，具体根据业务设定

"""


class RealTimeService(object):

    def __init__(self, rec_num=15):

        self.weight_service = WeightService(rec_num)

    def get_user_realtime_rec_movies(self, user_id):
        """用户冷启动推荐
              根据用户画像 进行实时推荐
           """

        finish_weights = get_finish_weight(user_id)

        if finish_weights is None or len(finish_weights) == 0:
            _, genre_movie_num = self.weight_service.get_user_weight_data(user_id)
        else:
            genre_movie_num = self.weight_service.rec_genre_movie_num(finish_weights)

        """用户历史浏览记录"""
        history_movies = get_user_history_rec_movies(user_id)

        rec_movies = []

        """排除已经浏览的电影"""
        for label, num in genre_movie_num.items():
            movies = client.hmget(REDIS_HOT_MOVIES, label)
            hot_movies = movies[0].split(",")
            #hot_movies = [i for i in hot_movies if i not in history_movies]
            hot_movies = diff_and_sort(hot_movies, history_movies)

            rec_movies += hot_movies[:num]

        """这里不能设置，到最后所有通道汇集之后，在统一设置"""
        ##set_user_history_rec_movies(user_id, rec_movies)

        return rec_movies


if __name__ == '__main__':
    service = RealTimeService()

    movies = service.get_user_realtime_rec_movies(1)

    print(movies)
