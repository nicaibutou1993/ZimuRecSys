# -*- coding: utf-8 -*-
from util.redis_client import RedisClient
from util.config import *
from collections import Counter
import json


class ColdMovieService(object):

    """
    电影冷启动策略:
    用于对电影评分，
    根据电影的风格 推荐给 喜欢该风格的用户，
    曝光1000次，最终计算点击率、停留时间、一个小时、三小时、一天等等 相关指标,来衡量电影的质量的好坏
    
    电影质量的好的评定：
    1.通过人工筛选。
    2.曝光用户，根据用户的点击率 停留时间 评判。

    """
    client = RedisClient.get_redis_client()

    cold_movie_need_display_num = 100

    """
    向用户推荐 冷电影，用于衡量 电影的质量的好坏的指标
    """

    def get_cold_movie_rec(self, user_id):

        rec_cold_match = ""
        user_profile_label = self.get_user_profile(user_id)

        if user_profile_label:

            cold_movies = self.client.hmget(REDIS_COLD_MOVIES, user_profile_label)[0]

            if cold_movies:
                cold_movies = json.loads(cold_movies)

                if len(cold_movies) > 0:
                    cold_movies_arr = sorted(cold_movies.items(), key=lambda x: x[1], reverse=True)

                    rec_cold_match = cold_movies_arr[0][0]

        return rec_cold_match

    """
    如果对外推荐给用户，数据确实推荐给用户，需要更新redis 
    """

    def update_cold_movies_list(self, movie_id):

        movie_info = self.client.hmget(REDIS_MOVIE_INFO, movie_id)[0]

        if movie_info:
            label = GENRE2LABELMAP.get(movie_info[1])
            cold_movies = self.client.hmget(REDIS_COLD_MOVIES, label)[0]

            if cold_movies:

                click_num = cold_movies.get(movie_id)

                click_num = click_num + 1

                if click_num >= self.cold_movie_need_display_num:
                    cold_movies.pop(movie_id)
                else:
                    cold_movies[movie_id] = click_num

                self.client.hmset(REDIS_COLD_MOVIES, {label: json.dumps(cold_movies)})

    def get_user_profile(self, user_id):
        """
            获取用户的最喜欢的一个风格，当然这里必须要求用户已经经过了50刷，才会给出用户最喜欢的风格
            """

        profile_10 = self.client.hmget(REDIS_SHORT_TERM_INTEREST_10, user_id)[0]

        profile_20 = self.client.hmget(REDIS_MIDDLE_TERM_INTEREST_20, user_id)[0]

        profile_50 = self.client.hmget(REDIS_LONG_TERM_INTEREST_50, user_id)[0]

        user_profile_label = ""
        if profile_10 is not None and profile_20 is not None and profile_50 is not None:
            profile_10 = Counter(json.loads(profile_10))
            profile_20 = Counter(json.loads(profile_20))
            profile_50 = Counter(json.loads(profile_50))

            profile = dict(profile_10 + profile_20 + profile_50)

            profile = sorted(profile.items(), key=lambda x: x[1], reverse=True)

            user_profile_label = profile[0][0]

        return user_profile_label


if __name__ == '__main__':
    rec = ColdMovieService().get_cold_movie_rec(1)
    print(rec)
