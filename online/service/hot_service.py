from service.redis_service import *
from util.config import *
import numpy as np
from online.service.data_util import diff_and_sort
"""
获取热门电影
"""



class HotService(object):

    def __init__(self, rec_num=1):
        self.rec_num = rec_num

    def get_user_hot_movies_rec(self, user_id,rec_movies=None):
        label = np.random.choice(ALL_LABELS)
        hot_movie = get_hot_movies_by_labels(label)

        history_exposure_movies = get_user_history_rec_movies(user_id)

        if rec_movies is not None:
            history_exposure_movies = history_exposure_movies.union(set(rec_movies))

        hot_movie = diff_and_sort(hot_movie, history_exposure_movies)

        hot_movies = hot_movie[:self.rec_num]
        return hot_movies


if __name__ == '__main__':
    rec = HotService().get_user_hot_movies_rec(1)
    print(rec)
