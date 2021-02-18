# -*- coding: utf-8 -*-
from online.service.user_service import *
from online.service.weight_service import get_user_weight_data
from util.config import *
import math
import json

DISPLAY_MOVIE_NUM = 15
#client = RedisClient.get_redis_client()

"""
获取用户推荐电影
"""


def get_rec_movies(user_id,):
    '''
    获取用户喜欢电影风格的权重及每一个风格推荐数量
    '''
    norm_weights, genre_movie_num = get_user_weight_data(user_id)

    """用户历史浏览记录"""
    history_movies = get_user_history_rec_movies(user_id)

    rec_movies = []

    """排除已经浏览的电影"""
    for label, num in genre_movie_num.items():

        hot_movies = get_hot_movies_by_labels(label)

        hot_movies = [i for i in hot_movies if i not in history_movies]
        rec_movies += hot_movies[:num]

    set_user_history_rec_movies(user_id, rec_movies)

    movie_info = client.hmget(REDIS_MOVIE_INFO, rec_movies)

    movie_info = [json.loads(i) for i in movie_info]

    # d = dict(zip(rec_movies, movie_info))

    # movie_info = __movie_sorted(movie_info, norm_weights)
    #
    # movie = sorted(dict(zip(rec_movies, movie_info)).items(), key=lambda x: x[1][2])

    rec_movies = dict(zip(rec_movies, movie_info))

    rec_movies = __balance_weight_score(rec_movies, norm_weights)

    return rec_movies


"""
加权求和，w1 * 电影本身的评分 + w2 * 用户简易画像评分,得到最终的电影得分排序
"""


def __balance_weight_score(rec_movies, norm_weights):
    movie_data = {}
    for movie_id, movie_info in rec_movies.items():
        genre = movie_info[1]
        label = GENRE2LABELMAP.get(genre)

        movie_score = movie_info[0]

        norm_weight = norm_weights.get(label)

        score = round((movie_score + norm_weight) / 2,3)

        movie_data[movie_id] = [label,genre,score,movie_score,norm_weight]

    movie_data = sorted(movie_data.items(), key=lambda x: x[1][2], reverse=True)

    return movie_data

#
# def __movie_sorted(movie_info, weights):
#     left_bucket = {str(i) + "_" + GENRE2LABELMAP.get(j[1]): j[0] for i, j in enumerate(movie_info)}
#
#     bucket_keys = __sort_by_bucket(left_bucket, weights)
#
#     resorted = [int(k.split("_")[0]) for k in bucket_keys]
#
#     data = [[info[0], info[1], resorted.index(i)] for i, info in enumerate(movie_info)]
#
#     return data


"""
根据电影风格权重，获取每一种风格取多少条数据
暂时停用
"""


# def __get_rec_genre_movie_num(weights):
#     total = float(sum(weights.values()))
#
#     _total_num = 0
#
#     genre_movie_num = {}
#
#     for label, weight in weights.items():
#         num = math.floor(weight / total * DISPLAY_MOVIE_NUM)
#
#         _total_num += num
#
#         genre_movie_num[label] = num
#
#     left_num = DISPLAY_MOVIE_NUM - _total_num
#     if left_num > 0:
#
#         # sum_weight = sum(weights.values())
#
#         '''初始风格类别 与 简易画像 加权 '''
#         priority = {i: weights.get(i) / total + j for i, j in INIT_USER_LIKE_GENRE_PRIORITY.items()}
#
#         priority = sorted(priority.items(), key=lambda x: x[1], reverse=True)
#         labels = [k for k, v in priority]
#         # labels = sort_by_bucket(weights, INIT_USER_LIKE_GENRE_PRIORITY)
#
#         for _, label in zip(range(left_num), labels):
#             genre_movie_num[label] = genre_movie_num.get(label) + 1
#
#     return genre_movie_num
#

"""
用于两个桶排序

桶一
  —— ——— ——— 
4|     0     |
2|     1     |
1|     2     |
3|     3     |
 ——— ———— ———
 
桶二
   —— ——— ——— 
2|     0     |
3|     1     |
1|     2     |
4|     3     |
 ——— ———— ———
 
 4 ： 0 + 3 = 3
 2 ： 1 + 0 = 1
 1 ： 2 + 2 = 4
 3 ： 3 + 1 = 4
 
 最终展示 2 4 1 3
 
"""


# def __sort_by_bucket(left_bucket, right_bucket):
#     sort_map = {}
#     left_bucket = sorted(left_bucket.items(), key=lambda x: x[1], reverse=True)
#
#     left_bucket = [k for k, v in left_bucket]
#
#     right_bucket = sorted(right_bucket.items(), key=lambda x: x[1], reverse=True)
#
#     right_bucket = [k for k, v in right_bucket]
#
#     total_num = len(left_bucket)
#     for i, l in enumerate(left_bucket):
#         _l = l
#         if l.__contains__("_"):
#             _l = l.split("_")[1]
#
#         rank = i + right_bucket.index(_l) + float(i) / total_num
#         sort_map[l] = rank
#
#     buckets = sorted(sort_map.items(), key=lambda x: x[1])
#
#     bucket_values = [k for k, v in buckets]
#
#     return bucket_values


if __name__ == '__main__':
    print()
