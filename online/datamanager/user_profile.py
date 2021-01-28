# -*- coding: utf-8 -*-
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import time
import random
from collections import Counter

pd.set_option("display.max_column", None)
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option('max_colwidth', 512)

from online.datamanager.data_manager import DataManager
from online.service.weight_service import *
import json

rating_df = DataManager().load_rating_data()

t1 = time.time()
"""简易用户画像"""

"""
10,20,50刷,进行softmax
"""


def brush_softmax_weight(brush, num=10):
    softmax_brush = {}
    if len(brush) == num:
        brush = Counter(brush)
        total = sum(brush.values())
        softmax_brush = {k: round(float(v) / total, 3) for k, v in brush.items()}
    return softmax_brush


"""
根据10,20,50刷，平滑 计算每一个类别上的权重
暂时决定：10刷权重 0.4,20刷 0.3，50刷 权重 0.3
"""


def brush_finish_weight(brush_10, brush_20, brush_50):
    weight_1, weight_2, weight_3 = 1, 0, 0
    if len(brush_20) > 0 and len(brush_50) == 0:
        weight_1, weight_2 = 0.6, 0.4

    if len(brush_20) > 0 and len(brush_50) > 0:
        weight_1, weight_2, weight_3 = 0.4, 0.3, 0.3

    brush_c_10 = Counter({k: round(v * weight_1, 3) for k, v in brush_10.items()})
    brush_c_20 = Counter({k: round(v * weight_1, 3) for k, v in brush_20.items()})
    brush_c_50 = Counter({k: round(v * weight_1, 3) for k, v in brush_50.items()})
    return dict(brush_c_10 + brush_c_20 + brush_c_50)


"""计算用户简易用户画像"""


def compute_user_simply_profile():
    def compute_user_trace(user_id, x):

        user_trace_df = x.sort_values("timestamp")

        labels = list(user_trace_df["label"].values)

        current_weights = INIT_WEIGHT

        for i, (index, item) in enumerate(user_trace_df.iterrows()):

            '''计算每一次点击，各个权重的变换'''
            user_trace_df.loc[index, "init_weights"] = json.dumps(current_weights)
            rec_genre_num = rec_genre_movie_num(current_weights)
            user_trace_df.loc[index, "rec_genre_num"] = json.dumps(rec_genre_num)

            random_int = random.randint(0, 1)
            if random_int > 0:

                for i in range(random_int):
                    label = "0"
                    current_weights = update_weight(user_id, label, rec_genre_num, current_weights,
                                                    is_update_redis=False, is_print=False)
                    rec_genre_num = rec_genre_movie_num(current_weights)

            label = item["label"]

            current_weights = update_weight(user_id, label, rec_genre_num, current_weights, is_update_redis=False,
                                            is_print=False)
            current_genre_num = rec_genre_movie_num(current_weights)

            user_trace_df.loc[index, "current_weights"] = json.dumps(current_weights)
            user_trace_df.loc[index, "current_genre_num"] = json.dumps(current_genre_num)
            user_trace_df.loc[index, "random_int"] = random_int

            '''计算 经过 10刷，20刷 50刷 等等 记录'''

            brush_10 = labels[:i][-10:]
            brush_20 = labels[:i][-20:]
            brush_50 = labels[:i][-50:]

            brush_weight_10 = brush_softmax_weight(brush_10, 10)
            brush_weight_20 = brush_softmax_weight(brush_20, 20)
            brush_weight_50 = brush_softmax_weight(brush_50, 50)
            brush_weight = brush_finish_weight(brush_weight_10, brush_weight_20, brush_weight_50)

            """根据 最近10,20,50刷，与 之前计算的 敏感一次权重 进行 滑动平均"""
            total_weight = sum(current_weights.values())
            current_norm_weights = {l: round(float(w) / total_weight, 3) for l, w in current_weights.items()}
            current_c_weights = Counter({k: round(v * 0.6, 3) for k, v in current_norm_weights.items()})
            brush_c_weight = Counter({k: round(v * 0.4, 3) for k, v in brush_weight.items()})

            finish_weight = dict(current_c_weights + brush_c_weight)
            total_weight = sum(finish_weight.values())
            finish_weight = {l: round(float(w) / total_weight, 3) for l, w in finish_weight.items()}
            finish_genre_num = rec_genre_movie_num(finish_weight)

            user_trace_df.loc[index, "brush_weight_10"] = json.dumps(brush_weight_10)
            user_trace_df.loc[index, "brush_weight_20"] = json.dumps(brush_weight_20)
            user_trace_df.loc[index, "brush_weight_50"] = json.dumps(brush_weight_50)
            user_trace_df.loc[index, "brush_weight"] = json.dumps(brush_weight)
            user_trace_df.loc[index, "finish_weight"] = json.dumps(finish_weight)
            user_trace_df.loc[index, "finish_genre_num"] = json.dumps(finish_genre_num)
        print("==============================")

        redis_client = RedisClient.get_redis_client()

        redis_client.hmset(REDIS_CURRENT_WEIGHTS, {user_id: json.dumps(current_weights)})
        redis_client.hmset(REDIS_LONG_TERM_INTEREST_50, {user_id: json.dumps(brush_weight_50)})
        redis_client.hmset(REDIS_MIDDLE_TERM_INTEREST_20, {user_id: json.dumps(brush_weight_20)})
        redis_client.hmset(REDIS_SHORT_TERM_INTEREST_10, {user_id: json.dumps(brush_weight_10)})
        redis_client.hmset(REDIS_BALANCE_TERM_INTEREST_10_20_50, {user_id: json.dumps(brush_weight)})
        redis_client.hmset(REDIS_FINISH_WEIGHTS, {user_id: json.dumps(finish_weight)})

        return user_trace_df

    for column in ["init_weights", "rec_genre_num", "current_weights",
                   "current_genre_num", "random_int",
                   "brush_weight_10", "brush_weight_20", "brush_weight_50",
                   "brush_weight", "finish_weight", "finish_genre_num"]:
        rating_df[column] = ""

    def applyParallel(dfGrouped, func):

        print(multiprocessing.cpu_count())
        ret = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(name, group) for name, group in dfGrouped)
        return pd.concat(ret)

    result = applyParallel(rating_df.groupby("user_id"), compute_user_trace)

    ##apply = rating_df.groupby("user_id").apply(lambda x: compute_user_trace(x))

    result.to_csv("E:/data/ml-1m/user_profile4.csv")
    print(result.head(100))


"""将用户点击的电影及风格 存放在redis"""


def user_click_movies_to_redis():
    def save_user_click_to_redis(x):
        user_click_df = x.sort_values("timestamp")

        user_id = str(user_click_df["user_id"].values[0])

        click_labels = list(user_click_df["label"].values)[-50:]

        click_movies = list(user_click_df["movie_id"].values)[-50:]

        redis_client = RedisClient.get_redis_client()

        redis_client.hmset(REDIS_USER_HISTORY_CLICK_MOVIES, {user_id: json.dumps([str(i) for i in click_movies])})

        redis_client.hmset(REDIS_USER_HISTORY_CLICK_GENRES, {user_id: json.dumps([str(i) for i in click_labels])})

        return x

    rating_df.groupby("user_id").apply(lambda x: save_user_click_to_redis(x))

# compute_user_simply_profile()
# user_click_movies_to_redis()
print(time.time() - t1)
