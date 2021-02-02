# -*- coding: utf-8 -*-
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from util.config import *
import json
import numpy as np
from online.service.weight_service import *
from collections import Counter

pd.set_option("display.max_column", None)


class DataGeneration(object):
    root_path = "data/"
    movies_path = root_path + "movies.dat"
    ratings_path = root_path + "ratings.dat"

    def __init__(self):
        self.rating_df = self.load_data()

    """加载原始数据集"""

    def load_data(self):
        path = self.movies_path
        movie_df = pd.read_csv(path, header=None, sep="::")
        movie_df.columns = ["movie_id", "title", "genres"]
        movie_df["release_year"] = movie_df["title"].apply(lambda x: int(x[-5:-1]))
        movie_df["genre"] = movie_df["genres"].apply(lambda x: x.split("|")[0])
        movie_df["label"] = movie_df["genre"].apply(lambda x: GENRE2LABELMAP.get(x))

        _movie_df = movie_df[["movie_id", "label", "release_year"]].set_index("movie_id")

        self.movie_info_dict = _movie_df.to_dict(orient="index")

        path = self.ratings_path
        rating_df = pd.read_csv(path, header=None, sep="::")
        rating_df.columns = ["user_id", "movie_id", "score", "timestamp"]


        rating_df = pd.merge(rating_df, movie_df, on="movie_id")


        #print(len(rating_df[rating_df["user_id"] == 1]))


        return rating_df

    """
    过滤冷门电影
    """

    def filter_cold_movies(self):

        movie_count_df = self.rating_df["movie_id"].value_counts()

        movie_count_df = pd.DataFrame(movie_count_df). \
            reset_index(). \
            rename(columns={"index": "movie_id", "movie_id": "click_num"})

        all_movies = set(movie_count_df["movie_id"].values)

        hot_movies = set(movie_count_df[movie_count_df["click_num"] >= 50]["movie_id"].values)

        cold_movies = all_movies - hot_movies

        print(len(cold_movies))
        print(len(hot_movies))

        return hot_movies, cold_movies

    """过滤冷门用户"""

    def filter_cold_users(self, hot_movies):

        rating_df = self.rating_df[self.rating_df["movie_id"].isin(hot_movies)]

        print(len(rating_df))

        user_count_df = rating_df["user_id"].value_counts()

        user_count_df = pd.DataFrame(user_count_df). \
            reset_index(). \
            rename(columns={"index": "user_id", "user_id": "click_num"})

        all_user_ids = set(user_count_df["user_id"].values)

        hot_user_ids = set(user_count_df[user_count_df["click_num"] >= 40]["user_id"].values)

        cold_user_ids = all_user_ids - hot_user_ids

        print(len(cold_user_ids), len(hot_user_ids))

        rating_df = rating_df[rating_df["user_id"].isin(hot_user_ids)]

        return rating_df

    def generate(self):

        hot_movies, cold_movies = self.filter_cold_movies()

        rating_df = self.filter_cold_users(hot_movies)

        self.generate_data(rating_df)

    def generate_data(self, rating_df):

        self.all_movie_ids = set(rating_df["movie_id"].values)

        def compute_user_trace(user_id, user_trace_df):

            if user_id == 1:
                print(len(user_trace_df))



            user_trace_df = user_trace_df.sort_values("timestamp")

            #print(user_trace_df.head())

            labels = list(user_trace_df["label"].values)

            movie_ids = list(user_trace_df["movie_id"].values)

            last_weights = INIT_WEIGHT
            last_finish_weight = INIT_WEIGHT

            # neg_sampling = []

            datas = []

            for i, (_, item) in enumerate(user_trace_df.iterrows()):

                current_label = item.label

                current_weights, finish_weight = self.get_user_profile(last_weights, labels[:i], current_label)

                if i >= 20 and i <= 80:
                    data = self.genrate_feature_data(user_id, item, i, last_finish_weight, movie_ids, labels,type="train")
                    datas.extend(data)
                elif i > 80:
                    data = self.genrate_feature_data(user_id, item, i, last_finish_weight, movie_ids,labels, type="test")
                    datas.extend(data)
                else:
                    pass

                last_weights = current_weights

                last_finish_weight = finish_weight

            data_df = pd.DataFrame(datas, columns=["user_id", "user_recent_click_movie_ids", "user_recent_click_labels",
                                                   "user_like_genres", "movie_id", "current_label", "release_year",
                                                   "target", "train_type"])

            #print("================================")
            return data_df

        def applyParallel(dfGrouped, func):

            print(multiprocessing.cpu_count())
            ret = Parallel(n_jobs=1)(delayed(func)(name, group) for name, group in dfGrouped)
            return pd.concat(ret)

        data_df = applyParallel(rating_df.groupby("user_id"), compute_user_trace)

        data_df.to_csv("./data/basic_data.csv")


    def genrate_feature_data(self, user_id, item, i, last_finish_weight, movie_ids,labels, type="train"):

        current_label = item.label
        movie_id = item.movie_id
        release_year = item.release_year

        row_data = []

        sorted_genres = sorted(last_finish_weight.items(), key=lambda x: x[1], reverse=True)

        user_like_genres = [int(sorted_genres[0][0]) - 1, int(sorted_genres[1][0]) - 1]

        user_recent_click_movie_ids = movie_ids[:i][-20:]
        user_recent_click_labels = labels[:i][-20:]

        user_recent_click_labels = [int(i) - 1 for i in user_recent_click_labels]


        user_tower = [user_id, user_recent_click_movie_ids, user_recent_click_labels, user_like_genres]

        movie_tower = [movie_id, int(current_label)-1, release_year]

        positive = []
        positive.extend(user_tower)
        positive.extend(movie_tower)
        positive.extend([1])
        positive.extend([type])

        row_data.append(positive)

        """负采样 一条正样本对应4条负样本"""
        for i in range(4):
            neg_match_id = np.random.choice(list(self.all_movie_ids - set(movie_ids)))

            movie_info = self.movie_info_dict.get(neg_match_id)

            neg_genre = movie_info.get("label")
            neg_release_year = movie_info.get("release_year")

            neg_movie_tower = [neg_match_id, int(neg_genre)-1, neg_release_year]

            negative = []
            negative.extend(user_tower)
            negative.extend(neg_movie_tower)
            negative.extend([0])
            negative.extend([type])
            row_data.append(negative)

        return row_data

    def get_user_profile(self, last_weights, labels, current_label):

        rec_genre_num = rec_genre_movie_num(last_weights)

        current_weights = update_weight(0, current_label, rec_genre_num, last_weights, is_update_redis=False,
                                        is_print=False)

        brush_10 = labels[-10:]
        brush_20 = labels[-20:]
        brush_50 = labels[-50:]

        brush_weight_10 = self.brush_softmax_weight(brush_10, 10)
        brush_weight_20 = self.brush_softmax_weight(brush_20, 20)
        brush_weight_50 = self.brush_softmax_weight(brush_50, 50)
        brush_weight = self.brush_finish_weight(brush_weight_10, brush_weight_20, brush_weight_50)

        """根据 最近10,20,50刷，与 之前计算的 敏感一次权重 进行 滑动平均"""
        total_weight = sum(current_weights.values())
        current_norm_weights = {l: round(float(w) / total_weight, 3) for l, w in current_weights.items()}
        current_c_weights = Counter({k: round(v * 0.6, 3) for k, v in current_norm_weights.items()})
        brush_c_weight = Counter({k: round(v * 0.4, 3) for k, v in brush_weight.items()})

        finish_weight = dict(current_c_weights + brush_c_weight)
        total_weight = sum(finish_weight.values())
        finish_weight = {l: round(float(w) / total_weight, 3) for l, w in finish_weight.items()}

        return current_weights, finish_weight

    """
    10,20,50刷,进行softmax
    """

    def brush_softmax_weight(self, brush, num=10):
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

    def brush_finish_weight(self, brush_10, brush_20, brush_50):
        weight_1, weight_2, weight_3 = 1, 0, 0
        if len(brush_20) > 0 and len(brush_50) == 0:
            weight_1, weight_2 = 0.6, 0.4

        if len(brush_20) > 0 and len(brush_50) > 0:
            weight_1, weight_2, weight_3 = 0.4, 0.3, 0.3

        brush_c_10 = Counter({k: round(v * weight_1, 3) for k, v in brush_10.items()})
        brush_c_20 = Counter({k: round(v * weight_1, 3) for k, v in brush_20.items()})
        brush_c_50 = Counter({k: round(v * weight_1, 3) for k, v in brush_50.items()})
        return dict(brush_c_10 + brush_c_20 + brush_c_50)


if __name__ == '__main__':
    time_time = time.time()
    DataGeneration().generate()

    print(time.time() - time_time)
