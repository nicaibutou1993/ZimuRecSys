# -*- coding: utf-8 -*-
import pandas as pd
from util.config import *
import math
from util.redis_client import RedisClient
import json

pd.set_option("display.max_column", None)
pd.set_option("display.max_rows",None)

class DataManager():
    root_path = "E:/pycharm_project/ZimuRecSys/data/"
    users_path = root_path + "users.dat"
    movies_path = root_path + "movies.dat"
    ratings_path = root_path + "ratings.dat"

    def __init__(self):

        self.client = RedisClient.get_redis_client()

        self.movie_df = None

    def load_data_to_redis(self):
        self.user_df = self.load_user_data()
        self.movie_df = self.load_movie_data()
        rating_df = self.load_rating_data()
        self.get_genre_hot_list(rating_df)
        self.get_user_click_movie_trace(rating_df)

    def load_user_data(self):
        path = self.users_path

        user_df = pd.read_csv(path, header=None, sep="::")
        user_df.columns = ["user_id", "gender", "age", "position", "zicode"]

        return user_df

    def load_movie_data(self):
        path = self.movies_path
        movie_df = pd.read_csv(path, header=None, sep="::")
        movie_df.columns = ["movie_id", "title", "genres"]

        movie_df["release_year"] = movie_df["title"].apply(lambda x: int(x[-5:-1]))
        movie_df["title"] = movie_df["title"].apply(lambda x: (x[:-5]).strip())

        movie_df["genres"] = movie_df["genres"].apply(lambda x: x.split("|"))
        movie_df["genre"] = movie_df["genres"].apply(lambda x: x[0])

        # print(movie_df.dtypes)
        # print(movie_df.head())

        return movie_df

    def load_rating_data(self):
        path = self.ratings_path
        rating_df = pd.read_csv(path, header=None, sep="::")
        rating_df.columns = ["user_id", "movie_id", "score", "timestamp"]

        if self.movie_df is None:
            movie_df = self.load_movie_data()
        else:
            movie_df = self.movie_df

        rating_df = pd.merge(rating_df, movie_df, on="movie_id")

        rating_df["label"] = rating_df["genre"].apply(lambda x: GENRE2LABELMAP.get(x))

        #print(rating_df["label"].value_counts())
        return rating_df

    def get_genre_hot_list(self, rating_df):
        stats_df = rating_df.groupby(["label", "movie_id"], as_index=False, sort=False) \
            .agg({"score": "mean", "timestamp": "count"}) \
            .rename(columns={'score': 'avg_score', "timestamp": "click"})

        stats_df["norm_click"] = stats_df["click"].apply(lambda x: math.log(x, 5))

        stats_df["final_score"] = stats_df.apply(
            lambda item: round(item.avg_score / 5 * 0.6 + item.norm_click / 5 * 0.4, 3),
            axis=1)

        # genre_hot_dict = {}
        # for label in set(stats_df["label"].values):
        #     label_df = stats_df[stats_df["label"] == label] \
        #         .sort_values("final_score", ascending=False)
        #
        #     genre_hot_dict[label] = list(label_df["movie_id"].values)

        def hot_movies(x):
            x = x.sort_values("final_score", ascending=False)

            movie_ids = ",".join(str(i) for i in list(x["movie_id"].values))

            return movie_ids

        hot_movies_df = stats_df.groupby("label").apply(lambda x: hot_movies(x))

        hot_movies_map = dict(hot_movies_df)

        self.client.hmset(REDIS_HOT_MOVIES, hot_movies_map)

        movie_df = stats_df[["movie_id", "final_score"]]

        movie_df = pd.merge(movie_df, self.movie_df[["movie_id", "genre"]], on="movie_id")

        movie_df = movie_df.set_index(keys='movie_id')
        movie_dict = movie_df.to_dict(orient='index')
        movie_dict = {i: json.dumps([j.get("final_score"), j.get("genre")]) for i, j in movie_dict.items()}

        self.client.hmset(REDIS_MOVIE_INFO, movie_dict)

        # print(self.client.hmget(REDIS_MOVIE_SCORE,260))

    def get_user_click_movie_trace(self, rating_df):

        def trace(x):
            x = x.sort_values("timestamp")
            id_str = ",".join(str(i) for i in list(x["movie_id"].values[-20:]))

            return json.dumps([id_str, list(x["genre"].values[-20:])])

        trace_df = rating_df.groupby(['user_id']).apply(lambda x: trace(x))

        trace_map = dict(trace_df)
        self.client.hmset(REDIS_USER_TRACE, trace_map)

        hmget = self.client.hmget(REDIS_USER_TRACE, 1)
        print(trace_map.get(1))

        print(hmget)


    """计算哪些电影是冷电影，即没有曝光指定的次数"""
    def get_cold_movies(self,rating_df):
        click_df = rating_df.groupby(["movie_id","label"],as_index=False, sort=False)\
            .agg({"timestamp": "count"})\
            .rename(columns={"timestamp": "click"})

        cold_movies_df = click_df[click_df["click"] < 100]

        if not cold_movies_df.empty:

            def cold_movies_to_redis(x):
                label = x.head(1)["label"].values[0]
                x = x[["movie_id", "click"]].set_index("movie_id")
                mapping = x.to_dict(orient='index')
                mapping = {k:v.get("click")for k,v in mapping.items()}

                self.client.hmset(REDIS_COLD_MOVIES,{label:json.dumps(mapping)})

                return x

            cold_movies_df.groupby("label").apply(lambda x:cold_movies_to_redis(x))


if __name__ == '__main__':

    manager = DataManager()
    rating_data = manager.load_rating_data()
    manager.get_cold_movies(rating_data)







