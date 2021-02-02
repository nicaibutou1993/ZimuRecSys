from util.redis_client import RedisClient
import pandas as pd
from offline.data_preprocess import DataPreprocess
import json

"""
这里数据已经编码过了，不能直接对外展示，需要转码
"""
class MatchDataManager(object):

    def __init__(self):
        self.data_preprocess = DataPreprocess()

        self.redis_client = RedisClient.get_redis_client()

    """获取用户端信息：主要是最近点击历史电影20条，最近点击20个电影的风格，用户最近的简易画像"""

    def get_user_recent_history_click_movie_trace(self):
        data_df = pd.read_csv(self.data_preprocess.encoder_data_path)

        data_df = data_df[data_df["target"] == 1]

        data_df = data_df[["user_id", "user_recent_click_movie_ids",
                           "user_recent_click_labels","user_like_genres",
                           "movie_id", "current_label",
                            "release_year", "train_type"]]

        train_df = data_df[data_df["train_type"] == "train"]

        train_df.pop("train_type")

        train_df["click_movie_ids"] = ""
        train_df["click_current_labels"] = ""

        from joblib import Parallel, delayed
        import multiprocessing

        def topn(user_id, user_trace_df):
            click_movie_ids = json.dumps([int(i) for i in list(user_trace_df["movie_id"].values)])
            click_current_labels = json.dumps([int(i) for i in list(user_trace_df["current_label"].values)])

            user_trace_df = user_trace_df.tail(1)

            user_trace_df["click_movie_ids"] = click_movie_ids
            user_trace_df["click_current_labels"] = click_current_labels
            return user_trace_df

        def applyParallel(dfGrouped, func):
            print(multiprocessing.cpu_count())
            ret = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(name, group) for name, group in dfGrouped)
            return pd.concat(ret)

        recent_df = applyParallel(train_df.groupby("user_id"), topn)

        recent_df = recent_df.set_index(keys="user_id")

        recent_dict = recent_df.to_dict(orient="index")


        recent_dict = {user_id: json.dumps(user_info) for user_id, user_info in recent_dict.items()}

        self.redis_client.hmset("match_user_recent_history_click_movie_trace", recent_dict)

        self.redis_client.hmget("match_user_recent_history_click_movie_trace", 1)
        self.redis_client.hmget("match_user_recent_history_click_movie_trace", "1")


if __name__ == '__main__':
    data_manager = MatchDataManager()
    data_manager.get_user_recent_history_click_movie_trace()

    print(data_manager.redis_client.hmget("match_user_recent_history_click_movie_trace", 1))
    print(data_manager.redis_client.hmget("match_user_recent_history_click_movie_trace", "1"))
