from online.service.realtime_service import RealTimeService
from online.service.hot_service import HotService
from online.service.cold_movie_service import ColdMovieService
import requests
import numpy as np
from service.redis_service import get_movie_info
from collections import Counter

"""
多路推荐展示规则：

    多路数据获取：
        1.实时推荐：根据简易画像，基于刷，实时的推荐
        2.热点推荐：获取热门数据，进行推荐。
        3.冷物品推荐：获取用户简易画像，针对没有达到设置的基线的物品 根据用户画像 进行匹配，推荐。
        4.基于模型推荐：
                召回：
                    1.实时召回：基于用户简易画像
                    2.热点召回：基于热点物品
                    3.基于双塔模型：
                    4. 。。。
                    
                排序：
                    1.FM 排序  这里可以在召回使用
                    2.DeepFM 模型
                    3.Din 模型
                    4. 。。。    
                    
    多路数据排序：
    
        比如一共推荐15条数据：
            1. 统计类别数 根据类别数目 由大到小排序
            比如类别： 动漫 5条，动作4条，科幻 4条，历史 2条
            
            然后每一个类别 依次一个一个取：
            动漫 动作 科幻 历史 动漫 动作 科幻 历史  动漫 动作 科幻  动漫 动作 科幻 动漫 
            
            大类别之间的排序：
                首先根据数量排序，如果数量一致，比如动作与科幻一致，计算动作与科幻推荐的平均分 进行比较，谁大排在前面
            
            类别内部排序：
                根据 自己设定的规则，比如 {"realtime": 1, "hot": 2, "din": 3, "deepfm": 4, "fm": 5, "cold": 6}
                如果一致则 根据分数进行排序
                

"""


class RecService(object):
    SERVER_URL = "http://localhost:5002/rank/get_rank_rec_movies"

    def __init__(self):
        self.realtime_service = RealTimeService(3)
        self.hot_service = HotService(2)
        self.cold_movie_service = ColdMovieService()

        self.model_rec_num = 3

        self.sort_strategy = {"realtime": 1, "hot": 2, "din": 3, "deepfm": 4, "fm": 5, "cold": 6}

    def get_user_rec_movies(self, user_id):
        rec_movies = []

        rec_movies_info_mapping = {}

        realtime_movies = self.realtime_service.get_user_realtime_rec_movies(user_id)

        rec_movies_info_mapping.update({m: "realtime" for m in realtime_movies})

        rec_movies.extend(realtime_movies)

        hot_movies = self.hot_service.get_user_hot_movies_rec(user_id, rec_movies)

        rec_movies_info_mapping.update({m: "hot" for m in hot_movies})

        rec_movies.extend(hot_movies)

        cold_movies = self.cold_movie_service.get_cold_movie_rec(user_id)

        rec_movies_info_mapping.update({m: "cold" for m in [cold_movies]})

        rec_movies.append(cold_movies)

        model_movies = self.get_model_rec_movies(user_id)

        for model_name, movie_scores in model_movies.items():
            movies = np.array(movie_scores)[:, 0].astype(int).astype(str)

            movies = [i for i in movies if i not in rec_movies]

            print(len(movies), movies)
            movies = movies[:self.model_rec_num]

            rec_movies.extend(movies)

            rec_movies_info_mapping.update({m: model_name for m in movies})

        print(len(rec_movies))

        rec_datas = self.sort_movies(rec_movies, rec_movies_info_mapping)

        return rec_datas

    def sort_movies(self, rec_movies, rec_movies_info_mapping):
        movie_info = get_movie_info(rec_movies)

        match_2_label = {}
        match_2_score = {}

        label_matches = {}
        for match_id, info in movie_info.items():
            match_2_label[match_id] = info[2]
            match_2_score[match_id] = info[0]

            matches = label_matches.get(info[2], [])
            matches.append(match_id)
            label_matches[info[2]] = matches

        label_num_dict = dict(Counter(match_2_label.values()))

        all_data = {}

        max_num = 0
        for label, num in label_num_dict.items():
            matches = label_matches.get(label)

            bucket_sort = {}
            bucket_score = 0.0
            for match_id in matches:
                score = match_2_score.get(match_id)
                bucket_score += score

                strategy_sort = self.sort_strategy.get(rec_movies_info_mapping.get(match_id))

                if strategy_sort is None:
                    print()
                bucket_sort[match_id] = (strategy_sort, score)

            bucket_score = bucket_score / len(matches)

            bucket_sort = sorted(bucket_sort.items(), key=lambda x: (x[1][0], -x[1][1]))

            all_data[label] = (num, bucket_score, bucket_sort)

            if max_num < num:
                max_num = num

        all_data = sorted(all_data.items(), key=lambda x: (-x[1][0], -x[1][1]))

        data = []
        for d in all_data:

            match_ids = []
            for m in d[1][2]:
                match_ids.append(m[0])

            if len(match_ids) < max_num:
                for i in range(max_num - len(match_ids)):
                    match_ids.append("0")

            data.append(match_ids)

        data = np.array(data).T

        row, low = data.shape

        rec_res = []
        for i in range(row):
            for j in range(low):
                if data[i][j] != "0":
                    rec_res.append(data[i][j])

        rec_res = [[movie, int(match_2_label.get(movie))-1,
                    rec_movies_info_mapping.get(movie),
                    match_2_score.get(movie)] for movie in rec_res]
        print(rec_res)
        return rec_res

    def get_model_rec_movies(self, user_id):
        url = self.SERVER_URL + "?user_id=%s" % str(user_id)

        response = requests.get(url)

        models_rec_movies = eval(response.text)

        return models_rec_movies


if __name__ == '__main__':
    RecService().get_user_rec_movies(1)
