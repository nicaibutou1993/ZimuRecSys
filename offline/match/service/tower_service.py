from service.redis_service import get_user_match_feature
from offline.match.models.tower import TowerModel
from service.redis_service import get_movie_info
from offline.data_preprocess import DataPreprocess
import numpy as np
import json
import requests
import faiss
import pickle


class TowerService(object):
    tower_rec_num = 100

    def __init__(self):
        self.tower_model_cls = TowerModel()
        data_preprocess = DataPreprocess()

        self.faiss_model, self.faiss_2_movie_ids_mapping = self.load_movie_vectors_into_faiss()

        self.id2user, self.user2id = pickle.load(open(data_preprocess.user_encoder_path, 'rb'))

        self.id2movie, self.movie2id = pickle.load(open(data_preprocess.movie_encoder_path, 'rb'))


    def get_tower_rec(self, user_id):
        """双塔召回模型 推荐"""

        encoder_user_id = self.user2id.get(user_id)

        print("input user_id is :", user_id, "current encoder user id is :", encoder_user_id)

        user_vector, user_feature = self.get_user_vector(encoder_user_id)

        rec_movies = {}
        if len(user_vector) > 0:
            user_vector = user_vector.astype(np.float32)

            D, I = self.faiss_model.search(np.ascontiguousarray(user_vector), self.tower_rec_num)

            for d, i in zip(D[0], I[0]):
                movie_id = self.faiss_2_movie_ids_mapping[i]
                rec_movies[movie_id] = d

            print("filter before data len : ", len(rec_movies))
            self.filter_click_movie_ids(rec_movies,user_feature)

        return rec_movies

    def filter_click_movie_ids(self, rec_movies, user_feature):

        if len(rec_movies) > 0:
            click_movie_ids = eval(user_feature.get("click_movie_ids"))

            click_current_labels = eval(user_feature.get("click_current_labels"))

            user_like_genres = eval(user_feature.get("user_like_genres"))

            print(click_movie_ids)
            print(click_current_labels)
            print(user_like_genres)
            for click_movie_id in click_movie_ids:
                if rec_movies.__contains__(click_movie_id):
                    rec_movies.pop(click_movie_id)

            print("filter after movie nums",len(rec_movies))
            if len(rec_movies) > 0:
                rec_movies = {int(self.id2movie.get(encoder_id)): score for encoder_id, score in rec_movies.items()}

                movie_ids = list(rec_movies.keys())

                movies_info = get_movie_info(movie_ids)

                rec_movies = sorted(rec_movies.items(), key=lambda x: x[1], reverse=True)

                rec_movies = [ [movie_id,score,int(movies_info.get(movie_id)[2])-1 ]for movie_id,score in rec_movies]

                print( " ---------- rec movies ------------------")
                for i in rec_movies:
                    print(i)


    def get_user_vector(self, user_id):
        """获取用户向量"""
        user_feature = get_user_match_feature(user_id)

        user_vector = np.array([])
        if len(user_feature) > 0:
            try:
                containers = ['http://192.168.18.99:8502', 'http://192.168.18.99:8501', 'http://192.168.18.99:8503']
                container = np.random.choice(containers)
                SERVER_URL = container + '/v1/models/tower_user_model:predict'

                data = {"inputs": {"user_id": [int(user_id)],
                                   "user_recent_click_movie_ids": [
                                       eval(user_feature.get("click_movie_ids"))[-20:]],
                                   "user_recent_click_labels": [eval(user_feature.get("click_current_labels"))[-20:]],
                                   "user_like_genres": [eval(user_feature.get("user_like_genres"))]
                                   }}

                input_data_json = json.dumps(data)

                response = requests.post(SERVER_URL, data=input_data_json)
                response = json.loads(response.text)
                user_vector = np.array(response['outputs'])
            except Exception as e:
                print(e)

        return user_vector, user_feature

    def load_movie_vectors_into_faiss(self):
        """加载电影向量 到 faiss 中"""
        movie_output_vectors, movie_ids_index = self.tower_model_cls.get_movie_vectors()

        movie_output_vectors = movie_output_vectors.astype(np.float32)

        faiss_model = faiss.IndexFlatIP(self.tower_model_cls.dense_size)
        faiss_model.add(movie_output_vectors)
        return faiss_model, movie_ids_index


if __name__ == '__main__':
    tower_service = TowerService()

    tower_service.get_tower_rec(5)
