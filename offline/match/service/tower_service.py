from service.redis_service import get_user_match_feature
from offline.match.models.tower import TowerModel
import numpy as np
import json
import requests
import faiss


class TowerService(object):

    def __init__(self):
        self.tower_model_cls = TowerModel()

        self.faiss_model,faiss_2_movie_ids_mapping = self.load_movie_vectors_into_faiss()

    def get_tower_rec(self,user_id):
        """双塔召回模型 推荐"""

        user_vector = self.get_user_vector(user_id)

        if len(user_vector) > 0:
            user_vector = user_vector.astype(np.float32)





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
                                   "user_recent_click_movie_ids": [eval(user_feature.get("user_recent_click_movie_ids"))],
                                   "user_recent_click_labels": [eval(user_feature.get("user_recent_click_labels"))],
                                   "user_like_genres": [eval(user_feature.get("user_like_genres"))]
                                   }}

                input_data_json = json.dumps(data)

                response = requests.post(SERVER_URL, data=input_data_json)
                response = json.loads(response.text)
                user_vector = np.array(response['outputs'])
            except Exception as e:
                print(e)
        return user_vector


    def load_movie_vectors_into_faiss(self):
        """加载电影向量 到 faiss 中"""
        movie_output_vectors,movie_ids_index = self.tower_model_cls.get_movie_vectors()

        movie_output_vectors = movie_output_vectors.astype(np.float32)

        faiss_model = faiss.IndexFlatIP(self.tower_model_cls.movie_embedding_size)
        faiss_model.add(movie_output_vectors)
        return faiss_model,movie_ids_index


if __name__ == '__main__':
    get_user_vector(1)
