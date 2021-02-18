from offline.rank.service.fm_service import FMService
from offline.rank.service.deepfm_service import DeepFMService
from offline.rank.service.din_service import DinService
import requests

from offline.rank.service.static_fn import StaticData

from offline.rank.service.static_fn import static_movie2id


class RankService(object):
    SERVER_URL = "http://localhost:5001/match/get_match_rec_movies"

    def __init__(self):
        rec_num = 30
        self.fm_service = FMService(rec_num)
        self.deepfm_service = DeepFMService(rec_num)
        self.din_service = DinService(rec_num)

    def get_user_rec_movies(self, user_id):

        match_movie_ids = self.get_match_rec_movies(user_id)

        static_data_cls = StaticData(match_movie_ids)

        fm_rec_movies = self.fm_service.get_user_fm_rec_movies(user_id, static_data_cls)

        deepfm_rec_movies = self.deepfm_service.get_user_deepfm_rec_movies(user_id, static_data_cls)

        din_rec_movies = self.din_service.get_user_din_rec_movies(user_id, static_data_cls)

        rec_movies = {"fm": fm_rec_movies, "deepfm": deepfm_rec_movies, "din": din_rec_movies}

        return rec_movies

    def get_match_rec_movies(self, user_id):

        url = self.SERVER_URL + "?user_id=%s" % str(user_id)

        response = requests.get(url)

        movie_ids = eval(response.text)

        encode_movie_ids = []
        for movie_id in movie_ids:
            encode_movie_id = static_movie2id.get(int(movie_id))

            if encode_movie_id:
                encode_movie_ids.append(encode_movie_id)

        return encode_movie_ids


if __name__ == '__main__':
    service = RankService()

    movies = service.get_user_rec_movies(1)

    print(movies)
