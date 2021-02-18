from offline.match.service.tower_service import TowerService
from offline.match.service.hot_service import HotService

from offline.match.service.real_time_service import RealTimeService


class MatchService(object):
    rec_num = 100

    def __init__(self):
        self.tower_service = TowerService(self.rec_num)

        self.hot_service = HotService()

        self.realtime_service = RealTimeService()

    def get_user_rec_movies(self, user_id):
        tower_movies = self.tower_service.get_tower_rec_movies(user_id)

        movies = set(tower_movies.keys())

        hot_movies = self.hot_service.get_hot_rec_movies(user_id)

        realtime_movies = self.realtime_service.get_real_time_match(user_id)

        movies = movies.union(set(hot_movies)).union(set(realtime_movies))

        return list(movies)


if __name__ == '__main__':
    MatchService().get_user_rec_movies(1)
