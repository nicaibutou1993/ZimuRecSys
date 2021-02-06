from offline.rank.service.fm_service import FMService
from offline.rank.service.deepfm_service import DeepFMService
from offline.rank.service.din_service import DinService



class RankService(object):

    def __init__(self):
        self.fm_service = FMService()
        self.deepfm_service = DeepFMService()
        self.din_service = DinService()



    def get_user_rec_movies(self,user_id):

        fm_rec_movies = self.fm_service.get_user_fm_rec_movies(user_id)

        deepfm_rec_movies = self.deepfm_service.get_user_deepfm_rec_movies(user_id)

        din_rec_movies = self.din_service.get_user_din_rec_movies(user_id)


        print()





if __name__ == '__main__':
    service = RankService()

    service.get_user_rec_movies(1)















