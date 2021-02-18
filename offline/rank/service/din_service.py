import json
import requests
from offline.rank.service.static_fn import *
from offline.rank.models.metrics import get_metrics
from offline.rank.inputs import get_input_data


class DinService(object):

    containers = ['http://192.168.18.99:8506']

    model_url = '/v1/models/din_model:predict'

    def __init__(self,rec_num=100):
        self.rec_num = rec_num

    def get_user_din_rec_movies(self, user_id,static_data_cls):
        """排序 fm 推荐"""

        input_x, user_feature = static_data_cls.get_static_input_data(user_id)

        container = np.random.choice(self.containers)
        SERVER_URL = container + self.model_url

        data = {"inputs": input_x}
        input_data_json = json.dumps(data)
        response = requests.post(SERVER_URL, data=input_data_json)
        response = json.loads(response.text)
        scores = np.array(response['outputs'])

        rec_movies = sorted(zip(np.array(static_data_cls.static_movie_ids).flatten(), scores.flatten()), key=lambda x: x[1],
                            reverse=True)[:self.rec_num]

        rec_movies = dict(rec_movies)

        rec_movies = static_data_cls.filter_rec_movies_info(rec_movies, user_feature)

        return rec_movies

    def get_user_global_predict(self, test_dataset=None):

        if test_dataset is None:
            preprocess = DataPreprocess()
            test_dataset = preprocess.generate_test_data(batch_size=512)
            # test_dataset = test_dataset.take(1)

        container = np.random.choice(self.containers)
        SERVER_URL = container + self.model_url

        pred_y = []
        true_y = []
        for test_x, test_y in test_dataset:
            input_x = get_input_data(test_x)

            data = {"inputs": input_x}
            input_data_json = json.dumps(data)
            response = requests.post(SERVER_URL, data=input_data_json)
            response = json.loads(response.text)
            array = np.array(response['outputs'])

            _array = np.where(array > 0.5, 1, 0)
            pred_y.extend(_array)
            true_y.extend(test_y.numpy()[:, np.newaxis])

        get_metrics(true_y, pred_y, is_print=True)


if __name__ == '__main__':
    service = DinService()
    import time

    time_time = time.time()
    service.get_user_din_rec_movies(5)
    #service.get_user_global_predict()
    print(time_time - time.time())
