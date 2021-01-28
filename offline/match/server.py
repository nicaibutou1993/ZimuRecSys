# -*- coding: utf-8 -*-

from flask import Flask, request
import numpy as np
import json
import requests

app = Flask(__name__)


def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


app.after_request(after_request)

app.debug = True

from offline.match.models.tower import TowerModel

tower_model_cls = TowerModel(TowerModel.tower_mode_A)
test_dataset = tower_model_cls.get_test_dataset(batch_size=1)
#tower_user_model = tower_model.tower_user_model

@app.route('/model/tower_user_vectors', methods=["GET"])
def tower_user_vectors():
    try:

        true_y = []
        pred_y = []
        for test_x, test_y in test_dataset:
            user_output = get_user_vector(test_x)

            #movie_input_data = tower_model.get_movie_input_data(test_x)

            movie_output = tower_model_cls.get_movie_vectors(test_x)

            #print(user_output,movie_output)

            user_output = np.array(user_output)
            movie_output = np.array(movie_output)
            dot = np.sum(np.multiply(user_output, movie_output),axis=1)
            print(dot)

            vectors = tower_model_cls.search_faiss_vectors(user_output)

            break

            # _array = np.where(pred > 0.5, 1, 0)
            # pred_y.extend(_array)
            # true_y.extend(test_y.numpy()[:, np.newaxis])

        #tower_model.get_metrics(true_y,pred_y)


    except Exception as e:
        print(e)
    return ""

# def get_user_vector(test_x):
#     user_input_data = tower_model_cls.get_user_input_data(test_x)
#     containers = ['http://192.168.18.99:8502', 'http://192.168.18.99:8501', 'http://192.168.18.99:8503']
#     container = np.random.choice(containers)
#     SERVER_URL = container + '/v1/models/tower_user_model:predict'
#
#     data = {"inputs": {"user_id": user_input_data[0],
#                        "user_recent_click_movie_ids": user_input_data[1],
#                        "user_recent_click_labels": user_input_data[2],
#                        "user_like_genres": user_input_data[3]
#                        }}
#
#     input_data_json = json.dumps(data)
#
#     response = requests.post(SERVER_URL, data=input_data_json)
#     response = json.loads(response.text)
#     user_output = np.array(response['outputs'])
#
#     return user_output


@app.route('/model/tower_match', methods=["GET"])
def tower_match():
    try:

        for test_x, test_y in test_dataset:
            user_vector = get_user_vector(test_x)

            input_data = tower_model_cls.search_faiss_vectors(user_vector)

            # pred = tower_model(input_data, training=False)
            # print(pred)
            break

    except Exception as e:
        print(e)
    return ""

@app.route('/model/tower_predict', methods=["GET"])
def tower_predict():
    try:

        tower_model = tower_model_cls.tower_model
        for test_x, test_y in test_dataset:
            input_data = tower_model_cls.get_input_data(test_x)

            pred = tower_model(input_data, training=False)
            print(pred)
            break

    except Exception as e:
        print(e)
    return ""

if __name__ == '__main__':
    app.run('0.0.0.0', port=5001, debug=True)
