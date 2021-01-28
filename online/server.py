# -*- coding: utf-8 -*-

from flask import Flask, request

from online.datamanager.data_manager import DataManager
from online.service.movie_service import *
import json

app = Flask(__name__)


def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

app.after_request(after_request)

app.debug = True
manager = DataManager()
manager.load_data_to_redis()


# @app.route('/user_click_movie')
# def user_click_movie():
#     try:
#         user_id = request.args.get("user_id")
#         movie_id = request.args.get("movie_id")
#         update_user_click_movie(user_id, movie_id)
#     except Exception as e:
#         print(e)
#     return "success"


@app.route('/update_user_rec', methods=["POST"])
def update_user_rec():
    try:

        data = request.get_data()
        data = str(data, encoding="utf-8")
        user_id, movie, data = data.split("&")
        user_id = user_id.split("=")[1]
        movie_id = movie.split("=")[1]
        data = json.loads(data.split("=")[1])
        update_user_rec_info(user_id, movie_id, data)

    except Exception as e:
        print(e)
    return "success"


@app.route('/rec_movies')
def rec_movies():
    rec_movies = ""
    user_id = ""
    try:
        user_id = request.args.get("user_id")
        rec_movies = get_rec_movies(user_id)

    except Exception as e:
        print(e)
    data = [user_id, rec_movies]
    print(json.dumps(rec_movies))
    return json.dumps(data)


if __name__ == '__main__':
    # serve(app, host="0.0.0.0", port=5000)
    app.run('0.0.0.0', port=5000, debug=True)
