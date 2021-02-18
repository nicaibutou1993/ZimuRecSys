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

from offline.match.service.match_service import MatchService

match_service = MatchService()


@app.route('/match/get_match_rec_movies', methods=["GET"])
def get_match_rec_movies():
    movie_ids = []
    try:

        user_id = request.args.get("user_id")
        movie_ids = match_service.get_user_rec_movies(int(user_id))

    except Exception as e:
        print(e)
    return json.dumps(movie_ids)


if __name__ == '__main__':
    app.run('0.0.0.0', port=5001, debug=True)
