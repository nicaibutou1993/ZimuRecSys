import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, request
import numpy as np
import json
import requests

# from offline.data_preprocess import DataPreprocess
# from offline.rank.tasks.task_fm import TaskFM
# from offline.rank.tasks.task_deepfm import TaskDeepFM
# from offline.rank.tasks.task_din import TaskDIN
#
#
# fm = TaskFM()
# deepfm = TaskDeepFM()
# din = TaskDIN()
#
# preprocess = DataPreprocess()
# test_dataset = preprocess.generate_test_data(batch_size=512)
#
# test_dataset = test_dataset.take(1)

app = Flask(__name__)


def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


app.after_request(after_request)

app.debug = True

"""
15万条数据，128 batch_size  本地gpu 服务：用实际 6s
15万条数据，128 batch_size  cpu tf-serving 服务 实际 9s
15万条数据，512 batch_size  cpu tf-serving 服务 实际 7s

1个batch_size 128, 模型用时：30 ms

1个batch_size 512, gpu 模型用时：50 ms cpu: 70 ms

tf-serving 2核 8G 一台 batch_size=512 ,1秒钟 可以处理 30并发 
"""


@app.route('/rank/get_rank_rec_movies', methods=["GET"])
def get_rank_rec_movies():
    try:
        # fm.predict(test_dataset)
        print()

    except Exception as e:
        print(e)

    return "fm"






if __name__ == '__main__':
    app.run('0.0.0.0', port=5002)
