import requests
import numpy as np

import time
time_time = time.time()
def run(name):
    #time_time = time.time()

    API_ENDPOINT = np.random.choice(
        ['http://192.168.18.99:5002/rank/fm/predict'])

    r = requests.get(url=API_ENDPOINT)

    print('......',name, time.time() - time_time)
    print(r.text)
import threading


from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=30)

for j in range(1000):
    request_list = []
    #for i in range(100):

    executor.submit(run, (j))
    print("request :",j)

    # map(task_pool.putRequest, request_list)
    # task_pool.poll()

print(time.time()-time_time)