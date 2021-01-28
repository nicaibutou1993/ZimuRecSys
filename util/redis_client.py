# -*- coding: utf-8 -*-

import redis

host = 'localhost'
port = 6379


class RedisClient(object):
    redis_client = None

    @staticmethod
    def get_redis_client():
        if None == RedisClient.redis_client:
            # redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            pool = redis.ConnectionPool(host=host, port=port, decode_responses=True)
            redis_client = redis.Redis(connection_pool=pool)
            RedisClient.redis_client = redis_client
        else:
            redis_client = RedisClient.redis_client
        return redis_client
