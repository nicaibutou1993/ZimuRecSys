# -*- coding: utf-8 -*-
from service.redis_service import *
import math
from util.config import *

"""
用户冷启动方案之用户简易用户画像
针对点击 一个风格，需要反应特别敏感。
比如点击 动作类型的电影时，下一刷，其实是需要动作类型的电影的权重需要一下子 加强很多。

如果想要更平滑一点的话，那么需要计算用户 前10刷，20刷，50刷，
每一个阶段进行 加权平均，然后根据每一个类型权重 进行取值

"""

name = REDIS_CURRENT_WEIGHTS

"""一刷需要展示电影或者资讯的数量"""
DISPLAY_MOVIE_NUM = 15


"""每一个风格最低分数，防止所有类别分数都很低，然后突然点击了一个类别，会导致推荐很多条关于这个风格，不够平滑"""
basic_score = 0.3

"""根据用户获取相关权重数据，及每一个权重需要推荐的数目"""


def get_user_weight_data(user_id):
    weights = get_current_weight(user_id)
    rec_num = rec_genre_movie_num(weights)

    norm_weights = __norm_weight(weights)

    return norm_weights, rec_num


"""更新用户权重值"""


def update_user_weight_data(user_id, click_id, genre_movie_nums):
    update_weight(user_id, click_id, genre_movie_nums)


""" 这里归一化权重，这里加了 0.5 主要是为了和电影自身的评分，
    进行加和平均 ，否则 每一个类别的权重都太低了
"""


def __norm_weight(weights):
    total_weight = sum(weights.values())

    norm_weight = {l: round(float(w) / total_weight, 3) + 0.5 for l, w in weights.items()}

    return norm_weight


def update_weight(user_id, id, genre_movie_nums,last_weight=None, is_update_redis=True,is_print=True):

    if last_weight is None:
        weight = get_current_weight(user_id)
    else:
        weight = last_weight

    """打印上一轮各个类别的 权重值 经过归一化 和 没有经过归一化"""
    if is_print:
        __print_weight(weight)

    """根据归一化的权重值大小，获取每一个类别需要展示的数量"""
    # genre_movie_nums = rec_genre_movie_num(weight)

    """
    id :表示类别，1-9 是9个类别，0表示没有点击其中任何一条数据
    """

    """这里根据一次点击进行计算的，如果没有点击的类别，是会扣分的，
    正常情况下，应该是一刷进行一起计算，因为一刷是可能点击几个类别的，
    这里为了简单，就点击一个就不会点击其他的"""

    """
    总体设计：
    点击喜欢类别 会平滑增大权重，平滑增大推荐数目，
    当然如果 推荐很多条可能认为喜欢的东西时候，不怎么点击，
    那么便会根据推荐的数目越多，平滑降低就会越快"""

    if id != '0':

        """针对点击的那个类别，权重值 加0.8，这个值也不能太大，设置太大的话，
        比如一次加 2，那么会导致该类别一下子权重上升很快，
        下一次推荐该类别就会一下子很多条，不平滑
        """
        weight[id] = round(weight[id] + 0.8,3)

        """没有点击的类别是 需要奖罚的，需要减分"""
        for label, num in genre_movie_nums.items():
            if label != id:
                current_weight = weight.get(label)

                """针对当前类别如果 权重值是低于1.0 以 0.95 幂次方进行递减，
                由于如果该类别推荐的数目比较多的话，这里 pow(0.95,num) 就会降低的很快，
                这里 针对num做了一次平滑 int(math.sqrt(num)),这样速度会降低一些，
                不至于一样子就跟其他没有点击行为的类型一致了。"""
                if current_weight <= 1:
                    # weight[label] = current_weight * math.pow(0.97, num)

                    """
                    针对权重值 小于0.25，就基本不怎么降，当然这里区分下降速度，还是采取了幂次 平滑下降
                    """
                    if current_weight <= basic_score:
                        weight[label] = round(current_weight * math.pow(0.999, int(math.sqrt(num))),4)
                    else:
                        weight[label] = round(current_weight * math.pow(0.95, int(math.sqrt(num))),4)

                else:

                    """
                    这里针对 权重大于l,那么直接 0.1 * math.sqrt(max(num, 1))
                    """
                    current_weight = current_weight - math.sqrt(max(num, 1)) * 0.2
                    if current_weight <= 1:
                        current_weight = 1.0
                    weight[label] = round(current_weight,3)

    else:

        """这里针对一刷 都没有点击所有的类别数据"""

        for label, num in genre_movie_nums.items():
            current_weight = weight.get(label)
            if current_weight <= 1:

                if current_weight <= basic_score:
                    weight[label] = round(current_weight * math.pow(0.999, int(math.sqrt(num))),4)
                else:
                    weight[label] = round(current_weight * math.pow(0.95, int(math.sqrt(num))),4)
            else:
                """推荐的类别数目多，这里减分就很多，当前也不能减的太大，这里做了一个平滑  math.sqrt(max(num, 1))
                    惩罚项系数为 0.2
                """
                current_weight = current_weight - math.sqrt(max(num, 1)) * 0.2
                if current_weight <= 1:
                    current_weight = 1.0
                weight[label] = round(current_weight,4)

            # weight[label] = weight.get(label) - max(num,1) * 0.2  if weight.get(label) - max(num,1) * 0.2 > 1.0 else weight.get(label) * 0.97
    if is_print:
        print("推荐前的数量： ", sorted(genre_movie_nums.items(), key=lambda x: x[0]))
        print("推荐后的数量： ", sorted(rec_genre_movie_num(weight).items(), key=lambda x: x[0]))
        __print_weight(weight)

    if is_update_redis:
        set_current_weight(weight, user_id)

    return weight

"""打印归一化与不归一化的权重值"""
def __print_weight(weights):
    total = float(sum(weights.values()))

    _w = {l: round(float(v) / total, 3) for l, v in weights.items()}
    sorted1 = sorted(_w.items(), key=lambda x: x[0])
    print("weight : ", sorted(weights.items(), key=lambda x: x[0]))
    print("正则化后的权重 : ", sorted1)


"""根据权重，获取每一个类别能够推荐的数目"""


def rec_genre_movie_num(weights):
    total = float(sum(weights.values()))

    _total_num = 0

    genre_movie_num = {}

    for label, weight in weights.items():
        num = math.floor(weight / total * DISPLAY_MOVIE_NUM)

        if round(num / DISPLAY_MOVIE_NUM,2) > 0.4:
            num = int(0.4 * DISPLAY_MOVIE_NUM)

        _total_num += num

        genre_movie_num[label] = num

    left_num = DISPLAY_MOVIE_NUM - _total_num
    if left_num > 0:

        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        for _, weight in zip(range(left_num), sorted_weights):
            genre_movie_num[weight[0]] = genre_movie_num.get(weight[0]) + 1

    return genre_movie_num
