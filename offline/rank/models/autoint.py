from offline.rank.models.fm import FM
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from offline.rank.models.layer import MutiHeadSelfAttention
from itertools import combinations

"""
autoInt:

特征交叉： 使用了 mutihead self-attention 方式

看特征与特征 之间 关注度 大小

这里区别于
din:
    query: 是 当前资讯及当前资讯类别 拼接的 query  [N,1, 32], 假设两个拼接维度是32  标杆向量
    key : 是 历史浏览 资讯及前资讯类别 拼接的key  【N, 20, 32】

    attention：  【 q , k, q- k , q * k】  ---> dense1 ---> dense2 ---> [n,20,1] 这里形成对每一个历史记录的关注度
    
    [n,20,1] 20 表示 对每一条资讯的权重
    matmul([n,20,1].T,[N,20,32]) --->  【N,1,32】 表示用户对一批资讯 总体的关注  这里其实相当于 attentionPooling的作用
    


autoInt:
    序列特征处理：历史浏览资讯特征：将历史记录的Embedding 【N,20,32】 先通过 avg 方式 形成一个 【N,1,32】

    形成所有的特征 【N,10,32】 10 表示一共有10个特征，这里的特征包括 类别特征及序列特征

    然后就是每一个特征 【N,1,32】 与其他的特征 做 mutihead self-attention
    表示该特征与其他特征的关注度，这里其他就是完成了特征交叉的过程

    inputs : [N,20,32]
    q:[32, 32 * 2]  : 32 表示每一个head 参数量，2：表示一共2个head
    k:[32, 32 * 2]
    v:[32, 32 * 2]

    r:[32,32 * 2] 残差权重


    完成 self
    q = matmul(inputs,q)  --> [N,20,32 * 2] ---> [N,2,20,32]
    k = matmul(inputs,k)  --> [N,20,32 * 2] ---> [N,2,20,32]
    v = matmul(inputs,v)  --> [N,20,32 * 2] ---> [N,2,20,32]

    a = matmul(q,k.T) ---> [N,2,20,20] ----> softmax :针对最后一维 进行softmax,
        假设 第一个物品对所有物品的权重 ：【0.2,0.1，0.005,...,0.02】

    o = matmul(a,v) ---> [N,2,20,32]  ---> reshape (N,20,32 * 2)

    这里可以添加 残差：
    res = matmul(inputs,r) ---> [N,20,32 * 2]
    o = o + res

"""


class AutoInt(object):

    def __init__(self, user_num=4691, movie_num=2514, year_num=76, genre_num=9, embedding_size=16,
                 att_embedding_size=16, head_num=2, att_layer_num=3, use_res=True):
        self.user_num = user_num
        self.movie_num = movie_num
        self.year_num = year_num
        self.genre_num = genre_num
        self.embedding_size = embedding_size
        self.att_layer_num = att_layer_num

        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res

        self.fm = FM(user_num=self.user_num,
                     movie_num=self.movie_num,
                     year_num=self.year_num,
                     genre_num=self.genre_num,
                     embedding_size=self.embedding_size)

    def get_deep_output(self, inputs):
        x = Flatten()(inputs)
        x = Dense(128, activation="relu")(x)

        out = Dense(128, activation="relu")(x)

        return out

    def get_transformer_logit(self, sparse_embedding):
        inputs = Concatenate(axis=1)(sparse_embedding)
        x = inputs
        for i in range(self.att_layer_num):
            x = MutiHeadSelfAttention(self.att_embedding_size, self.head_num, self.use_res)(x)

        att_output = Flatten()(x)

        deep_output = self.get_deep_output(inputs)

        x = Concatenate()([att_output, deep_output])

        out = Dense(1, use_bias=False)(x)

        return out

    def get_autoint_model(self):
        linear_logit = self.fm.get_linear_logit()

        sparse_embedding = self.fm.get_sparse_embedding()

        cross_logit = self.get_transformer_logit(sparse_embedding)

        x = Add()([linear_logit, cross_logit])

        output = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=self.fm.inputs, outputs=output)

        return model
