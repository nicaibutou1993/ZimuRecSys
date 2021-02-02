# -*- coding: utf-8 -*-
from offline.data_preprocess import DataPreprocess
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import os
import numpy as np

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True # 按需动态分配显存
# session = tf.Session(config=config)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
"""
双塔模型
"""

"""
方式一：

采用 用户端 与 电影端 分开训练，最后通过dot 内积方式 直接相乘，这种方式训练模型：

优点是： 速度快
    在模型预测的时候，已经离线 训练好了 电影的Embedding，
    到时候加载到内存中，通过faiss 向量搜索的方式 直接获得召回的电影
    
缺点： 由于用户端 与 电影端 分开训练，缺失了用户端与电影端 信息之间的融合，这样会导致 信息的减少，
想通过最后一层内积 是不能学习到电影与用户之间的复杂关系

参数：
embedding_size = 16
Dense [128,128]
验证集准确率：0.8785
Epoch 7/20
2480/2480 [==============================] - 40s 16ms/step - loss: 0.2482 - accuracy: 0.8988 - val_loss: 0.3089 - val_accuracy: 0.8785

增大参数：验证集准确率: 0.88
embedding_size = 32
Dense [256,128]

方式二：
采用用户端与电影端 合并一起训练，这样用户端与电影端 特征可以尽情的交叉
优点是：准确率 比 方式一 要高1.5% 

缺点是：用于召回的话，速度相对较慢，可以通过
方式一所用时间： 用户端 过模型的时间 + faiss 向量搜索的时间
方式二所用时间： 用户端与电影端 一起过模型，比如当前过1000电影，那么需要批量过模型很多次，速度相对慢很多。

参数：
embedding_size = 16
Dense [128,128]
验证集准确率：0.8920
Epoch 20/20
2480/2480 [==============================] - 41s 17ms/step - loss: 0.2093 - accuracy: 0.9234 - val_loss: 0.3848 - val_accuracy: 0.8920

总结：
针对召回：
    肯定是选择方式一，方式一虽然 准确率下降一点，但是速度特别快，至于准确率，交给排序模型。


用户 tf_serving 方式 在定义模型输入的时候，
必须输入name,及 dtype 类型，否则 使用tf 加载模型 会报类型错误，比如输入是int64,模型默认是 float32 类型
Input(shape=(),name = "",dtype = tf.int64)

"""


class TowerModel(object):
    epoch = 5
    user_num = 4691
    movie_num = 2514
    year_num = 76
    genre_num = 9

    batch_size = 512

    user_embedding_size = 32
    movie_embedding_size = 32
    genre_embedding_size = 32

    dense_size = 128


    is_concat_tower = False

    tower_mode_A = "tower_mode_A"
    tower_mode_B = "tower_mode_B"

    tower_user_model_name = "tower_user_model"
    tower_movie_model_name = "tower_movie_model"

    model_version = "0001"

    def __init__(self, model_name="", is_train=False):
        self.model_name = model_name
        self.is_train = is_train
        self.model_path = "E:/pycharm_project/ZimuRecSys/offline/match/models/checkpoint/tower/"
        self.data_process = DataPreprocess()

        # if not self.is_train:
        #     self.pred_model = tf.saved_model.load(self.model_path)

        if not self.is_train:
            self.tower_user_model = tf.saved_model.load(
                os.path.join(self.model_path + self.tower_user_model_name, self.model_version))

            self.tower_movie_model = tf.saved_model.load(
                os.path.join(self.model_path + self.tower_movie_model_name, self.model_version))

            self.tower_model = tf.saved_model.load(
                os.path.join(self.model_path + self.tower_mode_A, self.model_version))


    def get_model(self):
        movie_embedding = Embedding(self.movie_num, self.movie_embedding_size)
        genre_embedding = Embedding(self.genre_num, self.genre_embedding_size)

        user_model, user_concat = self.user_tower(movie_embedding, genre_embedding)
        movie_model, movie_concat = self.movie_tower(movie_embedding, genre_embedding)

        """方式二：融合用户端与电影端信息，然后丢到MLP 网络进行训练"""
        if self.tower_mode_B == self.model_name:

            x = Concatenate()([user_concat, movie_concat])

            x = Dense(256, activation="relu")(x)
            x = Dense(self.dense_size, activation="relu")(x)

            outputs = Dense(1, activation="relu")(x)

        else:
            """方式一：分开训练 用户端 与 电影端，最终通过内积的方式进行模型训练"""
            outputs = Dot(axes=1)([user_model.outputs[0], movie_model.outputs[0]])

        inputs = []
        inputs.extend(user_model.inputs)
        inputs.extend(movie_model.inputs)

        tower_model = Model(inputs=inputs, outputs=outputs)

        tower_model.compile("adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

        tower_model.summary()

        return tower_model, user_model, movie_model

    def user_tower(self, movie_embedding, genre_embedding):
        user_id_in = Input(shape=(), name="user_id", dtype=tf.int64)
        user_embedding = Embedding(self.user_num, self.user_embedding_size)(user_id_in)

        user_recent_click_movie_in = Input(shape=(20,), name="user_recent_click_movie_ids", dtype=tf.int64)
        user_recent_click_movie_embedding = movie_embedding(user_recent_click_movie_in)
        user_recent_click_movie_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=1))(
            user_recent_click_movie_embedding)

        user_recent_click_labels_in = Input(shape=(20,), name="user_recent_click_labels", dtype=tf.int64)
        user_recent_click_labels_embedding = genre_embedding(user_recent_click_labels_in)
        user_recent_click_labels_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=1))(
            user_recent_click_labels_embedding)

        user_like_genres_in = Input(shape=(2,), name="user_like_genres", dtype=tf.int64)
        user_like_genres_embedding = genre_embedding(user_like_genres_in)
        user_like_genres_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=1))(user_like_genres_embedding)

        user_concant = Concatenate()(
            [user_embedding, user_recent_click_movie_embedding, user_recent_click_labels_embedding,
             user_like_genres_embedding])

        x = Dense(self.dense_size, activation="relu")(user_concant)

        user_outputs = Dense(self.dense_size, activation="relu")(x)

        user_inputs = [user_id_in, user_recent_click_movie_in, user_recent_click_labels_in, user_like_genres_in]

        user_model = Model(inputs=user_inputs, outputs=user_outputs)

        return user_model, user_concant

    def movie_tower(self, movie_embedding, genre_embedding):
        movie_in = Input(shape=(), name="movie_id", dtype=tf.int64)
        movie_embedding = movie_embedding(movie_in)

        movie_genre_in = Input(shape=(), name="current_label", dtype=tf.int64)
        movie_genre_embedding = genre_embedding(movie_genre_in)

        release_year_in = Input(shape=(), name="release_year", dtype=tf.int64)
        year_embedding = Embedding(self.year_num, 1)(release_year_in)

        movie_concat = Concatenate()([movie_embedding, movie_genre_embedding, year_embedding])

        x = Dense(self.dense_size, activation="relu")(movie_concat)
        movie_outputs = Dense(self.dense_size, activation="relu")(x)

        movie_inputs = [movie_in, movie_genre_in, release_year_in]

        movie_model = Model(inputs=movie_inputs, outputs=movie_outputs)

        return movie_model, movie_concat

    def train(self):
        self.train_dataset, self.test_dataset = self.data_process.generate_data(batch_size=self.batch_size,
                                                                                epoch=self.epoch)
        tower_model, user_model, movie_model = self.get_model()

        tower_model.fit(self.train_dataset, epochs=20, steps_per_epoch=1270010 // self.batch_size + 1,
                        callbacks=[self.get_callback(self.test_dataset, self.model_path, user_model, movie_model)])

    def get_callback(self, test_dataset, model_path, user_model, movie_model):

        class Evaluate(keras.callbacks.Callback):

            def __init__(self, test_dataset, model_path, user_model, movie_model):
                self.test_dataset = test_dataset
                self.model_path = model_path
                self.f1 = 0.0
                self.user_model = user_model
                self.movie_model = movie_model

            def on_epoch_end(self, epoch, logs=None):

                true_y = []
                pred_y = []
                for test_x, test_y in self.test_dataset:
                    array = np.array(self.model.predict(test_x))

                    _array = np.where(array > 0.5, 1, 0)

                    pred_y.extend(_array)
                    true_y.extend(test_y.numpy()[:, np.newaxis])

                acc, precision, recall, f1 = TowerModel.get_metrics(true_y, pred_y)

                if self.f1 < f1:
                    tower_path = self.model_path + TowerModel.tower_mode_A
                    tower_user_path = self.model_path + TowerModel.tower_user_model_name
                    tower_movie_path = self.model_path + TowerModel.tower_movie_model_name

                    version = TowerModel.model_version
                    # model_name = self.model_name
                    # model_path = os.path.join(model_version, model_name)
                    tf.saved_model.save(self.model, os.path.join(tower_path, version))

                    tf.saved_model.save(self.user_model, os.path.join(tower_user_path, version))

                    tf.saved_model.save(self.movie_model, os.path.join(tower_movie_path, version))

                    self.f1 = f1

        evaluate = Evaluate(test_dataset, model_path, user_model, movie_model)

        return evaluate

    @classmethod
    def get_metrics(self, true_y, pred_y, is_print=True):
        pred_y = np.array(pred_y)
        true_y = np.array(true_y)
        acc = sum(pred_y == true_y) / float(len(pred_y))

        precision = sum(true_y[true_y == pred_y] == 1) / float(sum(pred_y == 1))

        recall = sum(true_y[true_y == pred_y] == 1) / float(sum(true_y == 1))

        f1 = 2 * precision * recall / (precision + recall)
        if is_print:
            print("acc: ", acc, "precision: ", precision, "recall: ", recall, "f1: ", f1)

        return acc, precision, recall, f1

    def get_test_dataset(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        test_dataset = self.data_process.generate_test_data(batch_size)

        return test_dataset

    def predict(self, test_dataset=None):

        if test_dataset is None:
            test_dataset = self.get_test_dataset()

        true_y = []
        pred_y = []
        for test_x, test_y in test_dataset:
            data = self.get_input_data(test_x)
            array = self.tower_user_model(data, training=False)
            _array = np.where(array > 0.5, 1, 0)
            pred_y.extend(_array)
            true_y.extend(test_y.numpy()[:, np.newaxis])

        self.get_metrics(true_y, pred_y)

    def get_input_data(self, input_x):

        """模型发布后，输入不会是名称的方式，而是 input1,input2 这种名称，
        当然也可以传入以数组的方式，但是直接传一个字典方式 是识别不了的，
        所以这里的dataset下的字典，需要重新组装成 数组 也model input 端对应"""

        user_id = input_x.get("user_id").numpy().tolist()
        user_recent_click_movie_ids = input_x.get("user_recent_click_movie_ids").numpy().tolist()
        user_recent_click_labels = input_x.get("user_recent_click_labels").numpy().tolist()
        user_like_genres = input_x.get("user_like_genres").numpy().tolist()
        movie_id = input_x.get("movie_id").numpy().tolist()
        current_label = input_x.get("current_label").numpy().tolist()
        release_year = input_x.get("release_year").numpy().tolist()

        data = [user_id, user_recent_click_movie_ids, user_recent_click_labels, user_like_genres, movie_id,
                current_label, release_year]

        return data

    def get_movie_input_data(self, input_x):

        movie_id = input_x.get("movie_id").numpy().tolist()
        current_label = input_x.get("current_label").numpy().tolist()
        release_year = input_x.get("release_year").numpy().tolist()
        data = [movie_id, current_label, release_year]

        return data

    def get_user_input_data(self, input_x):

        user_id = input_x.get("user_id").numpy().tolist()
        user_recent_click_movie_ids = input_x.get("user_recent_click_movie_ids").numpy().tolist()
        user_recent_click_labels = input_x.get("user_recent_click_labels").numpy().tolist()
        user_like_genres = input_x.get("user_like_genres").numpy().tolist()
        data = [user_id, user_recent_click_movie_ids, user_recent_click_labels, user_like_genres]

        return data

    def get_movie_vectors(self, movie_input_data = None):
        """获取电影对应的向量，及 对应的movie_id"""

        movie_output_vectors = []
        movie_ids_index = []

        if movie_input_data is None:
            movie_info = self.data_process.get_all_movie_info()
            movie_ids_index = list(movie_info["movie_id"].values)
            movie_dataset = tf.data.Dataset.from_tensor_slices(dict(movie_info))
            movie_dataset = movie_dataset.batch(500)

            for movie_input in movie_dataset:
                output = self.tower_movie_model(self.get_movie_input_data(movie_input), training=False)
                movie_output_vectors.extend(output.numpy().tolist())

        else:
            output = self.tower_movie_model(self.get_movie_input_data(movie_input_data), training=False)
            movie_output_vectors.extend(output.numpy().tolist())

        movie_output_vectors = np.array(movie_output_vectors)
        return movie_output_vectors,movie_ids_index

    def search_faiss_vectors(self,user_vector):

        movie_vectors = self.get_movie_vectors()

        movie_vectors = movie_vectors.astype(np.float32)

        import faiss

        index = faiss.IndexFlatIP(128)
        index.add(movie_vectors)

        user_vector = np.array(user_vector).astype(np.float32)

        D, I = index.search(np.ascontiguousarray(user_vector), 200)

        for d,i in zip(D[0],I[0]):

            print(d,i)





if __name__ == '__main__':
    model = TowerModel(model_name=TowerModel.tower_mode_A)
    # model.predict()
    model.train()
    #model.get_movie_vectors()
