# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import os
import tensorflow as tf
from offline.convert_tf_record import dataframe_to_tf_record, tf_record_to_dataset
from sklearn.utils import shuffle

from util.config import *

pd.set_option("display.max_column", None)


class DataPreprocess(object):
    root_path = PROJECT_PATH + 'offline/data/'

    data_path = root_path + "basic_data.csv"

    user_encoder_path = root_path + "user_encoder.pkl"
    movie_encoder_path = root_path + "movie_encoder.pkl"
    release_year_encoder_path = root_path + "release_year.pkl"

    encoder_data_path = root_path + "encoder_data.csv"
    # encoder_data_path = root_path + "test.csv"

    train_tfrecord_file = root_path + "train.tfrecords"
    test_tfrecord_file = root_path + "test.tfrecords"

    """小测试集数据"""
    is_use_mini_test_data = True
    mini_test_data_path = root_path + "mini_test_data.csv"
    mini_test_tfrecord_file = root_path + "mini_test.tfrecords"

    movie_info_path = root_path + "movie_info.csv"

    def __init__(self):
        print()

    def generate_data(self, batch_size=128, is_shuffle=True, epoch=None, is_print=False):
        """获取数据train_dataset,test_dataset"""

        if self.is_use_mini_test_data:
            is_exist_test_data = os.path.exists(self.mini_test_tfrecord_file)
            test_tfrecord_file = self.mini_test_tfrecord_file
        else:
            is_exist_test_data = os.path.exists(self.test_tfrecord_file)
            test_tfrecord_file = self.test_tfrecord_file

        if not os.path.exists(self.train_tfrecord_file) or not is_exist_test_data:
            train_df, test_df = self.load_data()
            dataframe_to_tf_record(train_df, self.train_tfrecord_file)
            dataframe_to_tf_record(test_df, test_tfrecord_file)

        train_dataset = self.generate_dataset(self.train_tfrecord_file, is_shuffle, batch_size, epoch, is_print)

        test_dataset = self.generate_dataset(test_tfrecord_file,
                                             is_shuffle=False,
                                             batch_size=batch_size,
                                             epoch=None,
                                             is_print=False)

        return train_dataset, test_dataset

    def generate_test_data(self, batch_size=128):
        """获取测试集数据"""

        if self.is_use_mini_test_data:
            is_exist_test_data = os.path.exists(self.mini_test_tfrecord_file)
            test_tfrecord_file = self.mini_test_tfrecord_file
        else:
            is_exist_test_data = os.path.exists(self.test_tfrecord_file)
            test_tfrecord_file = self.test_tfrecord_file

        if not os.path.exists(self.train_tfrecord_file) or not is_exist_test_data:
            train_df, test_df = self.load_data()
            dataframe_to_tf_record(test_df, test_tfrecord_file)

        test_dataset = self.generate_dataset(test_tfrecord_file,
                                             is_shuffle=False,
                                             batch_size=batch_size,
                                             epoch=None,
                                             is_print=False)
        return test_dataset

    def generate_dataset(self, tf_record_path, is_shuffle, batch_size, epoch, is_print):
        """生成dataset"""
        dataset = tf_record_to_dataset(tf_record_path)

        if is_shuffle:
            dataset = dataset.shuffle(100000)

        dataset = dataset.batch(batch_size)

        if epoch:
            dataset = dataset.repeat()

        if is_print:

            for data in dataset.take(1):
                print(data)

        return dataset

    def load_data(self):

        if os.path.exists(self.encoder_data_path):
            data_df = pd.read_csv(self.encoder_data_path)
        else:
            data_df = self.encode_data()

        data_df = data_df[["user_id", "user_recent_click_movie_ids", "user_recent_click_labels",
                           "user_like_genres", "movie_id", "current_label",
                           "release_year", "target", "train_type"]]

        # data_df["user_recent_click_movie_ids"] = data_df["user_recent_click_movie_ids"].apply(lambda x:eval(x))
        # data_df["user_recent_click_labels"] = data_df["user_recent_click_labels"].apply(lambda x:eval(x))
        # data_df["user_like_genres"] = data_df["user_like_genres"].apply(lambda x:eval(x))

        train_df = data_df[data_df["train_type"] == "train"]

        train_df = shuffle(train_df)
        train_df.index = list(range(len(train_df)))

        train_df.pop("train_type")

        test_df = data_df[data_df["train_type"] == "test"]

        test_df.pop("train_type")

        return train_df, test_df

    def df_convert_ds(self, df, shuffle=True, batch_size=256):

        labels = df.pop("target")

        dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=128000)

        dataset = dataset.batch(batch_size)

        return dataset

    def encode_data(self):
        basic_df = pd.read_csv(self.data_path)

        # basic_df[]

        basic_df = self.encoder_user_id(basic_df)

        basic_df = self.encoder_movie_id(basic_df)

        basic_df = self.encoder_release_year(basic_df)

        basic_df.to_csv("./data/encoder_data.csv")
        return basic_df

    def encoder_user_id(self, data_df):
        if os.path.exists(self.user_encoder_path):
            id2user, user2id = pickle.load(open(self.user_encoder_path, 'rb'))
        else:
            id2user = {i: user_id for i, user_id in enumerate(set(data_df["user_id"].values))}
            user2id = {user_id: i for i, user_id in id2user.items()}
            pickle.dump((id2user, user2id), open(self.user_encoder_path, 'wb'))

        data_df["user_id"] = data_df["user_id"].apply(lambda x: user2id.get(x))

        return data_df

    def encoder_movie_id(self, data_df):
        if os.path.exists(self.movie_encoder_path):
            id2movie, movie2id = pickle.load(open(self.movie_encoder_path, 'rb'))
        else:
            id2movie = {i: movie_id for i, movie_id in enumerate(set(data_df["movie_id"].values))}
            movie2id = {movie_id: i for i, movie_id in id2movie.items()}

            pickle.dump((id2movie, movie2id), open(self.movie_encoder_path, 'wb'))

        data_df["movie_id"] = data_df["movie_id"].apply(lambda x: movie2id.get(x))

        data_df["user_recent_click_movie_ids"] = data_df["user_recent_click_movie_ids"].apply(
            lambda x: [movie2id.get(i) for i in eval(x)])
        return data_df

    def encoder_release_year(self, data_df):
        if os.path.exists(self.release_year_encoder_path):
            id2year, year2id = pickle.load(open(self.release_year_encoder_path, 'rb'))
        else:
            id2year = {i: year for i, year in enumerate(set(data_df["release_year"].values))}
            year2id = {year: i for i, year in id2year.items()}

            pickle.dump((id2year, year2id), open(self.release_year_encoder_path, 'wb'))

        data_df["release_year"] = data_df["release_year"].apply(lambda x: year2id.get(x))

        return data_df

    def data_dims(self):

        id2user, user2id = pickle.load(open(self.user_encoder_path, 'rb'))
        #
        # id2movie, movie2id = pickle.load(open(self.movie_encoder_path, 'rb'))
        #
        # id2year, year2id = pickle.load(open(self.release_year_encoder_path, 'rb'))
        #
        # print(len(id2user),len(id2movie),len(id2year))

        # data_df = pd.read_csv(self.data_path)
        #
        # data_df = data_df[data_df["user_id"] == 1]
        #
        # train_df = data_df[data_df["train_type"] == "train"]
        #
        # test_df = data_df[data_df["train_type"] == "test"]
        #
        # print(train_df)
        # print(test_df)

        get = id2user.get(1)
        print(get)

    def mini_test_data(self):

        data_df = pd.read_csv(self.encoder_data_path)

        test_df = data_df[data_df["train_type"] == "test"]

        test_df.pop("train_type")

        from joblib import Parallel, delayed
        import multiprocessing

        def topn(user_id, user_trace_df):
            user_trace_df = user_trace_df.head(50)

            print("+++++++++++")
            return user_trace_df

        def applyParallel(dfGrouped, func):
            print(multiprocessing.cpu_count())
            ret = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(name, group) for name, group in dfGrouped)
            return pd.concat(ret)

        test_df = applyParallel(test_df.groupby("user_id"), topn)

        test_df.to_csv(self.mini_test_data_path)

    def mini_test_data_to_tf_record(self):

        mini_test_df = pd.read_csv(self.mini_test_data_path)

        dataframe_to_tf_record(mini_test_df, self.mini_test_tfrecord_file)

    def get_all_movie_info(self):

        if os.path.exists(self.movie_info_path):

            movie_info_df = pd.read_csv(self.movie_info_path, index_col=0)

        else:
            data_df = pd.read_csv(self.encoder_data_path)

            data_df = data_df[["movie_id", "current_label", "release_year"]]

            movie_info_df = data_df.drop_duplicates(["movie_id"])

            movie_info_df.to_csv(self.movie_info_path)

        return movie_info_df


if __name__ == '__main__':
    DataPreprocess().get_all_movie_info()
