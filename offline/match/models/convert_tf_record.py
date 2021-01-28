import pandas as pd
import tensorflow as tf
from collections import OrderedDict

input_csv_file = "./data/test.csv"

output_tfrecord_file = "./data/all_train.tfrecords"


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


"""将dataframe数据集 转换成 tf-record"""
def dataframe_to_tf_record(data_df, output_path):
    #data_df = pd.read_csv(input_path)

    data_df = data_df[["user_id", "user_recent_click_movie_ids",
                       "user_recent_click_labels", "user_like_genres", "movie_id",
                       "current_label", "release_year", "target"]]

    row_count = data_df.shape[0]

    print(row_count)

    with  tf.io.TFRecordWriter(output_path) as writer:
        for i in range(row_count):
            features = OrderedDict()
            features["user_id"] = create_int_feature([data_df.iloc[i, 0]])

            """这里是sequence 序列 同 user_id 一样，只不过 user_id 需要加[] """
            features["user_recent_click_movie_ids"] = create_int_feature(eval(data_df.iloc[i, 1]))
            features["user_recent_click_labels"] = create_int_feature(eval(data_df.iloc[i, 2]))
            features["user_like_genres"] = create_int_feature(eval(data_df.iloc[i, 3]))
            features["movie_id"] = create_int_feature([data_df.iloc[i, 4]])
            features["current_label"] = create_int_feature([data_df.iloc[i, 5]])
            features["release_year"] = create_int_feature([data_df.iloc[i, 6]])
            features["target"] = create_int_feature([data_df.iloc[i, 7]])
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(record=tf_example.SerializeToString())
        writer.close()



#convert_tf_record(input_csv_file,output_tfrecord_file)


def parse_exmp(serial_exmp):
    feats = tf.io.parse_example(serial_exmp, features={
        'user_id': tf.io.FixedLenFeature([], tf.int64),
        'user_recent_click_movie_ids': tf.io.FixedLenFeature([20], tf.int64),
        'user_recent_click_labels': tf.io.FixedLenFeature([20], tf.int64),
        'user_like_genres': tf.io.FixedLenFeature([2], tf.int64),
        'movie_id': tf.io.FixedLenFeature([], tf.int64),
        'current_label': tf.io.FixedLenFeature([], tf.int64),
        'release_year': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64)
    })

    target = feats.pop("target")
    # user_recent_click_movie_ids = feats['user_recent_click_movie_ids']
    # user_recent_click_labels = feats['user_recent_click_labels']
    # user_like_genres = feats['user_like_genres']
    # movie_id = feats['movie_id']
    # current_label = feats['current_label']
    # release_year = feats['release_year']
    # target = feats['target']

    return feats,target

"""将tf_record 数据 转换成 dataset 数据格式"""
def tf_record_to_dataset(fname):
    dataset = tf.data.TFRecordDataset(fname)
    return dataset.map(parse_exmp)


if __name__ == '__main__':

    dataset = tf_record_to_dataset(output_tfrecord_file)

    dataset = dataset.shuffle(10).batch(10)

    for d in dataset.take(1):
        print(d)
