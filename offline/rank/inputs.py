import tensorflow as tf


def get_input_tensor_data(input_x):
    """
    模型本地化预测时候，需要加载 tensor 类型的数据
    :param input_x:
    :return:
    """

    user_id = tf.reshape(input_x.get("user_id"), (-1, 1))
    user_recent_click_movie_ids = input_x.get("user_recent_click_movie_ids")
    user_recent_click_labels = input_x.get("user_recent_click_labels")
    user_like_genres = input_x.get("user_like_genres")
    movie_id = tf.reshape(input_x.get("movie_id"), (-1, 1))
    current_label = tf.reshape(input_x.get("current_label"), (-1, 1))
    release_year = tf.reshape(input_x.get("release_year"), (-1, 1))

    data = [user_id, user_recent_click_movie_ids, user_recent_click_labels, user_like_genres, movie_id,
            current_label, release_year]

    return data


def get_input_data(input_x):
    """
    针对tf-serving 发布后，用于接口调用时候，必须是 list数据格式，只有这样 才可以序列化
    模型发布后，输入不会是名称的方式，而是 input1,input2 这种名称，
    当然也可以传入以数组的方式，但是直接传一个字典方式 是识别不了的，
    所以这里的dataset下的字典，需要重新组装成 数组 也model input 端对应"""

    user_id = tf.reshape(input_x.get("user_id"), (-1, 1)).numpy().tolist()
    user_recent_click_movie_ids = input_x.get("user_recent_click_movie_ids").numpy().tolist()
    user_recent_click_labels = input_x.get("user_recent_click_labels").numpy().tolist()
    user_like_genres = input_x.get("user_like_genres").numpy().tolist()
    movie_id = tf.reshape(input_x.get("movie_id"), (-1, 1)).numpy().tolist()
    current_label = tf.reshape(input_x.get("current_label"), (-1, 1)).numpy().tolist()
    release_year = tf.reshape(input_x.get("release_year"), (-1, 1)).numpy().tolist()

    # data = [user_id, user_recent_click_movie_ids, user_recent_click_labels, user_like_genres, movie_id,
    #         current_label, release_year]

    data = {"user_id": user_id, "user_recent_click_movie_ids": user_recent_click_movie_ids,
            "user_recent_click_labels": user_recent_click_labels, "user_like_genres": user_like_genres,
            "movie_id": movie_id, "current_label": current_label, "release_year": release_year}

    return data
