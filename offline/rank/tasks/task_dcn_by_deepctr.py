import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from offline.data_preprocess import DataPreprocess
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models.dcn import DCN
import tensorflow.keras as keras
from offline.rank.models.metrics import get_callback

"""
embedding_size = 16
测试集：acc:  [0.89912405] precision:  0.7657575757575757 recall:  0.7140435151172647 f1:  0.7389969293756398

embedding_size = 32
测试集：acc:  [0.90066246] precision:  0.7653172685445698 recall:  0.7259112743712913 f1:  0.745093616061358

"""

user_num = 4691
movie_num = 2514
year_num = 76
genre_num = 9

embedding_size = 32
epoch = 20
batch_size = 512

model_path = "checkpoint/dcn_model_by_deepctr/"

version = '0002'

data_process = DataPreprocess()
train_dataset, test_dataset = data_process.generate_data(batch_size=batch_size, epoch=epoch)

feature_columns = [SparseFeat("user_id", vocabulary_size=user_num, embedding_dim=embedding_size),
                   SparseFeat("movie_id", vocabulary_size=movie_num, embedding_dim=embedding_size),
                   SparseFeat("current_label", vocabulary_size=genre_num, embedding_dim=embedding_size),
                   SparseFeat("release_year", vocabulary_size=year_num, embedding_dim=embedding_size),
                   ]

feature_columns += [VarLenSparseFeat(
    SparseFeat("user_recent_click_movie_ids", vocabulary_size=movie_num, embedding_dim=embedding_size,
               embedding_name='movie_id'), maxlen=20),
    VarLenSparseFeat(
        SparseFeat("user_recent_click_labels", vocabulary_size=genre_num, embedding_dim=embedding_size,
                   embedding_name='current_label'), maxlen=20),
    VarLenSparseFeat(
        SparseFeat("user_like_genres", vocabulary_size=genre_num, embedding_dim=embedding_size,
                   embedding_name='current_label'), maxlen=2),
]

dnn_feature_columns = feature_columns
linear_feature_columns = feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

model = DCN(linear_feature_columns, dnn_feature_columns, task='binary')
model.summary()
model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

model.fit(train_dataset, epochs=epoch,
          steps_per_epoch=1270010 // batch_size + 1,
          callbacks=[get_callback(test_dataset, model_path, version)])
